# standard library imports
from typing import Union, List, Callable, TYPE_CHECKING, Any
import warnings

# third party imports
import numpy as np
import pyro.infer.mcmc.util
import torch as th
from pyro.infer import NUTS, MCMC
import arviz as az
from loguru import logger
import time
import contextlib

# local imports
from probeye.inference.torch_.priors import translate_prior_template
from probeye.inference.torch_.likelihood_models import translate_likelihood_model
from probeye.subroutines import len_or_one, make_list
from probeye.subroutines import pretty_time_delta, stream_to_logger
from probeye.subroutines import print_dict_in_rows
from probeye.subroutines import check_for_uninformative_priors

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.inference_problem import InferenceProblem
    from probeye.definition.forward_model import ForwardModelBase


class PyroSolver:
    """
    Solver routines based on pyro/torch for an InferenceProblem.

    Parameters
    ----------
    problem
        Describes the inference problem including e.g. parameters and data.
    seed
        Random state used for random number generation.
    show_progress
        When True, the progress of a solver routine will be shown (for example as a
        progress-bar) if such a feature is available. Otherwise, the progress will
        not shown.
    """

    def __init__(
        self, problem: "InferenceProblem", seed: int = 1, show_progress: bool = True
    ):

        # check that the problem does not contain a uninformative prior
        check_for_uninformative_priors(problem)

        # attributes from arguments
        self.show_progress = show_progress
        self.seed = seed

        # the following attributes will be set after the solver was run
        self.raw_results = None
        self.summary = {}  # type: dict

        # the problem is copied, and in the copy, the experimental data is reformatted
        # from numpy-arrays to torch-tensors; note that the first command makes sure
        # that also scalars are converted to numpy-arrays
        self.problem = problem.transform_experimental_data(func=np.atleast_1d)
        self.problem = self.problem.transform_experimental_data(func=th.from_numpy)

        # the dictionary dependency_dict will contain all latent parameter names as
        # keys; the value of each key will be a list with latent hyper-parameters of the
        # latent parameter's prior; note that the dependency_dict is made an attribute,
        # so that one can better test those routines
        logger.debug("Checking parameter's dependencies")
        self.dependency_dict = dict()  # type: dict
        for prm_name in self.problem.parameters.latent_prms:
            self.dependency_dict[prm_name] = []
            hyperparameters = self.problem.parameters[prm_name].prior.hyperparameters
            for prior_prm_name in hyperparameters:
                if prior_prm_name in self.problem.parameters.latent_prms:
                    self.dependency_dict[prm_name].append(prior_prm_name)

        # this makes sure that the items in dependency are in an order that they can be
        # sampled from beginning (index 0) sequentially until the last item without
        # encountering any dependency problems (i.e. that another parameter has to be
        # sampled before the considered prm can be sampled)
        consistent = False
        while not consistent:
            consistent = True
            idx_latent_dependencies = [
                i for i, v in enumerate(self.dependency_dict.values()) if len(v) > 0
            ]
            for idx in idx_latent_dependencies:
                key_idx = [*self.dependency_dict.keys()][idx]
                for dependency in self.dependency_dict[key_idx]:
                    if key_idx in self.dependency_dict[dependency]:
                        raise RuntimeError(
                            f"Found circular dependency between {key_idx} and "
                            f"{dependency}!"
                        )
                    idx_dependency = [*self.dependency_dict.keys()].index(dependency)
                    if idx_dependency > idx:
                        consistent = False
                        tuples = [*self.dependency_dict.items()]
                        tuples[idx], tuples[idx_dependency] = (
                            tuples[idx_dependency],
                            tuples[idx],
                        )
                        self.dependency_dict = dict(tuples)

        # translate the prior definitions to objects with computing capabilities
        logger.debug("Translating problem's priors")
        self.priors = {}
        for prm_name in self.dependency_dict:
            prior_template = self.problem.parameters[prm_name].prior
            self.priors[prior_template.ref_prm] = translate_prior_template(
                prior_template
            )

        # translate the general likelihood model objects into solver specific ones
        logger.debug("Translating problem's likelihood models")
        self.likelihood_models = []
        for likelihood_def in self.problem.likelihood_models.values():
            self.likelihood_models.append(translate_likelihood_model(likelihood_def))

        # translate the problem's forward models into torch compatible ones
        logger.debug("Wrapping problem's forward models")
        self.wrapped_forward_models = {}
        for fwd_model_name in self.problem.forward_models:
            self.wrapped_forward_models[fwd_model_name] = self._translate_forward_model(
                self.problem.forward_models[fwd_model_name]
            )

    @staticmethod
    def _only_values(func: Callable) -> Callable:
        """
        This function wrapper is required for the Autograd.apply function which is
        returned by translate_forward_model. While the forward model's response method
        takes a dict as input, the Autograd.apply function which will essentially take
        the forward model's response method's place requires the numeric values of inp
        given as positional arguments. This pre-processing step is done by this function
        wrapper.
        """

        def wrapper(inp: dict):
            return func(*inp.values())

        return wrapper

    def _translate_forward_model(self, forward_model: "ForwardModelBase") -> Callable:
        """
        Translates a given forward model (based on non-tensor in/outputs) to a torch-
        compatible forward model based on tensors.

        Parameters
        ----------
        forward_model
            The forward model object. Note that this object needs to have a jacobian
            method to make it translatable.
        """

        class Autograd(th.autograd.Function):
            @staticmethod
            def forward(ctx: Any, *values: th.Tensor) -> th.Tensor:  # type: ignore
                """
                Modified from the th.autograd.Function docstring: This function must
                accept a context ctx as the first argument, followed by any number of
                arguments (tensors or other types). The context can be used to store
                arbitrary data that can be then retrieved during the backward pass.

                Parameters
                ----------
                ctx
                    This context-object is for derivative-bookkeeping.
                values
                    The values of the inp-dict of the inp-argument of the forward
                    model's 'response' method in the same order.
                """

                # the idea is that the given forward model only takes numeric
                # (non-tensor) inputs; so there needs to be a conversion which takes
                # place here; also, the given values must be rearranged in the dict-
                # format required by forward_model's response method
                inp = {}
                keys = forward_model.input_channel_names
                for key, value in zip(keys, values):
                    if th.is_tensor(value):
                        inp[key] = value.detach().numpy()
                    else:
                        inp[key] = value
                    forward_model.input_structure[key] = len_or_one(inp[key])

                # evaluate the forward model and its jacobian for the given input
                # parameters; this is where the forward model is evaluated in its
                # original setup without any tensors
                response_dict = forward_model.response(inp)
                jac_dict = forward_model.jacobian(inp)
                jac_numpy = forward_model.jacobian_dict_to_array(
                    inp, jac_dict, self.problem.n_latent_prms_dim
                )

                # now we have the forward model's response in dict format; however, in
                # this format it cannot be processed here, so we will put all its
                # numeric results in one long vector
                n1 = len(forward_model.output_sensors)
                n1_times_n3, _ = jac_numpy.shape
                n3 = n1_times_n3 // n1
                response_numpy = np.zeros(n1_times_n3)
                response_structure = dict()
                for i, (key, value) in enumerate(response_dict.items()):
                    n_elements = len_or_one(value)
                    idx_start = i * n3
                    idx_end = idx_start + n_elements
                    response_numpy[idx_start:idx_end] = value
                    response_structure[key] = n_elements
                forward_model.response_structure = response_structure

                # now, we need to translate back to the tensor-world
                jac_torch = th.from_numpy(jac_numpy)
                y_torch = th.from_numpy(response_numpy)

                # the jacobian is recorded, and the model evaluation returned
                ctx.save_for_backward(jac_torch)
                return y_torch

            @staticmethod
            def backward(ctx: Any, *dl_dy_tuple: th.Tensor) -> tuple:
                """
                From the torch.autograd.Function docstring (modified): It must accept a
                context object as the first argument (ctx), followed by as many outputs
                as the 'forward' method of this class returned (None will be passed for
                non-tensor outputs of the 'forward' method), and it should return as
                many tensors, as there were inputs to 'forward' method. Each argument of
                this method (i.e., the 'backward' method) is the gradient w.r.t the
                given output, and each returned value should be the gradient w.r.t. the
                corresponding input. If an input is either not a Tensor or a Tensor not
                requiring gradients, you can just pass None as a gradient for that
                particular input.

                Parameters
                ----------
                ctx
                    This context-object is for derivative-bookkeeping.
                dl_dy_tuple
                    The jacobian of the likelihood with respect to the forward model
                    evaluations, i.e., dl/dy. Since the likelihood returns a scalar
                    value, this is a tuple of length 1.

                Returns
                -------
                    Each element i of the tuple is a tensor that represents the
                    derivative dl/dtheta_i. The order must correspond to the order in
                    the input arguments of the forward-method after the ctx argument.
                    An element is None, if it does not influence the value of the
                    likelihood function.
                """

                # this is dy/dtheta (the gradient with respect to the forward model
                # evaluation) stored from the forward pass; note that the '[0]' at the
                # end is necessary since ctx.saved_tensors is a tuple of all saved
                # tensors; here, there is only one element in it, but we still need to
                # request it
                dy_dtheta = ctx.saved_tensors[0]

                # this is dl/dtheta (the gradient of the likelihood function with
                # respect to the model parameters theta); note that dl_dy_tuple is a
                # tuple of length 1 since the likelihood is a scalar (and not a vector-
                # valued) function
                dl_dy = dl_dy_tuple[0]

                # now dl/dtheta (grad) is computed using the chain rule (which reads
                # dl/dy * dy/dtheta here)
                grad_total = th.matmul(dl_dy, dy_dtheta)

                # the returned gradients must be in tuple form; note that
                # ctx.needs_input_grad is a list with True/False elements of the same
                # lengths as there are parameters in theta; elements in theta that are
                # not being sampled (for example the value of an input sensor of the
                # forward model) are not required to have their gradients evaluated,
                # so these elements will have 'False' entries in ctx.needs_input_grad
                return_val = [None] * self.problem.n_latent_prms
                j = 0
                for i, dim in enumerate(forward_model.input_structure.values()):
                    if ctx.needs_input_grad[i]:
                        return_val[i] = grad_total[j : j + dim]  # type: ignore
                        j += dim  # type: ignore
                    else:
                        j += 1
                return tuple(return_val)

        return self._only_values(Autograd.apply)

    def evaluate_model_response(
        self, theta: th.Tensor, experiment_names: Union[str, List[str], None] = None
    ) -> dict:
        """
        Evaluates the model response for each forward model for the given parameter
        vector theta and the given experiments.

        Parameters
        ----------
        theta
            A numeric vector for which the model responses should be evaluated. Which
            parameters these numbers refer to can be checked by calling
            self.theta_explanation() once the problem is set up.
        experiment_names
            Contains the names of all or some of the experiments added to the inference
            problem. If this argument is None (which is a common use case) then all
            experiments defined in the problem (self.experiments) are used. The names
            provided here define the experiments that the fwd. model is evaluated for.

        Returns
        -------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.
        """

        # if experiments is not further specified all experiments added to the problem
        # will be accounted for when computing the model error
        if experiment_names is None:
            experiment_names = [*self.problem.experiments.keys()]
        else:
            # make sure that a given string is converted into a list
            experiment_names = make_list(experiment_names)

        # first, loop over all forward models, and then, over all experiments that are
        # associated with the corresponding model
        model_response_dict = {}
        for fwd_name, fwd_model_wrapped in self.wrapped_forward_models.items():
            forward_model = self.problem.forward_models[fwd_name]
            # get the model parameters for the considered forward model
            prms_model = self.problem.get_parameters(theta, forward_model.prms_def)
            # get all experiments referring to the considered forward model
            relevant_experiment_names = self.problem.get_experiment_names(
                forward_model_names=fwd_name, experiment_names=experiment_names
            )
            # evaluate the forward model for each relevant experiment
            for exp_name in relevant_experiment_names:
                exp_dict = self.problem.experiments[exp_name]
                # prepare the model input values from the experimental data
                sensor_values = exp_dict["sensor_values"]
                exp_inp = {
                    input_sensor.name: sensor_values[input_sensor.name]
                    for input_sensor in forward_model.input_sensors
                }
                inp = {**exp_inp, **prms_model}  # adds the two dictionaries
                # finally, evaluate the forward model for this experiment; note that the
                # additional effort here is necessary, since the wrapped forward model
                # returns a numeric vector that still has to be translated to the
                # dictionary format
                response = fwd_model_wrapped(inp)
                res = forward_model.response_structure
                i = 0
                for key in forward_model.response_structure.keys():
                    n_numbers = forward_model.response_structure[key]
                    res[key] = response[i : i + n_numbers]
                    i += n_numbers
                model_response_dict[exp_name] = res

        return model_response_dict

    def get_theta_samples(self) -> th.Tensor:
        """
        Provides a list of latent-parameter samples in form of torch.Tensors.

        Returns
        -------
            The sampled values based on the latent parameter's priors.
        """
        # even if a list is returned by this function, we initialize a dict here
        # to enable the action in the if-statement in the loop below
        pyro_parameter_samples = dict()  # type: dict
        # we have to sample each parameter for which a prior is defined
        for ref_prm, prior_obj in self.priors.items():
            hyperprms_dict = {}  # type: dict
            for name in prior_obj.hyperparameters.keys():
                if name in pyro_parameter_samples:
                    hyperprms_dict[name] = pyro_parameter_samples[name]
                else:
                    # this is the default case, where the prior's parameters
                    # (i.e. the hyperparameters) are simply constants
                    hyperprms_dict[name] = self.problem.parameters[name].value
            # this is where the parameter's sample is generated with pyro
            sample = prior_obj.sample(hyperprms_dict)  # type: ignore
            if sample.dim() == 0:
                sample = sample.reshape(1)
            pyro_parameter_samples[ref_prm] = sample
        return th.cat(tuple(pyro_parameter_samples.values()))

    def loglike(self, theta: th.Tensor):
        """
        Evaluates the log-likelihood function of the problem at theta.

        Parameters
        ----------
        theta
            A vector of pyro.samples (i.e. tensors) for which the log-likelihood
            function should be evaluated.
        """
        # compute the contribution to the log-likelihood function for each likelihood
        # model and sum it all up
        for likelihood_model in self.likelihood_models:
            # compute the model response for the likelihood model's experiment_names
            model_response = self.evaluate_model_response(
                theta, likelihood_model.experiment_names
            )
            # get the tensors for the likelihood model's parameters
            prms_like = self.problem.get_parameters(theta, likelihood_model.prms_def)
            # evaluate the loglike-contribution for the likelihood model
            likelihood_model.sample_cond_likelihood(model_response, prms_like)

    def posterior_model(self):
        """
        Returns the sampled log-likelihood in form of a torch.Tensor.

        Returns
        -------
        ll : torch.Tensor
            The evaluated log-likelihood function for the given theta-vector.
        """
        theta = self.get_theta_samples()
        return self.loglike(theta)

    def run_mcmc(
        self,
        n_walkers: int = 1,
        n_steps: int = 300,
        n_initial_steps: int = 30,
        step_size: float = 0.1,
        **kwargs,
    ) -> az.data.inference_data.InferenceData:
        """
        Runs MCMC with NUTS kernel for the InferenceProblem the PyroSolver was
        initialized with and returns the results as an arviz InferenceData obj.

        Parameters
        ----------
        n_walkers
            Number of walkers used by the estimator.
        step_size
            The step size of the NUTS sampler.
        n_steps
            The number of steps the sampler takes.
        n_initial_steps
            The number of steps for the burn-in phase.
        kwargs
            Additional keyword arguments passed to NUTS.

        Returns
        -------
        inference_data
            Contains the results of the sampling procedure.
        """

        # log which solver is used
        logger.info(
            f"Solving problem using pyro's NUTS sampler with {n_initial_steps} "
            f"+ {n_steps} samples, ..."
        )
        logger.info(f"... {n_walkers} chains and a step size of {step_size:.3f}")
        if kwargs:
            logger.info("Additional NUTS options:")
            print_dict_in_rows(kwargs, printer=logger.info)
        else:
            logger.info("No additional NUTS options specified")

        # prepare the sampling with the requested parameters
        logger.debug("Setting up NUTS sampler")
        th.manual_seed(self.seed)
        kernel = NUTS(self.posterior_model, step_size=step_size, **kwargs)
        logger.debug("Starting sampling (warmup + main)")
        time.sleep(0.1)  # for logging; otherwise no time for new line

        # this is where the actual sampling happens
        start = time.time()
        mcmc = MCMC(
            kernel,
            num_samples=n_steps,
            warmup_steps=n_initial_steps,
            num_chains=n_walkers,
            disable_progbar=not self.show_progress,
        )
        mcmc.run()
        end = time.time()

        # the following modification of the mcmc-object is necessary in cases (that
        # occur so far only when using prior-priors) where the samples of a 1D parameter
        # are saved in a 3D tensor instead of a 2D tensor; note that only a reshape of
        # the data occurs without changing the samples themselves
        for prm_name, samples in mcmc._samples.items():
            if len(samples.shape) > 2 and self.problem.parameters[prm_name].dim == 1:
                mcmc._samples[prm_name] = th.reshape(
                    mcmc._samples[prm_name], samples.shape[:2]
                )

        # log out the results of the process
        runtime_str = pretty_time_delta(end - start)
        logger.info(
            f"Sampling of the posterior distribution completed: "
            f"{n_steps} steps and {n_walkers} chains."
        )
        logger.info(f"Total run-time (including warmup): {runtime_str}.")
        logger.info("")
        logger.info("Summary of sampling results (pyro)")
        with contextlib.redirect_stdout(stream_to_logger("INFO")):  # type: ignore
            mcmc.summary()
        logger.info("")  # empty line for visual buffer
        self.raw_results = mcmc

        # create a summary dictionary similar to the one created by EmceeSolver
        self.summary = {"mean": {}, "median": {}, "sd": {}, "q05": {}, "q95": {}}
        stat_name_map = {
            "mean": "mean",
            "median": "median",
            "std": "sd",
            "5.0%": "q05",
            "95.0%": "q95",
        }
        for prm_name, prm_summary_dict in pyro.infer.mcmc.util.summary(
            mcmc._samples
        ).items():
            for stat, val in prm_summary_dict.items():
                if stat in stat_name_map:
                    if val.numel() == 1:
                        val_conv = float(val)
                    else:
                        # this is for vector parameters
                        val_conv = val.numpy()
                    self.summary[stat_name_map[stat]][prm_name] = val_conv

        # translate the results to a common data structure and return it
        var_names = self.problem.get_theta_names(tex=False, components=False)
        var_names_tex = self.problem.get_theta_names(tex=True, components=False)
        name_dict = {
            var_name: var_name_tex
            for var_name, var_name_tex in zip(var_names, var_names_tex)
        }
        # the following warning-filter is intended to hide the DepreciationWarning that
        # is currently always raised in the arviz.from_pyro method due to using np.bool
        # instead of bool; as soon as this issue is fixed within arviz, the
        # warning-filter can be removed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inference_data = az.from_pyro(mcmc, log_likelihood=False)
        inference_data.rename(name_dict, groups="posterior", inplace=True)

        return inference_data
