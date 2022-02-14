(sec:components)=

Components
**********

In order to provide a valid definition of an inference problem four fundamental components are required:

1. Parameters
2. Forward models
3. Experiments
4. Likelihood models

These four components have to be defined by the user by `adding` them to the problem. Consequently, the corresponding code looks like

.. code-block:: python

    from probeye.definition import InferenceProblem

    # initialize the problem
    problem = InferenceProblem("MyProblem")

    # add the four components
    problem.add_parameter(...)
    problem.add_forward_model(...)
    problem.add_experiment(...)
    problem.add_likelihood_model(...)

Of course the dots in parenthesis :code:`(...)` still need to be further specified (according to the problem at hand), but the fundamental structure of how to define an inference problem is given by the code block above. It should be pointed out that of each component multiple instances can be added - for example five parameters, two forward models, one hundred experiments and ten likelihood models - but at least one instance of each is required to obtain a valid problem definition. Each of these components are explained in more detail in the following sections.

Parameters
##########
In probeye, an inference problem is understood as a parameter estimation problem. Hence, it comes at no surprise that you need to define at least one parameter that should be inferred. After initializing an inference problem, adding the parameters to the problem is the natural next step. In principle, you could also add the experiments first, but we recommend to begin with the parameters, because problem definitions are more readable like that.

Latent and constant parameters
------------------------------
Generally, two kinds of parameters are distinguished in probeye: latent and constant parameters. This property is also referred to as the parameter's role. Latent parameters are parameters that should be inferred, while constant parameters have a pre-defined value, and will hence not be inferred in the inference step. Earlier, we pointed out that the definition of at least one parameter is required for a valid problem definition. Now we should state more precisely: the definition of at least one latent parameter is required for a valid problem definition.

A typical definition of a latent parameter looks like this:

.. code-block:: python
    :emphasize-lines: 4

    problem.add_parameter(
        "a",
        "model",
        prior=("normal", {"loc": 1.0, "scale": 2.0}),
        tex="$a$",
        info="Slope of the fitted function",
    )

And a typical definition of a constant parameter looks like this:

.. code-block:: python
    :emphasize-lines: 4

    problem.add_parameter(
        "sigma_meas",
        "likelihood",
        const=0.1,
        tex="r$\sigma_\mathrm{meas}$",
        info="Standard deviation of measurement error",
    )

As you can see, the definition of either a latent or a constant parameter is triggered by using the :code:`prior` or the :code:`const` keyword argument in the :code:`add_parameter`-method. The :code:`const` keyword argument can be a scalar like in the example above or a numpy-based vector, for example :code:`const=np.array([0.9, -0.3])`. The :code:`prior` keyword argument on the other hand has to be given as a pair, where the first element states the kind of distribution (possible options are currently :code:`"normal"` and :code:`"uniform"`), and the second argument is a dictionary stating the prior's parameters.

Prior definition of latent parameters
-------------------------------------
As described above, when defining a latent parameter, one has to provide a 2-tuple which first contains a string describing the parameter type followed by a dictionary stating the prior's parameters and their values. The following table provides the currently implemented options.

.. list-table:: Different priors and their parameterization
    :widths: 25 25 50
    :header-rows: 1

    * - Prior type
      - Prior parameters
      - Comments
    * - "normal"
      - :code:`loc`, :code:`scale`
      - :code:`loc` refers to the mean and :code:`scale` is the standard deviation.
    * - "lognormal"
      - :code:`low`, :code:`high`
      - :code:`loc` refers to the mean and :code:`scale` is the standard deviation on the log-scale. Currently not available for pyro-solver.
    * - "truncnormal"
      - :code:`loc`, :code:`scale`, :code:`a`, :code:`b`
      - Same as for "normal", while :code:`a` and :code:`b` refer to the lower and upper bound respectively. Currently not available for pyro-solver.
    * - "uniform"
      - :code:`low`, :code:`high`
      - :code:`low` is the lower and :code:`high` is the upper bound. These bounds are inclusive.
    * - "weibull"
      - :code:`loc`, :code:`scale`, :code:`shape`
      - :code:`loc` is the lower bound. Currently not available for pyro-solver.

It should be pointed out, that it is also possible to use a latent parameter as a prior parameter. The following example may illustrate that.

.. code-block:: python

    problem.add_parameter(
        "loc_a",
        "prior",
        prior=("uniform", {"loc": -1.0, "scale": 1.0}),
        tex="r$\mu_a$",
        info="Location parameter of a's prior",
    )
    problem.add_parameter(
        "a",
        "model",
        prior=("uniform", {"loc": "loc_a", "scale": 2.0}),
        tex="$a$",
        info="Slope of the fitted function",
    )

You will notice, that instead of providing a numeric value for :code:`a`'s location parameter, the name (hence a string) of the previously defined latent parameter :code:`loc_a` is provided. Note that it is important in this example that :code:`loc_a` is defined before :code:`a`, which refers to :code:`loc_a` is defined.

Parameter's name and type
-------------------------
Each parameter (latent and constant) must have a name and a type. The parameter's name, which is given by the first argument in the :code:`add_parameter`-method,  must be unique in the scope of the problem, i.e., no other parameter can have the same name. The parameter's type, on the other hand, states where the parameter appears in the problem definition. There are three possible types :code:`model`, :code:`prior` and :code:`likelihood`. A parameter of type :code:`model` appears in one the problem's forward models, while a parameter of type :code:`prior` will be used in the definition of some latent parameter's prior. Finally, a parameter of type :code:`likelihood` will appear in one of the problem's likelihood models.

Tex and info
------------
Each parameter can (but does not have to) have a tex and an info attribute. While the tex attribute is used for plotting, the info string is used when calling a problems info-method :code:`problem.info()` printing some information on the defined problem. Even if not required, it is recommended to define both of these attributes for each parameter added to the problem.

Forward models
##############
The forward model is a parameterized simulation model (e.g. a finite element model) the predictions of which should be compared against some experimental data. The parameters of the forward model are typically the parameters which are of primary interest within the stated problem. It should be pointed out that many inference problems might contain only one forward model, but it is also possible to set up a problem that contains multiple forward models.

.. image:: images/forward_model.png
   :width: 600

In probeye, a forward model is a function that has two kinds of arguments: input sensors and parameters, see also the sketch above. While input sensors refer to specific experimental data, parameters refer to the problem's parameters. Once all input sensors and parameters are provided, the forward model computes a result that it returns via its output sensors.

In order to add a forward model to an inference problem, two steps are required. At first, the forward model has to be defined. This definition is done by setting up a new model class (that can have an arbitrary name) which is based on the probeye-class :code:`ForwardModelBase`. This class must have both a :code:`definition`-method, which defines the forward model's parameters, input sensors and output sensors, and it must have a :code:`response`-method, which describes a forward model call. The :code:`response`-method has only one input, which is a dictionary that contains both the input sensors and the parameters. The method will then perform some computations and returns its results in a dictionary of the forward model's output sensors. For a simple linear model, such a definition could look like this:

.. code-block:: python

    class LinearModel(ForwardModelBase):

            def definition(self):
                self.parameters = ["m", "b"]
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y")

            def response(self, inp: dict) -> dict:
                x = inp["x"]
                m = inp["m"]
                b = inp["b"]
                return {"y": m * x + b}

After the forward model has been defined, it must be added to the problem. For the example shown above, this would look like this:

.. code-block:: python

    # add the forward model to the problem
    problem.add_forward_model("LinearModel", LinearModel())





Experiments
###########
XXX

Likelihood models
#################
XXX