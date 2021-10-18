# standard library
import copy as cp

# third party imports
import arviz as az
import numpy as np
import matplotlib.pyplot as plt


def create_pair_plot(inference_data, problem, plot_with="arviz",
                     plot_priors=True, focus_on_posterior=False, kind="kde",
                     figsize=(9, 9), textsize=10, **kwargs):
    """
    Creates a pair-plot for the given inference data using arviz.

    Parameters
    ----------
    inference_data : obj[arviz.data.inference_data.InferenceData]
        Contains the results of the sampling procedure.
    problem : obj[InferenceProblem]
        The inference problem the inference data refers to.
    plot_with : {'arviz', 'seaborn', 'matplotlib'}, optional
        Defines the python package the plot will be generated with.
    plot_priors : bool, optional
        If True, the prior-distributions are included in the marginal subplots.
        Otherwise the priors are not shown.
    focus_on_posterior : bool, optional
        If True, the marginal plots will focus on the posteriors, i.e., the
        range of the horizontal axis will adapt to the posterior. This might
        result in just seeing a fraction of the prior distribution (if they are
        included). If False, the marginal plots will focus on the priors, which
        will have a broader x-range. If plot_priors=False, this argument has no
        effect on the generated plot.
    kind : str, optional
        Type of plot to display ('scatter', 'kde' and/or 'hexbin').
    figsize : tuple, None, optional
        Defines the size of the generated plot in the default unit. If None is
        chose, the figsize will be derived automatically.
    textsize : float, optional
        Defines the font size in the default unit.
    kwargs
        Additional keyword arguments passed to arviz' pairplot function.
    """

    if plot_with == 'arviz':

        # avoid double definition of kde_kwargs
        if 'kde_kwargs' in kwargs:
            kde_kwargs = cp.copy(kwargs['kde_kwargs'])
            del kwargs['kde_kwargs']
        else:
            kde_kwargs = {'contourf_kwargs': {"alpha": 0},
                          'contour_kwargs': {"colors": None}}

        # avoid double definition of marginal_kwargs
        if 'marginal_kwargs' in kwargs:
            marginal_kwargs = cp.copy(kwargs['marginal_kwargs'])
            del kwargs['marginal_kwargs']
        else:
            marginal_kwargs = {}

        # these names will appear on the axis labels
        var_names = problem.get_theta_names(tex=True)

        # call the main plotting routine from arviz
        axs = az.plot_pair(inference_data, var_names=var_names, marginals=True,
                           kind=kind, marginal_kwargs=marginal_kwargs,
                           kde_kwargs=kde_kwargs, figsize=figsize,
                           textsize=textsize, **kwargs)

        # by default, the y-axis of the first and last marginal plot have ticks,
        # tick-labels and axis-labels that are not meaningful to show on the
        # y-axis; hence, we remove them here
        for i in [0, -1]:
            axs[i, i].yaxis.set_ticks_position('none')
            axs[i, i].yaxis.set_ticklabels([])
            axs[i, i].yaxis.set_visible(False)

        if plot_priors:

            # add the prior-pdfs to the marginal subplots
            prm_names = problem.get_theta_names(tex=False)
            for i, prm_name in enumerate(prm_names):
                x = None
                if focus_on_posterior:
                    x_min, x_max = axs[i, i].get_xlim()
                    x = np.linspace(x_min, x_max, 200)
                # the following code adds labels to the prior and posterior plot
                # if they are represented as lines
                if axs[i, i].lines:
                    posterior_handle = [axs[i, i].lines[0]]
                    posterior_label = ['posterior']
                else:
                    posterior_handle, posterior_label = [], []
                problem.parameters[prm_name].prior.plot(
                    axs[i, i], problem.parameters, x=x)
                prior_handle, prior_label =\
                    axs[i, i].get_legend_handles_labels()
                axs[i, i].legend(posterior_handle + prior_handle,
                                 posterior_label + prior_label)
                print(type(axs[i, i]))

            # here, the axis of the non-marginal plots are adjusted to the new
            # axis ranges
            if not focus_on_posterior:
                n = len(prm_names)
                for i in range(n):
                    x_min, x_max = axs[i, i].get_xlim()
                    for j in range(i + 1, n):
                        axs[j, i].set_xlim((x_min, x_max))
                    for j in range(0, i):
                        axs[i, j].set_ylim((x_min, x_max))
        plt.tight_layout()
        plt.show()

    elif plot_with == 'seaborn':
        raise NotImplementedError(
            "The plot-creation with seaborn has not been implemented yet.")

    elif plot_with == 'matplotlib':
        raise NotImplementedError(
            "The plot-creation with matplotlib has not been implemented yet.")

    else:
        raise RuntimeError(
            f"Invalid 'plot_with' argument: '{plot_with}'. Available options "
            f"are currently 'arviz', 'seaborn', 'matplotlib'")

def create_posterior_plot(inference_data, problem, plot_with="arviz",
                          kind="kde", figsize=(10, 3), textsize=10, **kwargs):
    """
    Creates a posterior-plot for the given inference data using arviz.

    Parameters
    ----------
    inference_data : obj[arviz.data.inference_data.InferenceData]
        Contains the results of the sampling procedure.
    problem : obj[InferenceProblem]
        The inference problem the inference data refers to.
    plot_with : {'arviz', 'seaborn', 'matplotlib'}, optional
        Defines the python package the plot will be generated with.
    kind : str, optional
        Type of plot to display ('kde' or 'hist').
    figsize : tuple, None, optional
        Defines the size of the generated plot in the default unit. If None is
        chose, the figsize will be derived automatically.
    textsize : float, optional
        Defines the font size in the default unit.
    kwargs
        Additional keyword arguments passed to arviz' plot_posterior function.
    """

    if plot_with == "arviz":
        var_names = problem.get_theta_names(tex=True)
        az.plot_posterior(inference_data, var_names=var_names, kind=kind,
                          figsize=figsize, textsize=textsize, **kwargs)
        plt.show()

    elif plot_with == 'seaborn':
        raise NotImplementedError(
            "The plot-creation with seaborn has not been implemented yet.")

    elif plot_with == 'matplotlib':
        raise NotImplementedError(
            "The plot-creation with matplotlib has not been implemented yet.")

    else:
        raise RuntimeError(
            f"Invalid 'plot_with' argument: '{plot_with}'. Available options "
            f"are currently 'arviz', 'seaborn', 'matplotlib'")

def create_trace_plot(inference_data, problem, plot_with="arviz", kind="trace",
                      figsize=(10, 6), textsize=10, **kwargs):
    """
    Creates a trace-plot for the given inference data using arviz.

    Parameters
    ----------
    inference_data : obj[arviz.data.inference_data.InferenceData]
        Contains the results of the sampling procedure.
    problem : obj[InferenceProblem]
        The inference problem the inference data refers to.
    plot_with : {'arviz', 'seaborn', 'matplotlib'}, optional
        Defines the python package the plot will be generated with.
    kind: str, optional
        Allows to choose between plotting sampled values per iteration ("trace")
        and rank plots ("rank_bar", "rank_vlines").
    figsize : tuple, None, optional
        Defines the size of the generated plot in the default unit. If None is
        chose, the figsize will be derived automatically.
    textsize : float, optional
        Defines the font size in the default unit.
    kwargs
        Additional keyword arguments passed to arviz' plot_posterior function.
    """

    if plot_with == 'arviz':
        var_names = problem.get_theta_names(tex=True)
        az.plot_trace(inference_data, var_names=var_names, kind=kind,
                      figsize=figsize, plot_kwargs={'textsize': textsize},
                      **kwargs)
        plt.show()

    elif plot_with == 'seaborn':
        raise NotImplementedError(
            "The plot-creation with seaborn has not been implemented yet.")

    elif plot_with == 'matplotlib':
        raise NotImplementedError(
            "The plot-creation with matplotlib has not been implemented yet.")

    else:
        raise RuntimeError(
            f"Invalid 'plot_with' argument: '{plot_with}'. Available options "
            f"are currently 'arviz', 'seaborn', 'matplotlib'")
