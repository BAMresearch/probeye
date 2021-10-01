# third party imports
import arviz as az
import matplotlib.pyplot as plt


def create_pair_plot(inference_data, problem, kind="kde", figsize=(9, 9),
                     textsize=10, **kwargs):
    """
    Creates a pair-plot for the given inference data using arviz.

    Parameters
    ----------
    inference_data : obj[arviz.data.inference_data.InferenceData]
        Contains the results of the sampling procedure.
    problem : obj[InferenceProblem]
        The inference problem the inference data refers to.
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
    var_names = problem.get_theta_names(tex=True)
    az.plot_pair(inference_data, var_names=var_names, marginals=True, kind=kind,
                 marginal_kwargs={'color': 'royalblue'}, figsize=figsize,
                 textsize=textsize, **kwargs)
    plt.tight_layout()
    plt.show()

def create_posterior_plot(inference_data, problem, kind="kde", figsize=(10, 3),
                          textsize=10, **kwargs):
    """
    Creates a posterior-plot for the given inference data using arviz.

    Parameters
    ----------
    inference_data : obj[arviz.data.inference_data.InferenceData]
        Contains the results of the sampling procedure.
    problem : obj[InferenceProblem]
        The inference problem the inference data refers to.
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
    var_names = problem.get_theta_names(tex=True)
    az.plot_posterior(inference_data, var_names=var_names, kind=kind,
                      figsize=figsize, textsize=textsize, **kwargs)
    plt.show()

def create_trace_plot(inference_data, problem, kind="trace", figsize=(10, 6),
                          textsize=10, **kwargs):
    """
    Creates a trace-plot for the given inference data using arviz.

    Parameters
    ----------
    inference_data : obj[arviz.data.inference_data.InferenceData]
        Contains the results of the sampling procedure.
    problem : obj[InferenceProblem]
        The inference problem the inference data refers to.
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
    var_names = problem.get_theta_names(tex=True)
    az.plot_trace(inference_data, var_names=var_names, kind=kind,
                  figsize=figsize, plot_kwargs={'textsize': textsize}, **kwargs)
    plt.show()
