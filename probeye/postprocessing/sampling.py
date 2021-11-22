# third party imports
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# local imports
from probeye.subroutines import len_or_one
from probeye.subroutines import add_index_to_tex_prm_name


def create_pair_plot(
    inference_data,
    problem,
    plot_with="arviz",
    plot_priors=True,
    focus_on_posterior=False,
    kind="kde",
    figsize=(9, 9),
    textsize=10,
    true_values=None,
    show_legends=True,
    show=True,
    **kwargs,
):
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
    true_values : None, dict, optional
        Used for plotting 'true' parameter values. Keys are the parameter names
        and values are the values that are supposed to be shown in the marginal
        plots.
    show_legends : bool, optional
        If True, legends are shown in the marginal plots. Otherwise no legends
        are included in the plot.
    show : boolean, optional
        When True, the show-method is called after creating the plot. Otherwise,
        the show-method is not called. The latter is useful, when the plot
        should be further processed.
    kwargs
        Additional keyword arguments passed to arviz' pairplot function.

    Returns
    -------
    array[matplotlib.axes._subplots.AxesSubplot]
        The subplots of the created plot.
    """

    if plot_with == "arviz":

        # set default value for kde_kwargs if not given in kwargs; note that
        # this default value is mutable, so it should not be given as a default
        # argument in create_pair_plot
        if "kde_kwargs" not in kwargs:
            kwargs["kde_kwargs"] = {
                "contourf_kwargs": {"alpha": 0},
                "contour_kwargs": {"colors": None},
            }

        # process true_values if specified
        if true_values is not None:
            reference_values = dict()
            for prm_name, value in true_values.items():
                dim = problem.parameters[prm_name].dim
                tex = problem.parameters[prm_name].tex
                if dim > 1:
                    if tex in inference_data.posterior.keys():
                        # in this case, the inference_data object contains
                        # samples of a multidimensional parameter; the
                        # corresponding reference values must be given in a
                        # dictionary <tex>\n0, <tex>\n1, <tex>\n2, etc.
                        for i in range(dim):
                            key = f"{tex}\n{i}"
                            reference_values[key] = value[i]
                    else:
                        # in this case, the samples of a multidimensional
                        # parameter have been already decomposed, so all the
                        # channels in the inference data are 1D
                        for i in range(dim):
                            key = add_index_to_tex_prm_name(tex, i + 1)
                            reference_values[key] = value[i]
                else:
                    key = tex
                    reference_values[key] = value
            kwargs["reference_values"] = reference_values
            if "reference_values_kwargs" not in kwargs:
                kwargs["reference_values_kwargs"] = {"marker": "o", "color": "red"}

        # call the main plotting routine from arviz
        axs = az.plot_pair(
            inference_data,
            marginals=True,
            kind=kind,
            figsize=figsize,
            textsize=textsize,
            show=show,
            **kwargs,
        )

        # by default, the y-axis of the first and last marginal plot have ticks,
        # tick-labels and axis-labels that are not meaningful to show on the
        # y-axis; hence, we remove them here
        for i in [0, -1]:
            axs[i, i].yaxis.set_ticks_position("none")
            axs[i, i].yaxis.set_ticklabels([])
            axs[i, i].yaxis.set_visible(False)

        # adds a reference value in each marginal plot; for some reason this is
        # not done by arviz.pair_plot when passing 'reference_values'
        if "reference_values" in kwargs:
            reference_values_kwargs = None
            if "reference_values_kwargs" in kwargs:
                reference_values_kwargs = kwargs["reference_values_kwargs"]
            ref_value_list = [*kwargs["reference_values"].values()]
            for i, prm_value in enumerate(ref_value_list):
                axs[i, i].scatter(
                    prm_value,
                    0,
                    label="true value",
                    zorder=10,
                    **reference_values_kwargs,
                    edgecolor="black",
                )

        if plot_priors:

            # add the prior-pdfs to the marginal subplots
            prm_names = problem.get_theta_names(tex=False, components=False)
            i = 0  # not included in for-header due to possible dim-jumps
            for prm_name in prm_names:
                # for multivariate priors, no priors are plotted
                if problem.parameters[prm_name].dim > 1:
                    i += problem.parameters[prm_name].dim
                    continue
                x = None
                if focus_on_posterior:
                    x_min, x_max = axs[i, i].get_xlim()
                    x = np.linspace(x_min, x_max, 200)
                # the following code adds labels to the prior and posterior plot
                # if they are represented as lines
                if axs[i, i].lines:
                    posterior_handle = [axs[i, i].lines[0]]
                    posterior_label = ["posterior"]
                else:
                    # this is for the case, when the posterior is not shown as
                    # a line, but for example as a histogram etc.
                    posterior_handle, posterior_label = [], []
                problem.parameters[prm_name].prior.plot(
                    axs[i, i], problem.parameters, x=x
                )
                if show_legends:
                    prior_handle, prior_label = axs[i, i].get_legend_handles_labels()
                    axs[i, i].legend(
                        posterior_handle + prior_handle,
                        posterior_label + prior_label,
                        loc="upper right",
                    )
                i += 1

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
        else:

            # the following code adds legends to the marginal plots for the case
            # where no priors are supposed to be plotted
            if show_legends:
                prm_names = problem.get_theta_names(tex=False, components=True)
                for i, prm_name in enumerate(prm_names):
                    existing_handles, existing_labels = axs[
                        i, i
                    ].get_legend_handles_labels()
                    if axs[i, i].lines:
                        posterior_handle = [axs[i, i].lines[0]]
                        posterior_label = ["posterior"]
                    else:
                        # this is for the case, when the posterior is not shown
                        # as a line, but for example as a histogram etc.
                        posterior_handle, posterior_label = [], []
                    axs[i, i].legend(
                        posterior_handle + existing_handles,
                        posterior_label + existing_labels,
                        loc="upper right",
                    )

        # the following command reduces the otherwise wide margins
        plt.tight_layout()

        return axs

    elif plot_with == "seaborn":
        raise NotImplementedError(
            "The plot-creation with seaborn has not been implemented yet."
        )

    elif plot_with == "matplotlib":
        raise NotImplementedError(
            "The plot-creation with matplotlib has not been implemented yet."
        )

    else:
        raise RuntimeError(
            f"Invalid 'plot_with' argument: '{plot_with}'. Available options "
            f"are currently 'arviz', 'seaborn', 'matplotlib'"
        )


def create_posterior_plot(
    inference_data,
    problem,
    plot_with="arviz",
    kind="hist",
    figsize=(10, 3),
    textsize=10,
    hdi_prob=0.95,
    true_values=None,
    show=True,
    **kwargs,
):
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
    hdi_prob : float, optional
        Defines the highest density interval. Must be a number between 0 and 1.
    true_values : None, dict, optional
        Used for plotting 'true' parameter values. Keys are the parameter names
        and values are the values that are supposed to be shown in the marginal
        plots.
    show : boolean, optional
        When True, the show-method is called after creating the plot. Otherwise,
        the show-method is not called. The latter is useful, when the plot
        should be further processed.
    kwargs
        Additional keyword arguments passed to arviz' plot_posterior function.

    Returns
    -------
    array[matplotlib.axes._subplots.AxesSubplot]
        The subplots of the created plot.
    """

    if plot_with == "arviz":

        # process true_values if specified
        if true_values is not None:
            var_names_raw = problem.get_theta_names(tex=False)
            ref_val = []
            for var_name in var_names_raw:
                if len_or_one(true_values[var_name]) == 1:
                    ref_val.append(true_values[var_name])
                else:
                    for true_value in true_values[var_name]:
                        ref_val.append(true_value)
            kwargs["ref_val"] = ref_val

        # call the main plotting routine from arviz and return the axes object
        return az.plot_posterior(
            inference_data,
            kind=kind,
            figsize=figsize,
            textsize=textsize,
            hdi_prob=hdi_prob,
            show=show,
            **kwargs,
        )

    elif plot_with == "seaborn":
        raise NotImplementedError(
            "The plot-creation with seaborn has not been implemented yet."
        )

    elif plot_with == "matplotlib":
        raise NotImplementedError(
            "The plot-creation with matplotlib has not been implemented yet."
        )

    else:
        raise RuntimeError(
            f"Invalid 'plot_with' argument: '{plot_with}'. Available options "
            f"are currently 'arviz', 'seaborn', 'matplotlib'"
        )


def create_trace_plot(
    inference_data,
    problem,
    plot_with="arviz",
    kind="trace",
    figsize=(10, 6),
    textsize=10,
    show=True,
    **kwargs,
):
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
    show : boolean, optional
        When True, the show-method is called after creating the plot. Otherwise,
        the show-method is not called. The latter is useful, when the plot
        should be further processed.
    kwargs
        Additional keyword arguments passed to arviz' plot_posterior function.

    Returns
    -------
    array[matplotlib.axes._subplots.AxesSubplot]
        The subplots of the created plot.
    """

    if plot_with == "arviz":

        # set default value for kde_kwargs if not given in kwargs; note that
        # this default value is mutable, so it should not be given as a default
        # argument in create_pair_plot
        if "plot_kwargs" not in kwargs:
            kwargs["plot_kwargs"] = {"textsize": textsize}

        # call the main plotting routine from arviz and return the axes object
        return az.plot_trace(
            inference_data, kind=kind, figsize=figsize, show=show, **kwargs
        )

    elif plot_with == "seaborn":
        raise NotImplementedError(
            "The plot-creation with seaborn has not been implemented yet."
        )

    elif plot_with == "matplotlib":
        raise NotImplementedError(
            "The plot-creation with matplotlib has not been implemented yet."
        )

    else:
        raise RuntimeError(
            f"Invalid 'plot_with' argument: '{plot_with}'. Available options "
            f"are currently 'arviz', 'seaborn', 'matplotlib'"
        )
