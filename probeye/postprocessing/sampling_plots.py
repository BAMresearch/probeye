# standard library
from typing import Union, Optional, TYPE_CHECKING

# third party imports
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# local imports
from probeye.subroutines import len_or_one
from probeye.subroutines import add_index_to_tex_prm_name

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.inverse_problem import InverseProblem


def create_pair_plot(
    inference_data: az.data.inference_data.InferenceData,
    problem: "InverseProblem",
    plot_with: str = "arviz",
    plot_priors: bool = True,
    focus_on_posterior: bool = True,
    kind: str = "kde",
    figsize: Optional[tuple] = None,
    inches_per_row: Union[int, float] = 2.0,
    inches_per_col: Union[int, float] = 2.0,
    textsize: Union[int, float] = 10,
    title_size: Union[int, float] = 14,
    title: Optional[str] = None,
    true_values: Optional[dict] = None,
    show_legends: bool = True,
    show: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Creates a pair-plot for the given inference data.

    Parameters
    ----------
    inference_data
        Contains the results of the sampling procedure.
    problem
        The inverse problem the inference data refers to.
    plot_with
        Defines the python package the plot will be generated with. Options are:
        {'arviz', 'seaborn', 'matplotlib'}.
    plot_priors
        If True, the prior-distributions are included in the marginal subplots.
        Otherwise the priors are not shown.
    focus_on_posterior
        If True, the marginal plots will focus on the posteriors, i.e., the range of the
        horizontal axis will adapt to the posterior. This might result in just seeing a
        fraction of the prior distribution (if they are included). If False, the
        marginal plots will focus on the priors, which will have a broader x-range. If
        plot_priors=False, this argument has no effect on the generated plot.
    kind
        Type of plot to display ('scatter', 'kde' and/or 'hexbin').
    figsize
        Defines the size of the generated plot in inches. If None is chosen, the figsize
        will be derived automatically by using inches_per_row and inches_per_col.
    inches_per_row
        If figsize is None, this will specify the inches per row in the subplot-grid.
        This argument has no effect if figsize is specified.
    inches_per_col
        If figsize is None, this will specify the inches per column in the subplot-grid.
        This argument has no effect if figsize is specified.
    textsize
        Defines the font size in the default unit.
    title_size
        Defines the font size of the figures title if 'title' is given.
    title
        The title of the figure.
    true_values
        Used for plotting 'true' parameter values. Keys are the parameter names and
        values are the values that are supposed to be shown in the marginal plots.
    show_legends
        If True, legends are shown in the marginal plots. Otherwise no legends are
        included in the plot.
    show
        When True, the show-method is called after creating the plot. Otherwise, the
        show-method is not called. The latter is useful, when the plot should be further
        processed.
    kwargs
        Additional keyword arguments passed to arviz' pairplot function.

    Returns
    -------
    axs
        The array of subplots of the created plot.
    """

    # a pairplot can only be generate when there are at least two parameter or parameter
    # components (the latter refers to vector-valued parameters)
    if problem.n_latent_prms_dim == 1:
        logger.warning(
            "The combined dimension of all latent parameters is one. Hence, no "
            "pairplot can be generated in this setup."
        )
        return np.array([])

    if plot_with == "arviz":

        # set default value for kde_kwargs if not given in kwargs; note that this
        # default value is mutable, so it should not be given as a default argument in
        # create_pair_plot
        if "kde_kwargs" not in kwargs:
            kwargs["kde_kwargs"] = {
                "contourf_kwargs": {"alpha": 0},
                "contour_kwargs": {"colors": None},
            }

        if "backend_kwargs" not in kwargs:
            if problem.n_latent_prms_dim == 2:
                kwargs["backend_kwargs"] = {"constrained_layout": True}

        histograms_on_diagonal = False
        if "marginal_kwargs" in kwargs:
            if "kind" in kwargs["marginal_kwargs"]:
                if kwargs["marginal_kwargs"]["kind"] == "hist":
                    histograms_on_diagonal = True

        # process true_values if specified
        if true_values is not None:
            reference_values_unsorted = dict()
            index_list = []
            for prm_name, value in true_values.items():
                dim = problem.parameters[prm_name].dim
                tex = prm_name  # prevents tex being None
                if problem.parameters[prm_name].tex is not None:
                    tex = problem.parameters[prm_name].tex
                if dim > 1:
                    # all the channels in the inference data are 1D
                    idx_start = problem.parameters[prm_name].index
                    index_list.append(idx_start)
                    for i in range(dim):
                        key = add_index_to_tex_prm_name(tex, i + 1)
                        reference_values_unsorted[key] = value[i]
                        index_list.append(idx_start + (i + 1))
                else:
                    key = tex
                    reference_values_unsorted[key] = value
                    index_list.append(problem.parameters[prm_name].index)
            key_list_unsorted = [*reference_values_unsorted.keys()]
            key_list = [key for _, key in sorted(zip(index_list, key_list_unsorted))]
            reference_values = dict()
            for key in key_list:
                reference_values[key] = reference_values_unsorted[key]

            kwargs["reference_values"] = reference_values
            if "reference_values_kwargs" not in kwargs:
                kwargs["reference_values_kwargs"] = {"marker": "o", "color": "red"}

        # call the main plotting routine from arviz
        axs = az.plot_pair(
            inference_data,
            marginals=True,
            kind=kind,
            textsize=textsize,
            show=False,
            **kwargs,
        )

        # adds a reference value in each marginal plot; for some reason this is not done
        # by arviz.pair_plot when passing 'reference_values'
        if "reference_values" in kwargs:
            reference_values_kwargs = None
            if "reference_values_kwargs" in kwargs:
                reference_values_kwargs = kwargs["reference_values_kwargs"]
            ref_value_list = [*kwargs["reference_values"].values()]
            if problem.n_latent_prms_dim > 2:
                # in this case, the relevant axis is always the horizontal one
                for i, prm_value in enumerate(ref_value_list):
                    axs[i, i].scatter(
                        prm_value,
                        0,
                        label="true value",
                        zorder=10,
                        edgecolor="black",
                        **reference_values_kwargs,
                    )
            else:
                # in this case, the plot on the bottom right is rotated
                axs[0, 0].scatter(
                    ref_value_list[0],
                    0,
                    label="true value",
                    zorder=10,
                    **reference_values_kwargs,
                    edgecolor="black",
                )
                axs[1, 1].scatter(
                    0,
                    ref_value_list[1],
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
                    if (problem.n_latent_prms_dim == 2) and (i == 1):
                        # the plot on the bottom right is rotated
                        x_min, x_max = axs[i, i].get_ylim()
                    else:
                        x_min, x_max = axs[i, i].get_xlim()
                    x = np.linspace(x_min, x_max, 200)
                # the following code adds labels to the prior and posterior plot if they
                # are represented as lines
                if axs[i, i].lines:
                    posterior_handle = [axs[i, i].lines[0]]
                    posterior_label = ["posterior"]
                else:
                    # this is for the case, when the posterior is not shown as a line,
                    # but for example as a histogram etc.
                    posterior_handle, posterior_label = [], []
                rotate = True if problem.n_latent_prms_dim == 2 and i == 1 else False
                problem.parameters[prm_name].prior.plot(
                    axs[i, i],
                    problem.parameters,
                    x=x,
                    rotate=rotate,
                    label="prior",
                )
                # don't use the histogram bin ticks when the prior is also plotted
                if histograms_on_diagonal and not focus_on_posterior:
                    if rotate:
                        y_min, y_max = axs[i, i].get_ylim()
                        tick_list = np.linspace(y_min, y_max, 9).tolist()
                        axs[i, i].set_yticks(tick_list)
                    else:
                        x_min, x_max = axs[i, i].get_xlim()
                        tick_list = np.linspace(x_min, x_max, 9).tolist()
                        axs[i, i].set_xticks(tick_list)
                # create the legends if requested
                if show_legends:
                    prior_handle, prior_label = axs[i, i].get_legend_handles_labels()
                    axs[i, i].legend(
                        posterior_handle + prior_handle,
                        posterior_label + prior_label,
                        loc="best",
                    )
                i += 1

            # here, the axis of the non-marginal plots are adjusted to the new ranges
            if (not focus_on_posterior) and (problem.n_latent_prms_dim > 2):
                n = problem.n_latent_prms_dim
                for i in range(n):
                    # the reference is the plot on the diagonal
                    x_min, x_max = axs[i, i].get_xlim()
                    # loop over axes in the column below
                    for j in range(i + 1, n):
                        axs[j, i].set_xlim((x_min, x_max))
                    # loop over axes in the row to the left
                    for j in range(0, i):
                        axs[i, j].set_ylim((x_min, x_max))
        else:

            # the following code adds legends to the marginal plots for the case where
            # no priors are supposed to be plotted
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
                        # this is for the case, when the posterior is not shown as a
                        # line, but for example as a histogram etc.
                        posterior_handle, posterior_label = [], []
                    axs[i, i].legend(
                        posterior_handle + existing_handles,
                        posterior_label + existing_labels,
                        loc="best",
                    )
            # synchronize the axes, which is only necessary if there are at least 3
            # latent parameters; in this case of only 2 latent parameters (note that
            # only one latent parameter is not allowed for a pair plot), a slightly
            # different plot is created where the marginal plot on the right is rotated
            n = problem.n_latent_prms_dim
            if n > 2:
                for i in range(n):
                    # the reference is the plot on the diagonal
                    x_min, x_max = axs[i, i].get_xlim()
                    # loop over axes in the column below
                    for j in range(i + 1, n):
                        axs[j, i].set_xlim((x_min, x_max))
                    # loop over axes in the row to the left
                    for j in range(0, i):
                        axs[i, j].set_ylim((x_min, x_max))

        # set the figure size; this is done either automatically if the user did not
        # specify the figsize argument, or it simply sets the requested figsize
        fig = axs.ravel()[0].figure
        if figsize is None:
            if problem.n_latent_prms_dim > 2:
                n_rows, n_cols = axs.shape
                fig.set_size_inches(n_cols * inches_per_col, n_rows * inches_per_row)
            else:
                fig.set_size_inches(6.0, 5.0)
        else:
            fig.set_size_inches(figsize[0], figsize[1])

        # add a title to the plot (if requested) and apply a tight layout
        if title:
            fig.suptitle(title, fontsize=title_size)

        # the following command reduces the otherwise wide margins; when only two
        # parameter (components) are given, the tight_layout()-call only results in a
        # warning without having an effect - hence, the if-clause
        if problem.n_latent_prms_dim > 2:
            fig.tight_layout()

        # by default, the y-axis of the first and last marginal plot have ticks, tick-
        # labels and axis-labels that are not meaningful to show on the y-axis; hence,
        # we remove them here; since the default plot looks different for only two
        # latent parameters, there is a check before
        if problem.n_latent_prms_dim > 2:
            for i in [0, -1]:
                axs[i, i].yaxis.set_ticks_position("none")
                axs[i, i].yaxis.set_ticklabels([])
                axs[i, i].yaxis.set_visible(False)
            for i in range(problem.n_latent_prms_dim - 1):
                xlim = axs[-1, i].get_xlim()
                axs[i, i].set_xticks(ticks=axs[-1, i].get_xticks())
                axs[i, i].set_xlim(xlim)
            ylim = axs[-1, 0].get_ylim()
            axs[-1, -1].set_xticks(ticks=axs[-1, 0].get_yticks())
            axs[-1, -1].set_xlim(ylim)

        # when histograms are used to plot the marginals, the tick labels are often
        # rather close together, so that they overlap; here, they are rotated to
        # alleviate this overlap
        if histograms_on_diagonal:
            for i in range(problem.n_latent_prms_dim):
                axs[-1, i].tick_params(axis="x", labelrotation=45)

        # show the plot if requested
        if show:
            plt.show()  # pragma: no cover

        # Note: the returned axs-object can be saved to a file via:
        #     fig = axs.ravel()[0].figure
        #     fig.savefig(filename, ...)

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
            f"Invalid 'plot_with' argument: '{plot_with}'. Available options are "
            f"currently 'arviz', 'seaborn', 'matplotlib'"
        )


def create_posterior_plot(
    inference_data: az.data.inference_data.InferenceData,
    problem: "InverseProblem",
    plot_with: str = "arviz",
    kind: str = "hist",
    figsize: Optional[tuple] = None,
    inches_per_row: Union[int, float] = 3.0,
    inches_per_col: Union[int, float] = 2.5,
    textsize: Union[int, float] = 10,
    title_size: Union[int, float] = 14,
    title: Optional[str] = None,
    hdi_prob: float = 0.95,
    true_values: Optional[dict] = None,
    show: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Creates a posterior-plot for the given inference data.

    Parameters
    ----------
    inference_data
        Contains the results of the sampling procedure.
    problem
        The inverse problem the inference data refers to.
    plot_with
        Defines the python package the plot will be generated with. Options are:
        {'arviz', 'seaborn', 'matplotlib'}.
    kind
        Type of plot to display ('kde' or 'hist').
    figsize
        Defines the size of the generated plot in inches. If None is chosen, the figsize
        will be derived automatically by using inches_per_row and inches_per_col.
    inches_per_row
        If figsize is None, this will specify the inches per row in the subplot-grid.
        This argument has no effect if figsize is specified.
    inches_per_col
        If figsize is None, this will specify the inches per column in the subplot-grid.
        This argument has no effect if figsize is specified.
    textsize
        Defines the font size in the default unit.
    title_size
        Defines the font size of the figures title if 'title' is given.
    title
        The title of the figure.
    hdi_prob
        Defines the highest density interval. Must be a number between 0 and 1.
    true_values
        Used for plotting 'true' parameter values. Keys are the parameter names and
        values are the values that are supposed to be shown in the marginal plots.
    show
        When True, the show-method is called after creating the plot. Otherwise, the
        show-method is not called. The latter is useful, when the plot should be further
        processed.
    kwargs
        Additional keyword arguments passed to arviz' plot_posterior function.

    Returns
    -------
    axs
        The array of subplots of the created plot.
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
        axs = az.plot_posterior(
            inference_data,
            kind=kind,
            textsize=textsize,
            hdi_prob=hdi_prob,
            show=False,
            **kwargs,
        )

        # set the figure size; this is done either automatically if the user did not
        # specify the figsize argument, or it simply sets the requested figsize
        if isinstance(axs, np.ndarray):
            fig = axs.ravel()[0].figure
            if len(axs.shape) == 1:
                n_rows, n_cols = 1, axs.size
            else:
                n_rows, n_cols = axs.shape  # pragma: no cover
        else:
            fig = axs.figure
            n_rows, n_cols = 1, 1
        if figsize is None:
            fig.set_size_inches(n_cols * inches_per_col, n_rows * inches_per_row)
        else:
            fig.set_size_inches(figsize[0], figsize[1])

        # add a title to the plot (if requested) and apply a tight layout
        if title:
            fig.suptitle(title, fontsize=title_size)
        fig.tight_layout()

        # show the plot if requested
        if show:
            plt.show()  # pragma: no cover

        # Note: the returned axs-object can be saved to a file via:
        #     fig = axs.ravel()[0].figure
        #     fig.savefig(filename, ...)

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
            f"Invalid 'plot_with' argument: '{plot_with}'. Available options are "
            f"currently 'arviz', 'seaborn', 'matplotlib'"
        )


def create_trace_plot(
    inference_data: az.data.inference_data.InferenceData,
    problem: "InverseProblem",  # for consistent interface
    plot_with: str = "arviz",
    kind: str = "trace",
    figsize: Optional[tuple] = None,
    inches_per_row: Union[int, float] = 2.0,
    inches_per_col: Union[int, float] = 3.0,
    textsize: Union[int, float] = 10,
    title_size: Union[int, float] = 14,
    title: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Creates a trace-plot for the given inference data.

    Parameters
    ----------
    inference_data
        Contains the results of the sampling procedure.
    problem
        The inverse problem the inference data refers to.
    plot_with
        Defines the python package the plot will be generated with. Options are:
        {'arviz', 'seaborn', 'matplotlib'}.
    kind
        Allows to choose between plotting sampled values per iteration ("trace") and
        rank plots ("rank_bar", "rank_vlines").
    figsize
        Defines the size of the generated plot in inches. If None is chosen, the figsize
        will be derived automatically by using inches_per_row and inches_per_col.
    inches_per_row
        If figsize is None, this will specify the inches per row in the subplot-grid.
        This argument has no effect if figsize is specified.
    inches_per_col
        If figsize is None, this will specify the inches per column in the subplot-grid.
        This argument has no effect if figsize is specified.
    textsize
        Defines the font size in the default unit.
    title_size
        Defines the font size of the figures title if 'title' is given.
    title
        The title of the figure.
    show
        When True, the show-method is called after creating the plot. Otherwise, the
        show-method is not called. The latter is useful, when the plot should be further
        processed.
    kwargs
        Additional keyword arguments passed to arviz' plot_trace function.

    Returns
    -------
    axs
        The array of subplots of the created plot.
    """

    if plot_with == "arviz":

        # set default value for plot_kwargs if not given in kwargs; note that this
        # default value is mutable, so it should not be given as a default argument in
        # create_trace_plot
        if "plot_kwargs" not in kwargs:
            kwargs["plot_kwargs"] = {"textsize": textsize}

        # call the main plotting routine from arviz and return the axes object
        axs = az.plot_trace(inference_data, kind=kind, show=False, **kwargs)

        # set the figure size; this is done either automatically if the user did not
        # specify the figsize argument, or it simply sets the requested figsize
        fig = axs.ravel()[0].figure
        if figsize is None:
            n_rows, n_cols = axs.shape
            fig.set_size_inches(n_cols * inches_per_col, n_rows * inches_per_row)
        else:
            fig.set_size_inches(figsize[0], figsize[1])

        # add a title to the plot (if requested) and apply a tight layout
        if title:
            fig.suptitle(title, fontsize=title_size)
        fig.tight_layout(h_pad=1.75)

        # show the plot if requested
        if show:
            plt.show()  # pragma: no cover

        # Note: the returned axs-object can be saved to a file via:
        #     fig = axs.ravel()[0].figure
        #     fig.savefig(filename, ...)

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
            f"Invalid 'plot_with' argument: '{plot_with}'. Available options are "
            f"currently 'arviz', 'seaborn', 'matplotlib'"
        )
