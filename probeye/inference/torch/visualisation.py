import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt

def visualisation(mcmc,problem, pair_plot = True, posterior_plot = True, plot_trace = True):
    """
    Currently very quick implementation. WIP
    :param mcmc:
    :param problem
    :return:
    """
    data = az.from_pyro(mcmc)
    if pair_plot:
        az.plot_pair(data, kind="kde")
    if posterior_plot:
        az.plot_posterior(data, var_names=problem.get_theta_names(), kind='kde')
    if plot_trace:
        az.plot_trace(data)

    plt.show()