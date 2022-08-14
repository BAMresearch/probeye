Mathematics
***********

In the following, a brief overview with respect to the inverse problems that can be described when using **probeye** should be outlined. This requires to define the statistical models (the data generating processes) that can be build from a given deterministic simulation model, as it is assumed to be the case when using this package.

Let us assume that :math:`\mathbf{y}_e\in\mathbb{R}^d`, :math:`d\in\mathbb{N}` denotes the observed data (observations) of some process that depends on a set of controllable input data which we summarize as :math:`\mathbf{x}_e\in\mathbb{R}^{d_x}`, :math:`d_x\in\mathbb{N}`. In **probeye** it is assumed, that the observations :math:`\mathbf{y}_e` are realizations of a data generating process that can be described by a multivariate normal distribution that depends on the input data :math:`\mathbf{x}_e`. If :math:`\mathbf{Y}_e` denotes the corresponding random variable, this can be expressed as

.. math::
   :name: eq:1

   \mathbf{Y}_e \sim \mathcal{N}(\bm{\mu}(\mathbf{x}_e), \bm{\Sigma}(\mathbf{x}_e)).

Here, :math:`\bm\mu(\mathbf{x}_e)\in\mathbb{R}^d` and :math:`\bm\Sigma(\mathbf{x}_e)\in\mathbb{R}^{d \times d}` refer to the mean vector and the covariance matrix respectively. Next to the previously described observations, we assume that a deterministic forward model :math:`\mathbf{y}` is available that describes the mean vector :math:`\bm{\mu}` of the data generating process, that is

.. math::

    \bm{\mu}(\mathbf{x}_e) = \mathbf{y}(\mathbf{x}_e,\bm\theta_y)

where :math:`\bm\theta_y\in\mathbb{R}^n`, :math:`n\in\mathbb{N}` is the model parameter vector. Since :math:`\bm\mu` is given by the forward model response, the different data generating processes that can be described in **probeye** only differ in the definition of the covariance matrix :math:`\bm{\Sigma}`. The latter depends on the assumed sub-structure of the data generation process. In this context, two general options are supported. A data generation process with an additive or a multiplicative model prediction error. In the first case, Expression :ref:`(1) <eq:1>` is specified to

.. math::
    :name: eq:2

    \mathbf{Y}_e = \mathbf{y}(\mathbf{x}_e,\bm\theta_y) + \mathbf{E}_\mathrm{model}(\mathbf{x}_e,\bm{\theta}_\ell) + \mathbf{E}_\mathrm{meas}

Here, :math:`\mathbf{E}_\mathrm{model}(\mathbf{x}_e,\bm{\theta}_\ell) \sim \mathcal{N}(\mathbf{0}, \bm{\Sigma}_\mathrm{model}(\mathbf{x}_e,\bm{\theta}_\ell))` denotes a Gaussian model for the prediction error, introducing additional latent parameters :math:`\bm\theta_\ell\in\mathbb{R}^m`, :math:`m\in\mathbb{N}`, while :math:`\mathbf{E}_\mathrm{meas} \sim \mathcal{N}(\mathbf{0}, \mathrm{diag}(\sigma_m^2))`, :math:`\sigma_m\in\mathbb{R}_+` describes an independent and identical distributed (i.i.d.) measurement error. The alternative to the additive model prediction error is a multiplicative one. In this case, Expression :ref:`(1) <eq:1>` is specified to

.. math::
    :name: eq:3

    \mathbf{Y}_e =  \mathbf{K}_\mathrm{model}(\mathbf{x}_e,\bm{\theta}_\ell)\mathbf{y}(\mathbf{x}_e,\bm\theta_y) + \mathbf{E}_\mathrm{meas}

where  :math:`\mathbf{K}_\mathrm{model}(\mathbf{x}_e,\bm{\theta}_\ell) \sim \mathcal{N}(\mathbf{1}, \bm{\Sigma}_\mathrm{model}(\mathbf{x}_e,\bm{\theta}_\ell))` denotes the unit-mean prediction error while :math:`\mathbf{E}_\mathrm{meas} \sim \mathcal{N}(\mathbf{0}, \mathrm{diag}(\sigma_m^2))` describes the measurement error as defined before. Both data generation processes, Equations :ref:`(2) <eq:2>` and :ref:`(3) <eq:3>`, describe a random variable following a multivariate normal distribution as given by :ref:`(1) <eq:1>` where the covariance matrix is given by

.. math::

    \bm{\Sigma}(\mathbf{x}_e) = \begin{cases}
        \bm{\Sigma}_\mathrm{model}(\mathbf{x}_e,\bm{\theta}_\ell) + \mathrm{diag}(\sigma_m^2)  & \text{(additive)} \\
        \mathrm{diag}(\mathbf{y}(\mathbf{x}_e,\bm\theta_y))\bm{\Sigma}_\mathrm{model}(\mathbf{x}_e,\bm{\theta}_\ell)\mathrm{diag}(\mathbf{y}(\mathbf{x}_e,\bm\theta_y)) + \mathrm{diag}(\sigma_m^2) & \text{(multiplicative).}
        \end{cases}

Once the mean vector :math:`\bm\mu(\mathbf{x}_e)` and the covariance matrix :math:`\bm{\Sigma}(\mathbf{x}_e)` are determined, the likelihood of the statistical model can be evaluated via

.. math::

    \ell(\bm{x}_e,\bm\theta_y,\bm\theta_\ell) =	\frac{\exp\left(-\frac{1}{2}(\bm{y}_e - \bm\mu(\bm{x}_e))^T\bm\Sigma(\bm{x}_e)^{-1}(\bm{y}_e - \bm\mu(\bm{x}_e))\right)}{\sqrt{(2\pi)^d\det\bm\Sigma(\bm{x}_e)}}.