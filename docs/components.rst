Components
**********

In order to provide a valid definition of an inverse problem (i.e., a parameter estimation problem) using **probeye**, four main ingredients (or components, as they are called here) are required.

1. Parameters
2. Experiments
3. Forward models
4. Likelihood models

These four components have to be defined by the user in a way of `adding` them to a problem instance. Consequently, the base structure of the corresponding **probeye**-code looks like this:

.. code-block:: python

    from probeye.definition import InverseProblem

    # initialize a problem instance
    problem = InverseProblem("MyProblem")

    # add the four components
    problem.add_parameter(...)
    problem.add_experiment(...)
    problem.add_forward_model(...)
    problem.add_likelihood_model(...)

Of course the dots in parenthesis :code:`(...)` still need to be further specified (according to the problem at hand), but the fundamental structure of how to define an inverse problem is given by the code block above. It should be pointed out that of each component multiple instances can be added - for example five parameters, two forward models, one hundred experiments and ten likelihood models - but at least one instance of each is required to obtain a valid problem definition. Also the order of adding those components should look like above. So, at first the parameters are added, then the experiments followed by the forward models and the likelihood models are added at last. Each of these components is explained in more detail in the following sections.

Parameters
##########
In **probeye**, an inverse problem is understood as a parameter estimation problem. Hence, it comes at no surprise that one needs to define at least one parameter that should be inferred. After initializing an inverse problem, adding the parameters to the problem is the natural next step. In principle, you could also add the experiments first, but it is recommended to begin with the parameters, because problem definitions are more readable like that.

Latent and constant parameters
------------------------------
Generally, two kinds of parameters are distinguished in **probeye**: latent and constant parameters. This property is also referred to as the parameter's `role`. Latent parameters are parameters that should be inferred, while constant parameters have a pre-defined value, and will hence not be inferred in the inference step. Earlier, it was pointed out that the definition of at least one parameter is required for a valid problem definition. Now we should state more precisely: the definition of at least one latent parameter is required for a valid problem definition.

A typical definition of a latent parameter looks like this:

.. code-block:: python
    :emphasize-lines: 3

    problem.add_parameter(
        "a",
        prior=Normal("mean"=1.0, "std"=2.0),
        tex="$a$",
        info="Slope of the fitted function",
    )

And a typical definition of a constant parameter looks like this:

.. code-block:: python
    :emphasize-lines: 3

    problem.add_parameter(
        "sigma_meas",
        value=0.1,
        info="Standard deviation of measurement error",
    )

As one can see, the definition of either a latent or a constant parameter is triggered by using the :code:`prior` or the :code:`value` keyword argument in the :code:`add_parameter`-method. The :code:`value` keyword argument can be a scalar like in the example above or a vector, for example :code:`value=np.array([0.9, -0.3])`. The :code:`prior` keyword argument on the other hand has to be given as a prior object that is initialized with its specific parameter values. Possible prior options are currently :code:`Normal`, :code:`MultivariateNormal`, :code:`LogNormal`, :code:`TruncNormal`, :code:`Uniform`, :code:`Weibull` and :code:`SampleBased`. More information on the priors and their parameters is given in this :ref:`section<Prior definition of latent parameters>` below.

Finally, it should be pointed out that it is possible to give a very short definition of a latent parameter by neither specifying the :code:`prior` nor the :code:`const` keyword argument. Examples could look like this:

.. code-block:: python

    problem.add_parameter("a")
    problem.add_parameter("b", domain="(0, 1]")

In both of these cases an `uninformative` prior is assumed, meaning a prior that is constant over its domain. Note however, that internally, the `uninformative` prior is not a proper prior like the conventional prior classes, but just a flag stating that the corresponding parameter is a latent parameter without a prior. These types of latent parameters can only be used for maximum likelihood estimations. When using a sampling-based solver, it is required to specify a proper prior.

A parameter's name and type
---------------------------
Each parameter (latent and constant) must have a name and a type. The parameter's name, which is given by the first argument in the :code:`add_parameter`-method,  must be unique in the scope of the problem, i.e., no other parameter can have the same name. This name is also referred to as the parameter's `global name`.

The parameter's type on the other hand, states where the parameter appears in the problem definition. There are three possible types :code:`model`, :code:`prior` and :code:`likelihood`. A parameter of type :code:`model` appears in one the problem's forward models, while a parameter of type :code:`prior` will be used in the definition of some latent parameter's prior. Finally, a parameter of type :code:`likelihood` will appear in one of the problem's likelihood models. The specification of the prior type is optional. At the beginning of **probeye**'s development they used to be stated explicitly in the :code:`add_parameter`-method as the second positional argument. But today this is not necessary anymore because: if the type is not given, it will be determined automatically.

Prior definition of latent parameters
-------------------------------------
As described above, when defining a latent parameter, one has to provide a prior object that is initialized with the prior's parameters and their values. The following table provides the currently implemented options.

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - Prior type
      - Prior parameters
      - Comments
    * - :code:`Normal`
      - :code:`mean`, :code:`std`
      - Gaussian or normal distribution where :code:`mean` refers to the mean and :code:`std` to the standard deviation.
    * - :code:`MultivariateNormal`
      - :code:`mean`, :code:`cov`
      - Multivariate normal distribution where :code:`mean` refers to the mean and :code:`cov` to the covariance matrix.
    * - :code:`LogNormal`
      - :code:`mean`, :code:`std`
      - Log-normal distribution where :code:`mean` refers to the mean and :code:`std` is the standard deviation on the log-scale.
    * - :code:`TruncNormal`
      - :code:`mean`, :code:`std`, :code:`low`, :code:`high`
      - Truncated normal distribution. Same as for "normal", while :code:`low` and :code:`high` refer to the lower and upper bound respectively.
    * - :code:`Uniform`
      - :code:`low`, :code:`high`
      - Uniform distribution where :code:`low` is the lower and :code:`high` is the upper bound. Note that these bounds are inclusive.
    * - :code:`Weibull`
      - :code:`scale`, :code:`shape`
      - Weibull distribution. Check out the `scipy-documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html>`_ for more information on the parameters.
    * - :code:`SampleBased`
      - :code:`samples`
      - Gaussian `kernel density estimate <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html>`_ based on a given vector of samples.

It should be pointed out that it is also possible to use a latent parameter as a prior parameter. The following example may illustrate that.

.. code-block:: python

    problem.add_parameter(
        "mean_a",
        prior=Uniform("low"=-1.0, "high"=1.0),
        tex="r$\mu_a$",
        info="Mean parameter of a's prior",
    )
    problem.add_parameter(
        "a",
        prior=Normal("mean"="mean_a", "std"=2.0),
        tex="$a$",
        info="Slope of the fitted function",
    )

Note that instead of providing a numeric value for :code:`a`'s :code:`mean` parameter, the name (hence a string) of the previously defined latent parameter :code:`mean_a` is provided. It is important in this example that :code:`mean_a` is defined before :code:`a`.

A latent parameter's domain
---------------------------
Sometimes, the value of a latent parameter should stay in certain bounds. For example, if a parameter appears in the denominator of a fraction, it cannot assume the value zero. One measure to address such situations is to define the parameter's prior in a way that its domain does not contain problematic values. However, during sampling-procedures it is still possible that values outside of a prior's domain are proposed, and hence evaluated. To prevent that, one can define a latent parameter's domain via the :code:`domain` argument when adding it to the problem. This would look like this:

.. code-block:: python

    problem.add_parameter(
        "gamma",
        domain="(0, 1)",
        prior=("uniform", {"low": 0.0, "high": 1.0}),
    )

Here, the domain of :code:`gamma` is specified to an open interval from zero to one. Other valid strings for the domain argument are for example :code:`"[0, 1]"` for a closed interval, :code:`"(0, 1]"` or :code:`"[0, 1)"` for half-closed intervals, or :code:`"(-oo, oo)"` for a domain from minus to plus infinity. Other variations are of course possible. For a multivariate parameter, the definition looks very similar as shown by the following example.

.. code-block:: python

    problem.add_parameter(
        "mb",
        dim=2,
        domain="(-oo, +oo) (-oo, +oo)",
        prior=MultivariateNormal(
            "mean"=np.array([0.0, 0.0]),
            "cov"=np.array([[1.0, 0.0], [0.0, 1.0]]),
        ),
    )

So in this case, the :code:`domain`-string is simply a concatenation of :code:`domain`-strings for a 1D-interval. Note that for multidimensional parameter, also a :code:`dim`-argument is required, that specifies the parameter's dimensionality. If a latent parameter is added to a problem without specifying its domain, it is assumed that there are no restrictions. So, in the code block above, the domain-specification would actually be unnecessary since this domain would also have been assumed if no domain was specified.

The tex and info arguments
--------------------------
Each parameter can (but does not have to) have a tex and an info attribute. While the tex attribute is used for plotting, the info string is used when calling a problems info-method :code:`problem.info()` printing some information on the defined problem. Even if not required, it is recommended to define both of these attributes for each parameter added to the problem.

Experiments
###########
The experiments that are added to an inverse problem are the containers of the experimentally recorded data that is used to calibrate the problem's latent parameters with. But they also will contain input data for the forward model like initial or boundary conditions that are needed to simulate a specific experiment. To add an experiment in **probeye**, the code looks like this:

.. code-block:: python

        problem.add_experiment(
            name="TestSeries_Aug12_2018",
            sensor_data={
                'y1': np.array([1.12321, 0.37320, 0.14189, -0.22992, -0.04648]),
                'y2': np.array([0.20105, 1.61940, 0.33614,  0.53154,  0.04718]),
                'y3': np.array([2.68936, 0.29683, 1.10388,  0.81638,  1.48964]),
                'time': np.array([0.0, 0.25, 0.5 , 0.75, 1.0])
                'offset': 1.29,
            },
        )

The first argument (here: :code:`"TestSeries_Aug12_2018"`) is a unique name of the experiment. The second argument states the actual measurement data, i.e., the values that have been recorded by the experiment's sensors. Those values can be given as scalars (float, int) or as vectors in form of tuples or numpy arrays. It is not required that all entries have the same length. Note however, that these arrays have to be one-dimensional and cannot be of higher dimension. The keys of the :code:`sensor_data`-dictionary will be referenced later in the forward model's definition, as it will be discussed in the next section.

Forward models
##############
The forward model is a parameterized simulation model (for example a finite element model) the predictions of which should be compared against some experimental data. The forward model's parameters are typically the parameters which are of primary interest within the stated problem. It should be pointed out that many inverse problems might contain only one forward model, but it is also possible to set up a problem that contains multiple forward models.

In **probeye**, a forward model is a function that has two kinds of arguments: `input sensors` and `parameters`, see also the figure below. While `input sensors` refer to specific experimental data that is required to simulate it (for example certain initial or boundary conditions, load settings, etc.), `parameters` refer to the forward model's parameters. Once all input sensors and parameters are provided, the forward model computes a result that it returns via its output sensors.

.. figure:: images/forward_model.png
   :align: center
   :width: 90%

In order to add a forward model to an inverse problem, two steps are required. At first, the forward model has to be defined as a Python class. This definition is done by setting up a new model class (that can have an arbitrary name) which is based on the **probeye**-class :code:`ForwardModelBase`. This class must have both an :code:`interface`-method, which defines the forward model's parameters, input sensors and output sensors, and it must have a :code:`response`-method, which describes a forward model evaluation. The :code:`response`-method has only one input, which is a dictionary that contains both the input sensors and the parameters. The method will then perform some computations and returns its results in form of a dictionary of the forward model's output sensors. For a simple linear model, such a definition could look like this:

.. code-block:: python

    from probeye.definition.forward_model import ForwardModelBase
    from probeye.definition.sensor import Sensor

    class LinearModel(ForwardModelBase):
        def interface(self):
            self.parameters = ["a", "b", "offset"]
            self.input_sensors = Sensor("time")
            self.output_sensors = [
                Sensor("y1", x=0.0, std_model="sigma_1"),
                Sensor("y2", x=0.5, std_model="sigma_2"),
                Sensor("y3", x=1.0, std_model="sigma_3"),
            ]

        def response(self, inp: dict) -> dict:
            t = inp["time"]
            a = inp["a"]
            b = inp["b"]
            offset = inp["offset"]
            response = dict()
            for os in self.output_sensors:
                response[os.name] = a * os.x + b * t + offset
            return response

After the forward model has been defined, it must be added to the problem. For the example shown above, this would look like this:

.. code-block:: python

    # add the forward model to the problem
    problem.add_forward_model(
        LinearModel("LinearModel"), experiments=["TestSeries_Aug12_2018"]
    )

Next to the forward model instance (initialized with an arbitrary name), which is provided as the first argument of the :code:`add_forward_model`-method, it is necessary to provide a list of experiments (which have been added previously) that are described by the forward model. Note that the stated experiments must have all of the forward model's input and output sensor names as keys in the :code:`sensor_data` dictionary. This means, the experiments need to contain all the data required or simulated by the forward model.

The interface method
--------------------
The :code:`interface`-method defines three attributes of the user defined forward model. The forward model's parameters, its input sensors and output sensors. The parameters (:code:`self.parameters`) define which parameters defined within the problem scope are used by the forward model. These parameters can be latent or constant ones. Parameters are given as a list of strings or as a single string, if the model only uses a single parameter. It is also possible to use another name for a globally defined parameter within the forward model, a local parameter name. This can be achieved by providing a one-element dictionary containing the global and local name instead of a single string of the global name. The following code lines show examples of how the (:code:`self.parameters`) attribute can be set.

.. code-block:: python

    self.parameters = "m"
    self.parameters = ["m"]
    self.parameters = ["m", "b"]
    self.parameters = [{"m": "a"}, "b"]

In the last example, the globally defined parameter :code:`m` will be known as :code:`a` within the scope of the forward model. Even though this option exists, it is recommended to not use local names if not necessary, since it might be more confusing than helpful.

The definition of the input and output sensors (:code:`self.input_sensors`, :code:`self.output_sensors`) is done by providing a list of :code:`Sensor`-objects (or a single :code:`Sensor`-object if only one sensor is assigned). In their most basic definition, sensors are just objects with a :code:`name`-attribute. For example :code:`Sensor("x")` creates a :code:`Sensor`-object with the name attribute :code:`"x"`. These sensor-names will refer to specific data stored in the experiments added to the problem in the next step. For the forward model's input sensors, this very basic sensor type is already sufficient. By providing the input sensors (with their :code:`name`-attribute) one is essentially just naming all of the forward model's input channels that are not parameters.

When defining the forward model's output sensors, more information must be provided. Each output sensor still requires a name attribute, which refers to specific experimental data the forward model's output will be compared against. But it must also contain the definition of the global parameter, that describes the model error scatter in the considered output sensor. In the definition of the forward model above, the first output sensor of the forward model is defined as

.. code-block:: python

    Sensor("y1", x=0.0, std_model="sigma_1")

Here, the standard deviation of the model prediction error (:code:`std_model`) is described by the global parameter :code:`sigma_1`. Additionally, the output sensors are assigned a positional-attribute, here :code:`x`, which is referred to in the :code:`response`-method. Note that the only required argument for an output sensor is the name (here, :code:`y1`) and the model error standard deviation parameter :code:`std_model`.

The response method
-------------------
The forward model's response method is its computing method. It describes how given parameters and inputs are processed to provide the forward model's prediction in terms of the output sensors. The only input of this method is a dictionary :code:`inp` which contains as keys all (local) parameter names, as well as all input sensor names as defined in the :code:`self.parameters`- :code:`self.input_sensors`-attribute in the :code:`interface`-method. In the example given above, those keys are be :code:`"time", "a", "b", "offset"`. The values to those keys are either parameter constants, experimental data or - in the case of latent parameters - values chosen by the used solver during the inference step.

The computed result of the forward model must be put into a dictionary when returned. The keys of this dictionary must be the names of the forward model's output sensors as defined in the :code:`interface`-method.

Likelihood models
#################
The last component to be added to an inverse problem is the likelihood model (or several likelihood models). The likelihood model's purpose is to compute the likelihood (more precisely the log-likelihood) of a given choice of parameter values by comparing the forward model's predictions (using the given parameter values) with the experimental data. In this section, only likelihood models are considered that do not account for correlations. In such a framework, the addition of a likelihood model to the inverse problem (referring to the examples shown above) could look like this:

.. code-block:: python

        problem.add_likelihood_model(
            GaussianLikelihoodModel(
                experiment_name="TestSeries_Aug12_2018",
                model_error="additive",
            )
        )

The interface is fairly simple. The :code:`experiment_name` argument states the name of the experiment this likelihood model refers to. Note that each likelihood model refers to exactly one experiment. Hence, when several experiments are defined, the same number of likelihood models is required. The :code:`model_error` can be defined as :code:`"additive"` or :code:`"multiplicative"`, depending on the requested error model.
