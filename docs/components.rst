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

Parameter's name and type
-------------------------
Each parameter (latent and constant) must have a name and a type. The parameter's name must be unique in the scope of the problem, i.e., no other parameter can have the same name. The parameter's type, on the other hand, states where the parameter appears in the problem definition. There are three possible types :code:`model`, :code:`prior` and :code:`likelihood`. A parameter of type :code:`model` appears in one the problem's forward models, while a parameter of type :code:`prior` will be used in the definition of some latent parameter's prior. Finally, a parameter of type :code:`likelihood` will appear in one of the problem's likelihood models.

Tex and info
------------
Each parameter can (but does not have to) have a tex and an info attribute. While the tex attribute is used for plotting, the info string is used when calling a problems info-method :code:`problem.info()` printing some information on the defined problem. Even if not required, it is recommended to define both of these attributes for each parameter added to the problem.

Forward models
##############
XXX

Experiments
###########
XXX

Likelihood models
#################
XXX