class CorrelationModel:
    """
    Base class for the different (specific) correlation models.

    Parameters
    ----------
    kwargs
        The keys are the correlation variables and the values are strings or lists of
        strings that state correlation-parameters for the respective correlation
        variable. For example CorrelationModel(x="l_corr_x", t="l_corr_t"). Note that
        multi-dimensional correlation variables are written by using '__' to separate
        their components. For example CorrelationModel(x__y="l_corr").
    """

    def __init__(self, **kwargs):

        # the type will be specific to each model
        self.model_type = "base"

        # these are essentially the keys (as strings) of the given kwargs; keys that
        # contain '__' (e.g. x__y) are translated into tuples (x__y -> ('x', 'y'))
        self.correlation_variables = self._interpret_corr_variables([*kwargs.keys()])

        # assemble a list of all the correlation model's parameters
        self.parameters = self._read_parameters([*kwargs.values()])

        # this dictionary allows to get the parameters associated with a specific
        # correlation variable
        self.corr_dict = {
            key: val for key, val in zip(self.correlation_variables, [*kwargs.values()])
        }

    @staticmethod
    def _interpret_corr_variables(cv_list):
        """
        The correlation variables are given as the keys of the keyword arguments in the
        initialization of a CorrelationModel. Multidimensional correlation variables are
        given by separating a key with '__'. E.g., CorrelationModel(x__y="l_corr"). In
        this example, a 2D correlation variable consisting of 'x' and 'y' is assigned
        the parameter 'l_corr'. This interpretation (meaning: converting multi-dim.
        correlation variables in respective tuples) is done in this method.

        Parameters
        ----------
        cv_list
            A list of correlation variables. For example ['x', 't'] or ['x__y__z', 't'].

        Returns
        -------
        cv_list_interpreted
            The interpreted correlation variables. In the examples given above it would
            be returned ['x', 't'] and [('x', 'y', 'z'), 't'] respectively.
        """
        cv_list_interpreted = []
        for cv in cv_list:
            if "__" in cv:
                cv_list_interpreted.append(tuple(cv.split("__")))
            else:
                cv_list_interpreted.append(cv)
        return cv_list_interpreted

    @staticmethod
    def _read_parameters(value_list: list) -> list:
        """
        Check the values of the given kwargs during init and assembles them in a
        parameter list.

        Parameters
        ----------
        value_list
            The values of the given kwargs in init.

        Returns
        -------
        parameters
            A list of strings describing the correlation model's parameters.
        """
        parameters = []
        for value in value_list:
            if isinstance(value, str):
                parameters.append(value)
            elif isinstance(value, list):
                parameters += value
            else:
                raise ValueError(
                    f"Found invalid value in the given kwargs. They can be either "
                    f"strings or lists, but found type {type(value)} ({value})."
                )
        return parameters


class ExpModel(CorrelationModel):
    """
    Descriptive class for an exponential correlation model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "exponential"
