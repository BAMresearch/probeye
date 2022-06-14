class CorrelationModel:
    """
    Base class for the different (specific) correlation models.
    """

    def __init__(self, **kwargs):

        self.correlation_variables = self._interpret_corr_variables([*kwargs.keys()])
        self.parameters = [*kwargs.values()]
        self.corr_dict = {
            key: val for key, val in zip(self.correlation_variables, self.parameters)
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


class ExpModel(CorrelationModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
