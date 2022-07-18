# local imports
from probeye.definition.forward_model import ForwardModelBase


class SurrogateModelBase(ForwardModelBase):
    """
    Base class for a surrogate model, i.e., a forward model that approximates another
    (typically computationally more expensive) forward model.

    Parameters
    ----------
    name
        The name of the surrogate model. Must be unique among all surrogate model's
        names within a considered InverseProblem.
    forward_model
        The forward model object that the surrogate model approximates.
    """

    def __init__(self, name: str, forward_model: ForwardModelBase):

        super().__init__(name)

        # the surrogate model has access to the forward model it approximates; this
        # forward model can be called (evaluating its response) from the surrogate model
        # via self.forward_model.response(inp) where inp contains the input dictionary
        self.forward_model = forward_model

    def fit(self):
        """
        Prepares the surrogate model by approximating the forward model in some way.
        """
        pass
