from abc import ABC, abstractmethod

__all__ = ["Odometry"]


class Odometry(ABC):
    r"""Base class for all odometry providers.

    Your providers should also subclass this class. You should override the `provide()` method.
    """

    def __init__(self, *params):
        r"""Initializes internal Odometry state"""
        pass

    @abstractmethod
    def provide(self, *args, **kwargs):
        r"""Defines the odometry computation performed at every `.provide()` call. """
        raise NotImplementedError

    @abstractmethod
    def localize(self,*args, **kwargs):
        r"""Defines the odometry computation performed at every `.provide()` call. """
        raise NotImplementedError