from abc import ABC, abstractmethod

class BaseMap(ABC):
    r"""Base class for all odometry providers.

    Your providers should also subclass this class. You should override the `update_map()` method.
    """

    def __init__(self, *params):
        r"""Initializes the inital map"""
        pass

    @abstractmethod
    def update_map(self, *args, **kwargs):
        r"""Updare global map """
        raise NotImplementedError

