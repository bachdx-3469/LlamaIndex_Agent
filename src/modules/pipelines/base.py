from typing import Any
from abc import ABC, abstractmethod


class BasePipeline(ABC):

    @abstractmethod
    def run(self, **kwargs: Any):
        ...