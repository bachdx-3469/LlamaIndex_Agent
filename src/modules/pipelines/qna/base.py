from typing import Dict, Any, Tuple
from abc import abstractmethod

from llama_index.core import Settings
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import QueryType

from ..base import BasePipeline


class BaseQnAPipeline(BasePipeline):
    def __init__(self,
                 debug: bool = True):
        self.callback_manager = Settings.callback_manager
        self.debug = debug
        self.components = []
        self.additional_outputs = {}

    def _gather_additional_outputs(self) -> Dict[str, Any]:
        for component in self.components:
            if getattr(component, "log_output", None):
                self.additional_outputs = component.log_output(
                    self.additional_outputs
                )
        return self.additional_outputs

    @abstractmethod
    def _query(self, query: QueryType) -> RESPONSE_TYPE:
        ...

    def run(self, query: QueryType) -> Tuple[RESPONSE_TYPE, Dict]:
        response = self._query(query)
        self._gather_additional_outputs()
        return (response, self.additional_outputs)
