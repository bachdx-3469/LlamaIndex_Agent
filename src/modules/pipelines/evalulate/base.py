import os
import pandas as pd
import warnings
from abc import abstractmethod
from typing import List, Tuple, Any, Optional, Dict

import llama_index.core.instrumentation as instrument
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload

from ..base import BasePipeline

dispatcher = instrument.get_dispatcher(__name__)


class BaseEvalPipeline(BasePipeline):
    def __init__(self,
                 metrics: Optional[List[Any]] = None,
                 callback_manager: Optional[CallbackManager] = None,
                 **kwargs):
        self.outputs = []
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = []
        self.callback_manager = callback_manager or Settings.callback_manager

    @abstractmethod
    def add_metric(self, metric: str):
        ...

    @abstractmethod
    def metric_names(self) -> List[str]:
        ...

    def _generate_additional_header(self, output_dict: Dict[str, str]) -> List[str]:
        headers = list(output_dict.keys())
        return headers

    def _generate_additional_values(self, output_dict: Dict[str, str]) -> List[str]:
        val = list(output_dict.values())
        return val

    @dispatcher.span
    def run(self,
            query: str,
            actual_output: str,
            retrieval_context: List[str],
            expected_output: Optional[str] = None,
            output_dict: dict = None,
            **kwargs) -> List[Tuple]:
        with self.callback_manager.as_trace("evaluate"):
            with self.callback_manager.event(
                CBEventType.QUERY,
                payload={
                    EventPayload.QUERY_STR: [query, actual_output, retrieval_context, expected_output]
                }
            ) as eval_event:
                results =  self._run(
                    query,
                    actual_output,
                    retrieval_context,
                    expected_output,
                    output_dict,
                    **kwargs
                )

                eval_event.on_end(payload={EventPayload.RESPONSE: results})
        return results

    @abstractmethod
    def _run(self,
             query: str,
             actual_output: str,
             retrieval_context: List[str],
             expected_output: Optional[str] = None,
             output_dict: dict = None,
             **kwargs) -> List[Tuple]:
        ...

    def handle_error(self,
                     query: str,
                     actual_output: str,
                     retrieval_context: List[str],
                     expected_output: Optional[str] = None,
                     output_dict: dict = None):
        warnings.warn(f"Error evaluate query `{query}`")
        err_results = []
        for metric in self.metrics:
            err_results.append(
                {metric.__name__: (0.0, "Error while evaluating")}
            )
        additional_vals = self._generate_additional_values(output_dict)
        self.outputs.append(
            (query, actual_output, expected_output, retrieval_context,
             additional_vals, err_results)
        )

    def display_result(self,
                       output_path: str,
                       output_dict: Dict[str, str]):
        metrics_names = self.metric_names()
        metric_headers = []
        for metric_name in metrics_names:
            metric_headers.append(f"{metric_name} Result")
        additional_headers = self._generate_additional_header(output_dict)
        headers = [
            "Query", "Actual Output", "Expected Output", "Retrievals Chunk"
        ] + additional_headers + metric_headers
        df = pd.DataFrame(columns=headers)
        for i, output in enumerate(self.outputs):
            (query, actual_output, expected_output,
             retrieval_context) = output[0:4]
            additional_vals = output[4]
            metric_list = output[5]
            # convert metric_list from list of dict to list
            metric_results = []
            for metric in metric_list:
                v = list(metric.values())[0]
                metric_results.append(
                    f"Score: {v[0]}\nReason: {v[1]}")
            df.loc[i] = ([query, actual_output, expected_output, retrieval_context] +
                          additional_vals + metric_results)

        df.to_excel(os.path.join(output_path, "output_eval.xlsx"), index=False)
