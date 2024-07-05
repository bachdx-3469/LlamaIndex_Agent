import sys
import os
import subprocess
from typing import List, Optional, Tuple

from llama_index.core import Settings
from llama_index.core.callbacks.schema import CBEventType, EventPayload
import llama_index.core.instrumentation as instrument

from deepeval.metrics import (
    BaseMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase

from .base import BaseEvalPipeline


dispatcher = instrument.get_dispatcher(__name__)


class DeepEvalPipeline(BaseEvalPipeline):
    def __init__(self,
                 metrics: Optional[List[BaseMetric]] = None,
                 threshold: float = 0.7,
                 include_reason: bool = True,
                 **kwargs):
        super().__init__(metrics, **kwargs)
        self.threshold = threshold
        self.include_reason = include_reason
        self.callback_manager = Settings.callback_manager
        self._init_azure()

    def _init_azure(self):
        end_point = os.getenv("AZURE_OPENAI__ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI__KEY")
        deployment_name = os.getenv("AZURE_OPENAI__GPT_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI__VERSION")

        subprocess.run(
            ["deepeval", "set-azure-openai",
             "--openai-endpoint", end_point,
             "--openai-api-key", api_key,
             "--deployment-name", deployment_name,
             "--openai-api-version", api_version]
        )

    def add_metric(self, metric: str):
        assert metric in [
            "FaithfulnessMetric", "ContextualPrecisionMetric",
            "ContextualRecallMetric", "ContextualRelevancyMetric"
        ]
        metric = getattr(sys.modules[__name__], metric)(
            threshold=self.threshold,
            include_reason=self.include_reason)
        self.metrics.append(metric)

    def metric_names(self) -> List[str]:
        return [metric.__name__ for metric in self.metrics]

    @dispatcher.span
    def _run(self,
            query: str,
            actual_output: str,
            retrieval_context: List[str],
            expected_output: Optional[str] = None,
            output_dict: Optional[dict] = None,
            **kwargs) -> List[Tuple]:
        """
        Perform all metrics evaluation on a single sample
        :return: result for that single sample
        """
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context
        )
        results = []
        for metric in self.metrics:
            metric.measure(test_case)
            results.append({
                metric.__name__: (metric.score, metric.reason)
            })

        additional_vals = self._generate_additional_values(output_dict)
        self.outputs.append(
            (query, actual_output, expected_output, retrieval_context,
             additional_vals, results)
        )

        return self.outputs
