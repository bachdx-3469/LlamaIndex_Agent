import sys

from .deepeval_eval import DeepEvalPipeline
from .base import BaseEvalPipeline


__all__ = [
    "DeepEvalPipeline",
    "get_eval_pipeline_from_name"
]


def get_eval_pipeline_from_name(name: str,
                                 **kwargs) -> BaseEvalPipeline:
    eval_pipeline = getattr(sys.modules[__name__], name)(**kwargs)
    return eval_pipeline