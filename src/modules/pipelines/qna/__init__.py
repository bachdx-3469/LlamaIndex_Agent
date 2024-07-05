import sys

from .decompose_hyde import DecomposeHyDEQnAPipeline
from .default import DefaultQnAPipeline
from .hyde import DefaultHyDEQnAPipeline
from .base import BaseQnAPipeline


def get_qna_pipeline_from_name(name: str,
                               **kwargs) -> BaseQnAPipeline:
    qna_pipeline = getattr(sys.modules[__name__], name)(**kwargs)
    return qna_pipeline