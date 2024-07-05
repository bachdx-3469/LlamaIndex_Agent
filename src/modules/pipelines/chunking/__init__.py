import sys

from .agentic_chunk import AgenticChunkPipeline
from .agentic_chunk_v2 import AgenticChunkPipelineV2
from .agentic_chunk_v3 import AgenticChunkPipelineV3
from .llmsherpa_chunking import LLMSherpaChunkPipeline
from .base import BaseChunkPipeline


__all__ = [
    "AgenticChunkPipeline",
    "LLMSherpaChunkPipeline",
    "AgenticChunkPipelineV2",
    "AgenticChunkPipelineV3",
    "get_chunk_pipeline_from_name"
]


def get_chunk_pipeline_from_name(name: str,
                                 **kwargs) -> BaseChunkPipeline:
    chunk_pipeline = getattr(sys.modules[__name__], name)(**kwargs)
    return chunk_pipeline