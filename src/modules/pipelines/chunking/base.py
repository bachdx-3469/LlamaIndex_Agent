from typing import Any, List

from llama_index.core import Document

from ..base import BasePipeline


class BaseChunkPipeline(BasePipeline):

    def run(self, file_path: str, **kwargs: Any) -> List[Document]:
        raise NotImplementedError(f"Please implement `run` method for {self.__class__}")