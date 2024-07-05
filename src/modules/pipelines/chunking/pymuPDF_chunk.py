from typing import Any, List
from pathlib import Path

from llama_index.core import Document

from src.modules.components.parser import PymuPDFParser
from .base import BaseChunkPipeline


class PymuPDFChunkPipeline(BaseChunkPipeline):
    def __init__(self,
                 header_height: int = 125,
                 save_chunks: bool = True,
                 save_location: str = None,
                 **kwargs):
        self.parser = PymuPDFParser(
            save_chunks,
            save_location,
            header_height
        )

    def run(self, file_path: str, **kwargs: Any) -> List[Document]:
        chunks = self.parser.parse_file(file_path)
        documents = []
        for i, chunk in enumerate(chunks):
            file_name = Path(file_path).name
            id = f"{file_path}_{i}"
            documents.append(
                Document(
                    text=chunk,
                    id_=id,
                    metadata={
                        "name": file_name
                    },
                    excluded_llm_metadata_keys=["file_path"]
                )
            )

        return documents
