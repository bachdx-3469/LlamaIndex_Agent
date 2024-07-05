from typing import List
from pathlib import Path

from llama_index.core import Document

from src.modules.components.parser.llmsherpa import LLMSherpaParser, DEFAULT_LLMSHERPA_API_URL
from .base import BaseChunkPipeline


class LLMSherpaChunkPipeline(BaseChunkPipeline):
    def __init__(self,
                 llmsherpa_api_url: str = DEFAULT_LLMSHERPA_API_URL,
                 include_section_info: bool = True,
                 save_chunks: bool = False,
                 save_location: str = None,
                 **kwargs):
        self.parser = LLMSherpaParser(
            save_chunks,
            save_location,
            llmsherpa_api_url
        )
        self.include_section_info = include_section_info

    def run(self,
            file_path: str,
            category: str = "Sun*") -> List[Document]:
        chunks = self.parser.parse_file(
            file_path, include_section_info=self.include_section_info, category=category
        )
        documents = []
        for i, chunk in enumerate(chunks):
            file_name = Path(file_path).name
            id = f"{file_path}_{i}"
            documents.append(
                Document(
                    text=chunk,
                    id_=id,
                    metadata={
                        "name": file_name,
                        "category": category
                    },
                    excluded_llm_metadata_keys=["file_path"]
                )
            )

        return documents
