from typing import List, Optional
from pathlib import Path
import warnings

from llama_index.core import (
    SimpleDirectoryReader)
from llama_index.readers.file import (
    DocxReader,
    PDFReader)

from llmsherpa.readers import LayoutPDFReader

from .base import BaseParser


DEFAULT_LLMSHERPA_API_URL = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"


class LLMSherpaParser(BaseParser):
    def __init__(self,
                 save_chunks: bool = False,
                 save_location: Optional[str] = None,
                 llmsherpa_api_url: str = DEFAULT_LLMSHERPA_API_URL,
                 **kwargs):
        super().__init__(save_chunks, save_location)
        self.pdf_reader = LayoutPDFReader(llmsherpa_api_url)
        lmi_pdf_extractor = PDFReader()
        lmi_docx_extractor = DocxReader()
        self.lmi_file_extractor = {
            ".pdf": lmi_pdf_extractor,
            ".docx": lmi_docx_extractor
        }

    def _parse_file(self,
                   file_path: str,
                   include_section_info: bool = True,
                   category: str = "Sun*") -> List[str]:
        try:
            sherpa_documents = self.pdf_reader.read_pdf(file_path)
            chunks = []
            # TODO: Convert Llmsherpa Document to Llama-index Document
            if len(sherpa_documents.chunks()) > 0:
                for i, chunk in enumerate(sherpa_documents.chunks()):
                    text = chunk.to_context_text(include_section_info)
                    chunks.append(text)
        except Exception as e:
            warnings.warn(
                f"Unexpected error {e} while reading {file_path},"
                f"using default readers"
            )
            # TODO: Implement the metadata generation func
            raw_documents = SimpleDirectoryReader(
                input_files=[file_path], 
                file_extractor=self.lmi_file_extractor,
                filename_as_id=True
            ).load_data()

            chunks = []
            file_name = Path(file_path).name
            for i, doc in enumerate(raw_documents):
                doc.metadata.update({"name": file_name, "category": category})
                doc.excluded_llm_metadata_keys = ["file_path"]
                chunks.append(doc.text)

        return chunks
