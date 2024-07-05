from typing import List
from ast import literal_eval
from tqdm import tqdm
from pathlib import Path

from llama_index.core import Settings, Document
from llama_index.core.callbacks.schema import CBEventType, EventPayload
import llama_index.core.instrumentation as instrument

from src.modules.components.prompts.chunking_v3 import CHUNKING_PROMPT_TMPL
from src.modules.components.parser import PymuPDFParser
from .base import BaseChunkPipeline


dispatcher = instrument.get_dispatcher(__name__)


class AgenticChunkPipelineV3(BaseChunkPipeline):
    def __init__(self,
                 **kwargs):
        self.parser = PymuPDFParser(
            save_chunks=True
        )
        self.llm = Settings.llm
        self.prompt_helper = Settings.prompt_helper
        self.callback_manager = Settings.callback_manager
        self.chunking_prompt = CHUNKING_PROMPT_TMPL

    @dispatcher.span
    def run(self,
            file_path: str,
            category: str = "Sun*") -> List[Document]:
        file_name = Path(file_path).name
        doc_chunks = self.parser.parse_file(file_path)

        documents = []
        with self.callback_manager.as_trace("agentic_chunking"):
            for i, cur_doc in tqdm(enumerate(doc_chunks), desc="Agentic chunking..."):
                with self.callback_manager.event(
                    CBEventType.TEMPLATING,
                    payload={
                        EventPayload.TEMPLATE: self.chunking_prompt,
                        EventPayload.TEMPLATE_VARS: cur_doc
                    }
                ) as template_event:
                    query = self.chunking_prompt.format_messages(
                        passage=cur_doc,
                        title=file_name
                    )
                    template_event.on_end(
                        payload={EventPayload.PROMPT: query}
                    )
                with self.callback_manager.event(
                    CBEventType.QUERY,
                    payload={
                        EventPayload.QUERY_STR: query
                    }
                ) as query_event:
                    response = self.llm.chat(query)
                    text_chunks = literal_eval(response.message.content)
                    query_event.on_end(
                        payload={
                            EventPayload.COMPLETION: text_chunks
                        }
                    )

                for text_chunk in text_chunks:
                    documents.append(
                        Document(
                            text=text_chunk,
                            id_=f"{file_name}_{i}",
                            metadata={
                                "name": file_name,
                                "category": category
                            },
                            excluded_llm_metadata_keys=["file_path", "name"]
                        )
                    )

        return documents