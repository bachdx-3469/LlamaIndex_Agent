from typing import Optional, Dict, Any
from ast import literal_eval

from llama_index.core import Settings, QueryBundle
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.service_context_elements.llm_predictor import (
    LLMPredictorType,
)
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.utils import print_text
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload

from src.modules.components.prompts.decompose import DECOMPOSE_PROMPT_TMPL


class HyDEDecomposeTransform(BaseQueryTransform):
    def __init__(self,
                 llm: Optional[LLMPredictorType] = None,
                 decompose_prompt: Optional[BasePromptTemplate] = DECOMPOSE_PROMPT_TMPL,
                 include_original: bool = True,
                 callback_manager: Optional[CallbackManager] = None,
                 debug: bool = True):
        super().__init__()
        self._llm = llm or Settings.llm
        self._decompose_prompt = decompose_prompt
        self._include_original = include_original
        self.callback_manager = callback_manager
        self.debug = debug

    def _get_prompts(self) -> PromptDictType:
        return {"decompose_prompt": self._decompose_prompt}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        if "decompose_prompt" in prompts_dict:
            self._decompose_prompt = prompts_dict["decompose_prompt"]

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            hypothetical_doc = query_bundle.embedding_strs[0]
            query_str = query_bundle.query_str
            query = self._decompose_prompt.format_messages(
                passage=hypothetical_doc,
            )
            self.decomposed_str = self._llm.chat(query)
            list_decomposed_str: list = literal_eval(self.decomposed_str.message.content)
            if self.debug:
                print_text("Decomposed HyDE:\n", color="green")
                for decom_str in list_decomposed_str:
                    print_text(f"- {decom_str}\n", color="pink")
            if self._include_original:
                list_decomposed_str.insert(0, query_str)
            query_event.on_end(payload={EventPayload.RESPONSE: list_decomposed_str})

        return QueryBundle(
            query_str=query_str,
            custom_embedding_strs=list_decomposed_str
        )

    def log_output(self, output_dict: dict) -> Dict[str, Any]:
        output_dict.update(
            {"Decompose HyDE": self.decomposed_str}
        )
        return output_dict