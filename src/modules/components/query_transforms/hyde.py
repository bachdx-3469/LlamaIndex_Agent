from typing import Dict, Optional, Any

from llama_index.core.utils import print_text
from llama_index.core import QueryBundle
from llama_index.core.base.response.schema import Response
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload
import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.events.query import QueryStartEvent, QueryEndEvent


dispatcher = instrument.get_dispatcher(__name__)


class HyDE(HyDEQueryTransform):
    def __init__(self,
                 *args,
                 callback_manager: Optional[CallbackManager] = None,
                 debug: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.callback_manager = callback_manager
        self.debug = debug

    @dispatcher.span
    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            query_str = query_bundle.query_str
            self.hypothetical_doc = self._llm.predict(self._hyde_prompt, context_str=query_str)
            if self.debug:
                print_text(f"Hypothetical Doc:\n", color="green")
                print_text(self.hypothetical_doc + "\n", color="pink")
            embedding_strs = [self.hypothetical_doc]
            if self._include_original:
                embedding_strs.extend(query_bundle.embedding_strs)

            query_event.on_end(payload={EventPayload.RESPONSE: self.hypothetical_doc})

        return QueryBundle(
            query_str=query_str,
            custom_embedding_strs=embedding_strs,
        )

    def log_output(self, output_dict: dict) -> Dict[str, Any]:
        output_dict.update(
            {"HyDE": self.hypothetical_doc}
        )
        return output_dict
