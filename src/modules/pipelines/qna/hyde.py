from llama_index.core import Settings
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import QueryType

from src.modules.components.query_engines.transform_query_engine import TransformQueryEngine
from src.modules.components.query_transforms.hyde import HyDE
from .base import BaseQnAPipeline


class DefaultHyDEQnAPipeline(BaseQnAPipeline):
    def __init__(self,
                 index: VectorStoreIndex,
                 debug: bool = True,
                 top_k: int = 4,
                 similarity_cutoff: float = 0.75,
                 **kwargs):
        super().__init__(debug=debug)
        sub_query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
            ],
            **kwargs
        )
        hyde = HyDE(
            include_original=True, callback_manager=Settings.callback_manager, debug=debug
        )
        self.query_engine = TransformQueryEngine(
            sub_query_engine,
            [hyde],
            callback_manager=Settings.callback_manager
        )
        self.components.extend([hyde])

    def _query(self, query: QueryType) -> RESPONSE_TYPE:
        return self.query_engine.query(query)
