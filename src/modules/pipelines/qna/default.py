from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import QueryType

from .base import BaseQnAPipeline


class DefaultQnAPipeline(BaseQnAPipeline):
    def __init__(self,
                 index: VectorStoreIndex,
                 debug: bool = True,
                 top_k: int = 4,
                 similarity_cutoff: float = 0.75,
                 **kwargs):
        super().__init__(debug=debug)
        self.query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
            ],
            **kwargs
        )

    def _query(self, query: QueryType) -> RESPONSE_TYPE:
        return self.query_engine.query(query)
