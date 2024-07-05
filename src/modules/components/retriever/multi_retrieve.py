from typing import List, Dict, Any, Optional

from llama_index.core import QueryBundle, Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.core.indices.vector_store.retrievers import (
    VectorIndexRetriever
)
from llama_index.core.callbacks import CBEventType, EventPayload
import llama_index.core.instrumentation as instrument
from llama_index.core.schema import NodeWithScore, QueryType

dispatcher = instrument.get_dispatcher(__name__)


class MultiRetrieveRetriever(VectorIndexRetriever):
    def __init__(self,
                 *args,
                 sub_top_k: int = 4,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._sub_top_k = sub_top_k

    @dispatcher.span
    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        embeddings = []
        if self._vector_store.is_embedding_query:
            if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
                for i, embedding_str in enumerate(query_bundle.embedding_strs):
                    embedding = self._embed_model.get_query_embedding(
                        embedding_str
                    )
                    embeddings.append(embedding)
        query_bundle.embedding = embeddings # List[List[float]]

        return self._get_nodes_with_embeddings(query_bundle)

    def _get_nodes_with_embeddings(
        self, query_bundle_with_embeddings: QueryBundle,
    ) -> List[NodeWithScore]:
        self.node_list = []
        embeddings = query_bundle_with_embeddings.embedding
        embedding_strs = query_bundle_with_embeddings.custom_embedding_strs
        query_str = query_bundle_with_embeddings.query_str
        for i, (embedding, embedding_str) in enumerate(zip(embeddings, embedding_strs)):
            query = self._build_vector_store_query(
                embedding, query_str, i
            )
            query_result = self._vector_store.query(
                query, **self._kwargs
            )
            nodes = self._build_node_list_from_query_result(
                query_result
            )
            for node in nodes:
                if node not in self.node_list:
                    self.node_list.append(node)

        return self.node_list

    async def _aget_nodes_with_embeddings(
        self, query_bundle_with_embeddings: QueryBundle
    ) -> List[NodeWithScore]:
        self.node_list = []
        embeddings = query_bundle_with_embeddings.embedding
        query_str = query_bundle_with_embeddings.query_str
        for i, embedding in enumerate(embeddings):
            query = self._build_vector_store_query(
                embedding, query_str, i
            )
            query_result = await self._vector_store.aquery(
                query, **self._kwargs
            )
            nodes = self._build_node_list_from_query_result(
                query_result
            )
            for node in nodes:
                if node not in self.node_list:
                    self.node_list.append(node)

        return self.node_list

    def _build_vector_store_query(
        self, embedding, query_str, index: int
    ) -> VectorStoreQuery:
        return VectorStoreQuery(
            query_embedding=embedding,
            similarity_top_k=self._similarity_top_k if index == 0 else self._sub_top_k,
            node_ids=self._node_ids,
            doc_ids=self._doc_ids,
            query_str=query_str,
            mode=self._vector_store_query_mode,
            alpha=self._alpha,
            filters=self._filters,
            sparse_top_k=self._sparse_top_k,
        )

    def log_output(self, output_dict: dict) -> Dict[str, Any]:
        node_str = ""
        for i, node in enumerate(self.node_list):
            node_str += f"Node {i+1}: {node.text}\n"
        output_dict.update(
            {"MuRe": node_str}
        )
        return output_dict