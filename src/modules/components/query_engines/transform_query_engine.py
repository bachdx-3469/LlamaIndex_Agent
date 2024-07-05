from typing import List, Optional, Sequence

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.schema import NodeWithScore, QueryBundle


class TransformQueryEngine(BaseQueryEngine):
    def __init__(self,
                 query_engine: BaseQueryEngine,
                 query_transforms: List[BaseQueryTransform],
                 transform_metadata: Optional[dict] = None,
                 callback_manager: Optional[CallbackManager] = None):
        self._query_engine = query_engine
        self._query_transforms = query_transforms
        self._transform_metadata = transform_metadata
        super().__init__(callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        prompt_modules = {}
        for i, query_transform in enumerate(self._query_transforms):
            prompt_modules[f"query_transform_{i}"] = query_transform
        prompt_modules["query_engine"] = self._query_engine

        return prompt_modules

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        for query_transform in self._query_transforms:
            query_bundle = query_transform.run(
                query_bundle, metadata=self._transform_metadata
            )
        return self._query_engine.retrieve(query_bundle)

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        for query_transform in self._query_transforms:
            query_bundle = query_transform.run(
                query_bundle, metadata=self._transform_metadata
            )
        return self._query_engine.synthesize(
            query_bundle=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        for query_transform in self._query_transforms:
            query_bundle = query_transform.run(
                query_bundle, metadata=self._transform_metadata
            )
        return await self._query_engine.asynthesize(
            query_bundle=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        for query_transform in self._query_transforms:
            query_bundle = query_transform.run(
                query_bundle, metadata=self._transform_metadata
            )
        return self._query_engine.query(query_bundle)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        for query_transform in self._query_transforms:
            query_bundle = query_transform.run(
                query_bundle, metadata=self._transform_metadata
            )
        return await self._query_engine.aquery(query_bundle)