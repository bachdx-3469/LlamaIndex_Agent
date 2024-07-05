from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.tools.types import ToolMetadata, ToolOutput
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from typing import List, Any, Optional
from llama_index.core import (
    StorageContext,
    load_index_from_storage
)
from llama_index.core.tools.retriever_tool import RetrieverTool
from src.utils import initialize

initialize(".env") 

# # Đọc dữ liệu từ thư mục và tải vào VectorStoreIndex
# documents = SimpleDirectoryReader("data").load_data()
storage_context = StorageContext.from_defaults(
            persist_dir='index'
        )
index = load_index_from_storage(storage_context)
retriever = index.as_retriever()

# Implement a retriever that uses VectorStoreIndex
# class VectorStoreRetriever(BaseRetriever):
#     def __init__(self, index: VectorStoreIndex):
#         self.index = index
    
#     def retrieve(self, query: str) -> List[NodeWithScore]:
#         results = self.index.query(query)
#         nodes_with_scores = [NodeWithScore(node=result.node, score=result.score) for result in results]
#         return nodes_with_scores
    
#     async def aretrieve(self, query: str) -> List[NodeWithScore]:
#         results = self.index.query(query)
#         nodes_with_scores = [NodeWithScore(node=result.node, score=result.score) for result in results]
#         return nodes_with_scores

# # Define the retriever with the VectorStoreIndex
# retriever = VectorStoreRetriever(index=index)

# Define the tool metadata
metadata = ToolMetadata(name="retriever_tool", description="Retrieve documents based on a query")

# Define any node postprocessors (optional)
node_postprocessors = []

retriever_tool = RetrieverTool(
    retriever=retriever,
    metadata=metadata,
    node_postprocessors=node_postprocessors,
)

# Implement the query engine using the retriever tool
class QueryEngine:
    def __init__(self, retriever_tool: RetrieverTool):
        self.retriever_tool = retriever_tool
    
    def query(self, query_str: str) -> ToolOutput:
        return self.retriever_tool.call(query_str)

    async def aquery(self, query_str: str) -> ToolOutput:
        return await self.retriever_tool.acall(query_str)

# Create the query engine
query_engine = QueryEngine(retriever_tool=retriever_tool)

# Implement the RAG agent
class RAGAgent:
    def __init__(self, query_engine: QueryEngine):
        self.query_engine = query_engine

    def generate_response(self, query: str) -> str:
        retrieved_docs = self.query_engine.query(query).content
        # Dummy response generation logic, replace with actual generation logic
        response = f"Generated response based on the retrieved documents: {retrieved_docs}"
        return response

    async def agenerate_response(self, query: str) -> str:
        retrieved_docs = (await self.query_engine.aquery(query)).content
        # Dummy response generation logic, replace with actual generation logic
        response = f"Generated response based on the retrieved documents: {retrieved_docs}"
        return response

# Create the RAG agent
rag_agent = RAGAgent(query_engine=query_engine)

# Example usage
query = "When installing high risk software, Employee must not do what?"
response = rag_agent.generate_response(query)
print(response)

# Async example usage
import asyncio
response_async = asyncio.run(rag_agent.agenerate_response(query))
print(response_async)
