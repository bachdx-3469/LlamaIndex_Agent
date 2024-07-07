from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from src.utils import initialize

initialize(".env")

storage_context = StorageContext.from_defaults(
            persist_dir="index"
        )
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
response = query_engine.query("When installing high risk software, what must employee do with Anti Virus and Firewall?")
print(str(response))
