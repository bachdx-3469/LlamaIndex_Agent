from typing import Union, List
import os
import typer
from tqdm import tqdm

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)

from src.modules.pipelines.chunking import get_chunk_pipeline_from_name
from src.utils import initialize


def main(dotenv_path: str,
         pipeline: str,
         data_path: List[str],
         persist_dir: str = "index",
         include_section_info: bool = True,
         category: str = "Sun*"):
    """
    Main function to create VectorStoreIndex from files in folder
    Args:
        dotenv_path (str): Path to dotenv file
        pipeline (str): Type of chunking pipeline to use
        data_dir (str): Path to folder that contains files
        persist_dir (str): Path to save the VectorStoreIndex to disk
        category (str): category of the files in folder
    """
    initialize(dotenv_path)

    chunk_pipeline = get_chunk_pipeline_from_name(
        pipeline, include_section_info=include_section_info
    )
    documents = []
    if os.path.isdir(data_path[0]):
        for file in tqdm(os.listdir(data_path[0]), desc="Creating documents"):
            documents.extend(
                chunk_pipeline.run(
                    f"{data_path[0]}/{file}",
                    category=category)
            )
    else:
        for file in tqdm(data_path, desc="Creating documents"):
            documents.extend(
                chunk_pipeline.run(
                    file,
                    category=category
                )
            )
    # If the index exist, simply load it up
    # Then refresh the index with docs
    if os.path.exists(persist_dir):
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir
        )
        index = load_index_from_storage(storage_context)
        index.refresh_ref_docs(
            documents,
            insert_kwargs={"show_progress": True}
        )
        index.storage_context.persist(persist_dir)
    else:
        index = VectorStoreIndex.from_documents(
            documents, show_progress=True
        )
        index.storage_context.persist(persist_dir)


if __name__ == '__main__':
    typer.run(main)
