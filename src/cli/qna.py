import typer

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from src.modules.components.query_transforms.hyde import HyDE
from src.utils import initialize
from src.utils import display2exel


# Perform question and answer using RAG system
def main(
        dotenv_path: str = "local.env",
        query: str = None,
        query_dir: str = "test/qna/question.txt",
        persist_dir: str = "index",
        top_k: int = 4,
        similarity_cutoff: float = 0.8,
        use_hyde: bool = False
) -> None:
    initialize(dotenv_path)

    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir
    )
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        ]
    )
    if use_hyde:
        hyde = HyDE(include_original=True,
                    callback_manager=Settings.callback_manager)
        query_engine = TransformQueryEngine(query_engine,
                                            query_transform=hyde)
    if query:
        response = query_engine.query(query)
        print(response.response)
        if use_hyde:
            nodes = response.source_nodes
            for i, node in enumerate(nodes):
                print(f"Node {i}: {node.text}\n")
    elif query_dir:
        r_file = open(query_dir, 'r')
        questions = []
        responses = []

        # read question from .txt file
        while True:
            question = r_file.readline()

            if not question:
                break

            questions.append(question)
        r_file.close()

        for question in questions:
            response = query_engine.query(question)
            response = response.response
            responses.append(response)

        # Write file
        w_file = open('test/qna/response.txt', 'w')
        for response in responses:
            w_file.write(response + '\n')
        w_file.close()

        # convert2exel to compare
        display2exel()


if __name__ == '__main__':
    typer.run(main)
