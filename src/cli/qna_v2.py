import typer

from llama_index.core import (
    StorageContext,
    load_index_from_storage
)

from src.modules.pipelines.qna import get_qna_pipeline_from_name
from src.utils import initialize
from src.utils import display2exel


# Perform question and answer using RAG system
def main(
        dotenv_path: str,
        qna_pipeline: str,
        query: str = None,
        query_dir: str = "test/qna/question.txt",
        persist_dir: str = "index",
        top_k: int = 2,
        similarity_cutoff: float = 0.75
) -> None:
    initialize(dotenv_path)

    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir
    )
    index = load_index_from_storage(storage_context)
    qna_pipeline = get_qna_pipeline_from_name(
        qna_pipeline,
        index=index,
        top_k=top_k,
        similarity_cutoff=similarity_cutoff
    )
    if query:
        response, _ = qna_pipeline.run(query)
        print(response.response)
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
            response, _ = qna_pipeline.run(question)
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
