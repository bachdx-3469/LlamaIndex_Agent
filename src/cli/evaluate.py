import typer

from llama_index.core import (
    StorageContext,
    load_index_from_storage
)
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.evaluation import FaithfulnessEvaluator

from src.utils import initialize
from src.utils import display_eval_df
from src.modules.pipelines.evaluator import evaluator_func

# Perform question and answer using RAG system
def main(
        dotenv_path: str = "local.env",
        output_path: str = "test/eval",
        query: str = None,
        query_dir: str = "test/eval/question.txt",
        persist_dir: str = "index",
        top_k: int = 4,
        similarity_cutoff: float = 0.8,
        use_hyde: bool = False
) -> None:
    """
    Executes the main functionality of the Q&A system.

    Args:
        dotenv_path (str, optional): Path to the .env file. Defaults to None.
        output_path (str, optional): Path to the output file for evaluation results. Defaults to None.
        query (str, optional): The query to be executed. Defaults to None.
        query_dir (str, optional): Path to the file containing multiple queries. Defaults to None.
        persist_dir (str, optional): Directory to persist the index. Defaults to "index".
        top_k (int, optional): Number of top similar nodes to retrieve. Defaults to 4.
        similarity_cutoff (float, optional): Similarity cutoff value. Defaults to 0.8.
        use_hyde (bool, optional): Whether to use Hyde query transformation. Defaults to False.

    Returns:
        None
    """
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
        hyde = HyDEQueryTransform(include_original=True)
        query_engine = TransformQueryEngine(query_engine,
                                            query_transform=hyde)

    faith_evaluator = FaithfulnessEvaluator()

    if query:
        response = query_engine.query(query)
        eval_result = evaluator_func(response, faith_evaluator)
        print("Response: ", response.response)
        print("Score: ", eval_result.score)
        print("Evaluation Result: ", eval_result.passing)
        print("Reasoning: ", eval_result.feedback)

    elif query_dir:
        with open(query_dir, 'r') as r_file:
            questions = r_file.readlines()

        responses = []
        eval_results = []

        for question in questions:
            response = query_engine.query(question)
            eval_result = evaluator_func(response, faith_evaluator)
            eval_results.append(eval_result)
            responses.append(response)
            print(response)

        # convert2exel to compare
        display_eval_df(output_path, questions, responses, eval_results)


if __name__ == '__main__':
    typer.run(main)
