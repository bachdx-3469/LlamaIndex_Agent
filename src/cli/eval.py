from typing import List
from tqdm import tqdm
import typer
import pandas as pd

import llama_index.core.instrumentation as instrument
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)

from src.utils import initialize
from src.modules.pipelines.qna import get_qna_pipeline_from_name
from src.modules.pipelines.evalulate import get_eval_pipeline_from_name


dispatcher = instrument.get_dispatcher(__name__)


def main(
        qna_pipeline: str,
        eval_pipeline: str,
        metrics: List[str] = ["FaithfulnessMetric",
                              "ContextualPrecisionMetric",
                              "ContextualRecallMetric",
                              "ContextualRelevancyMetric"],
        dotenv_path: str = "local.env",
        output_path: str = "data/test/human-eval2/",
        input_path: str = "data/test/Human/Commitment - Mai.csv",
        persist_dir: str = "index",
        threshold: float = 0.7,
        include_reason: bool = True,
        do_eval: bool = False
):
    initialize(dotenv_path)
    callback_manager = Settings.callback_manager

    # Prepare Query Engine
    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir
    )
    index = load_index_from_storage(storage_context)
    qna_pipeline = get_qna_pipeline_from_name(
        qna_pipeline,
        index=index,
        debug=False
    )

    # Prepare Eval Pipeline
    eval_pipeline = get_eval_pipeline_from_name(
        eval_pipeline, threshold=threshold, include_reason=include_reason)
    for metric in metrics:
        eval_pipeline.add_metric(metric)
    # Evaluate
    data = pd.read_csv(input_path)
    for i in tqdm(range(len(data['Question (English)'])), desc="Evaluating..."):
        query = data['Question (English)'][i]
        response, output_dict = qna_pipeline.run(query)
        actual_output = response.response
        retrieval_context = response.source_nodes
        retrieval_context = [retrieval.node.text for retrieval in retrieval_context]
        expected_output = data['Answer'][i]

        if do_eval:
            try:
                eval_pipeline.run(
                    query, actual_output, retrieval_context, expected_output, output_dict
                )
            except:
                eval_pipeline.handle_error(
                    query, actual_output, retrieval_context, expected_output, output_dict
                )
        else:
            eval_pipeline.handle_error(
                query, actual_output, retrieval_context, expected_output, output_dict
            )

    eval_pipeline.display_result(output_path, output_dict)


if __name__ == '__main__':
    typer.run(main)
