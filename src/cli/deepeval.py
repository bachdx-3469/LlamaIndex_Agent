import typer
import pandas as pd

from llama_index.core import (
    StorageContext,
    load_index_from_storage
)
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

from src.utils import initialize
from src.modules.pipelines.evaluator import (
    faithfulnessMetric,
    contextualprecisionMetric,
    contextualRecall,
    contextualrelevancy
)
from src.utils import display_deepeval_df

def main(
    dotenv_path: str = "local.env",
    output_path: str = "data/test/human-eval2/",
    input_path: str = "data/test/Human/Commitment - Mai.csv",
    persist_dir: str = "index_v2",
    top_k: int = 4,
    similarity_cutoff: float = 0.8,
    use_hyde: bool = False,
    threshold: float = 0.7, 
    include_reason: bool = True
):
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
        
    faithfulnessScores = []
    faithfulnessReasons = []
    
    contextualprecisionScores = []
    contextualprecisionReasons = []
    
    contextualRecallScores = []
    contextualRecallReasons = []
    
    contextualrelevancyScores = []
    contextualrelevancyReasons = []
    
    actual_ouputs = []
    query_input = []
    expected_outputs = []
    retrievals_chunk = []

    data = pd.read_csv(input_path)
    for i in range(len(data['Question (English)'])):
        query = data['Question (English)'][i]
        response = query_engine.query(query)
        actual_ouput = response.response
        retrieval_context = response.source_nodes
        retrieval_context  = [retrieval.node.text for retrieval in retrieval_context ]
        expected_output = data['Answer'][i]
        
        query_input.append(query)
        actual_ouputs.append(actual_ouput)
        expected_outputs.append(expected_output)
        retrievals_chunk.append(retrieval_context)
        
        # Get Metric faithfulnessMetric
        faithfulnessScore, faithfulnessReason = faithfulnessMetric(input=query,
                                                    actual_output=actual_ouput,
                                                    retrieval_context=retrieval_context,
                                                    threshold=threshold, include_reason=include_reason
                                                )
        faithfulnessScores.append(faithfulnessScore)
        faithfulnessReasons.append(faithfulnessReason)
        
        # Get contextualprecisionMetric
        contextualprecisionScore, contextualprecisionReason = contextualprecisionMetric(
            input=query,actual_output=actual_ouput,
            expected_output=expected_output, retrieval_context=retrieval_context,
            threshold=threshold, include_reason=include_reason
        )
        
        contextualprecisionScores.append(contextualprecisionScore)
        contextualprecisionReasons.append(contextualprecisionReason)
        
        # contextualRecallMetrics
        contextualRecallScore, contextualRecallReason = contextualRecall(
            input=query,actual_output=actual_ouput,
            expected_output=expected_output, retrieval_context=retrieval_context,
            threshold=threshold, include_reason=include_reason
        )
        contextualRecallScores.append(contextualRecallScore)
        contextualRecallReasons.append(contextualRecallReason)
        
        # contextualRecallMetrics
        contextualrelevancyScore, contextualrelevancyReason = contextualrelevancy(
            input=query,actual_output=actual_ouput,
            expected_output=expected_output, retrieval_context=retrieval_context,
            threshold=threshold, include_reason=include_reason
        )
        contextualrelevancyScores.append(contextualrelevancyScore)
        contextualrelevancyReasons.append(contextualrelevancyReason)
    
    display_deepeval_df(output_path=output_path, questions=query_input, expected_outputs=expected_outputs,
                        actual_ouputs=actual_ouputs, retrievals_chunk=retrievals_chunk,
                        faithfulnessScores=faithfulnessScores,faithfulnessReasons=faithfulnessReasons,
                        contextualprecisionScores=contextualprecisionScores,
                        contextualprecisionReasons=contextualprecisionReasons,
                        contextualRecallScores=contextualRecallScores,
                        contextualRecallReasons=contextualRecallReasons,
                        contextualrelevancyScores=contextualrelevancyScores,
                        contextualrelevancyReasons=contextualrelevancyReasons)
    
if __name__ == '__main__':
    typer.run(main)
