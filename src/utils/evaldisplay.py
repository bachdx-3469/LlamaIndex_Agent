import pandas as pd
import os
from typing import List
from llama_index.core.evaluation import EvaluationResult
from llama_index.core.base.response.schema import RESPONSE_TYPE

def display_eval_df(output_path: str, questions: List[str], responses: List[RESPONSE_TYPE], eval_results: List[EvaluationResult]) -> None:
    eval_df = pd.DataFrame(
        {
            "Question": [q.strip() for q in questions],
            "Response": [None if str(r).strip() == [] else str(r).strip() for r in responses],
            # "Source": [r.source_nodes[0].node.text + "..." for r in responses],
            "Score": [eval_result.score for eval_result in eval_results],
            "Evaluation Result": ["Pass" if eval_result.passing else "Fail" for eval_result in eval_results],
            "Reasoning": [eval_result.feedback for eval_result in eval_results],
        }
    )
    #save the evaluation result to an excel file
    eval_df.to_excel(os.path.join(output_path, "output_eval.xlsx"), index=False)
    
def display_deepeval_df(output_path: str, questions: List[str], actual_ouputs: List[str], expected_outputs: List[str],
                    retrievals_chunk: list,
                    faithfulnessScores: list, faithfulnessReasons: List[str],
                    contextualprecisionScores: list, contextualprecisionReasons: List[str],
                    contextualRecallScores: list, contextualRecallReasons: List[str],
                    contextualrelevancyScores: list, contextualrelevancyReasons: List[str],) -> None:
    eval_df = pd.DataFrame(
        {
            "Question": questions,
            "Actual Output":  actual_ouputs,
            "Expected Output": expected_outputs,
            "Retrievals Chunk": retrievals_chunk,
            # "Source": [r.source_nodes[0].node.text + "..." for r in responses],
            "faithfulnessScores": faithfulnessScores,
            "faithfulnessReasons": faithfulnessReasons,
            "contextualprecisionScores": contextualprecisionScores,
            "contextualprecisionReasons": contextualprecisionReasons,
            "contextualRecallScores": contextualRecallScores,
            "contextualRecallReasons": contextualRecallReasons,
            "contextualrelevancyScores": contextualrelevancyScores,
            "contextualrelevancyReasons": contextualrelevancyReasons,
        }
    )
    #save the evaluation result to an excel file
    eval_df.to_excel(os.path.join(output_path, "output_eval.xlsx"), index=False)
