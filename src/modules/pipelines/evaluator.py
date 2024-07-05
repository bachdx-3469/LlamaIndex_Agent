from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.base.response.schema import RESPONSE_TYPE

# DeepEval Metrics
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

def evaluator_func(response: RESPONSE_TYPE, faith_evaluator: FaithfulnessEvaluator):
    eval_result = faith_evaluator.evaluate_response(response=response)
    return eval_result

def faithfulnessMetric(input: str, actual_output: str, retrieval_context: list, threshold: float = 0.7, include_reason: bool = True):
    metric = FaithfulnessMetric(
        threshold=threshold,
        include_reason=include_reason
    )
    
    test_case = LLMTestCase(
        input = input, 
        actual_output= actual_output,
        retrieval_context=retrieval_context
    )
    
    metric.measure(test_case)
    return metric.score, metric.reason

def contextualprecisionMetric(input: str, actual_output: str, expected_output:str, retrieval_context: list, threshold: float = 0.7, include_reason: bool = True):
    metric = ContextualPrecisionMetric(
        threshold=threshold,
        include_reason=include_reason
    )
    
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )

    metric.measure(test_case)
    return metric.score, metric.reason
    
def contextualRecall(input: str, actual_output: str, expected_output:str, retrieval_context: list, threshold: float = 0.7, include_reason: bool = True):
    metric = ContextualRecallMetric(
        threshold=threshold,
        include_reason=include_reason
    )
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )

    metric.measure(test_case)
    return metric.score, metric.reason
    
def contextualrelevancy(input: str, actual_output: str, expected_output:str, retrieval_context: list, threshold: float = 0.7, include_reason: bool = True):
    metric = ContextualRelevancyMetric(
        threshold=threshold,
        include_reason=include_reason
    )
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )

    metric.measure(test_case)
    return metric.score, metric.reason
