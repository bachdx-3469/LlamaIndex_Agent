from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType


INT_NUMBER_TO_STRING = {
    1: "ONE",
    2: "TWO",
    3: "THREE",
    4: "FOUR",
    5: "FIVE",
    6: "SIX",
    7: "SEVEN",
    8: "EIGHT",
    9: "NINE",
    10: "TEN"
}

DATA_GENERATION_TMPL = (
    "{example_questions}\n"
    "Above is some example questions, based on below context\n"
    "--------------------------------\n"
    "{context}\n"
    "--------------------------------\n"
    "Make EXACTLY {num_questions} similar to example questions"
)
DATA_GENERATION_PROMPT = PromptTemplate(
    DATA_GENERATION_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)