from pathlib import Path

from llama_index.core.prompts import ChatMessage, ChatPromptTemplate, MessageRole


with (
    Path(__file__).parents[0] / Path("decompose.txt")
).open("r") as f:
    DECOMPOSE_SYSTEM_PROMPT_TMPL_STR = f.read()

DECOMPOSE_SYSTEM_PROMPT_TMPL = ChatMessage(
    role=MessageRole.SYSTEM,
    content=DECOMPOSE_SYSTEM_PROMPT_TMPL_STR
)

DECOMPOSE_USER_PROMPT_TMPL_STR = (
    "Input:\n"
    "Passage: \n{passage}\n"
    "Output: "
)

DECOMPOSE_USER_PROMPT_TMPL = ChatMessage(
    role=MessageRole.USER,
    content=DECOMPOSE_USER_PROMPT_TMPL_STR
)

DECOMPOSE_PROMPT_TMPL = ChatPromptTemplate(
    [
        DECOMPOSE_SYSTEM_PROMPT_TMPL,
        DECOMPOSE_USER_PROMPT_TMPL
    ]
)