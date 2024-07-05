from pathlib import Path

from llama_index.core.prompts import ChatMessage, ChatPromptTemplate, MessageRole


with (
    Path(__file__).parents[0] / Path("chunking_v2.txt")
).open("r") as f:
    CHUNKING_SYSTEM_PROMPT = f.read()

CHUNKING_SYSTEM_PROMPT_TMPL = ChatMessage(
    role=MessageRole.SYSTEM,
    content=CHUNKING_SYSTEM_PROMPT
)

CHUNKING_USER_PROMPT_TMPL_STR = (
    "Input:\n"
    "Title: {title}\n"
    "Content: {passage}\n"
    "Output: "
)

CHUNKING_USER_PROMPT_TMPL = ChatMessage(
    role=MessageRole.USER,
    content=CHUNKING_USER_PROMPT_TMPL_STR
)

CHUNKING_PROMPT_TMPL = ChatPromptTemplate(
    [
        CHUNKING_SYSTEM_PROMPT_TMPL,
        CHUNKING_USER_PROMPT_TMPL
    ]
)