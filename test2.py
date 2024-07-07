from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.tools.types import ToolMetadata, ToolOutput
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.tools import RetrieverTool

from src.utils import initialize

initialize(".env")

from llama_index.core import PromptTemplate

react_system_header_str = """\

You are designed to help with a Question and Answer task, but the kownledge to \
answer this task is very special, so you must use tools to get data. But do not \ 
need all of them, so you must think step by step to get some usefull infomation to \
answer question. 

You have access to the following tools:
{tool_desc}

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

After you use retriever tool. Base on the information you get, you will have to think and filter out the truly important information and then rely on that to give an answer to my question.
Output Format: 

```
Thought: I have a lot information, so I must filter and chosse some important information.
Answer: [your answer here]
```

Finally, you must give me the output as your answer after you think carefully.
'''
Thought: I have enough information to answer the question.
Answer: [your answer here]
'''

"""

react_system_prompt = PromptTemplate(react_system_header_str)

# Load documents and build index
storage_context = StorageContext.from_defaults(
            persist_dir="index"
        )
index = load_index_from_storage(storage_context)
retriever = index.as_query_engine(similarity_top_k=10)

metadata=ToolMetadata(name="react_retriever_tool", description="A retriever tool")

retrieval_tool = RetrieverTool(
    retriever=retriever,
    metadata=metadata
)

query = "Take out the information related to the following question and use it to answer the following question: When installing high risk software, what must employee do with Anti Virus and Firewall?"

GEN_SYS_PROMPT_STR = """\
You are an agent who will answer questions using the tool retriever you are provided to retrieve relevant information. From there, based on the information you get, you will have to think and filter out the truly important information and then rely on that to give an answer to my question.

And please give me the output as your answer after you think carefully.
Only use tool that i give me that retrieval_tool. 
The question is that: 
{query} \

Retrieve all information relate with this query, after that using this to answer question. 
"""

agent = ReActAgent.from_tools(tools=[retrieval_tool], verbose=True)
agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})
agent.reset()

response = agent.chat(GEN_SYS_PROMPT_STR)
print(str(response))
# print(response.response)

# response_gen = agent.stream_chat(GEN_SYS_PROMPT_STR)
# response_gen.print_response_stream()

