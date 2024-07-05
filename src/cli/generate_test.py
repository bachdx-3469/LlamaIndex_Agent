from typing import List, Dict, Tuple

import re
import typer
import random
import pandas as pd
from tqdm import tqdm

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
import llama_index.core.instrumentation as instrument

from src.modules.components.prompts.data_generation import (
    DATA_GENERATION_PROMPT,
    INT_NUMBER_TO_STRING)
from src.utils import initialize


dispatcher = instrument.get_dispatcher(__name__)


@dispatcher.span
def generate_test_samples(
        callback_manager: CallbackManager,
        llm,
        prompt_helper,
        sources: List[str],
        sample_questions_mapping: Dict[str, list],
        data_folder: str,
        extension: str,
        num_iter: int,
        questions_per_iter: iter
) -> Tuple[List[str], List[str]]:
    cleaned_questions = []
    q_sources = []
    with callback_manager.as_trace("generate_test_data"):
        for _ in tqdm(range(num_iter), desc="Generating test data..."):
            # pick a source
            source = sources[random.randint(0, len(sources) - 1)]
            q_sources.extend([source] * questions_per_iter)
            source_path = f"{data_folder}/{source}{extension}"
            # get 2 example questions from source
            sample_questions = sample_questions_mapping[source]
            sample_questions = random.sample(sample_questions, 2)
            # prompt
            sample_q_str = ""
            with callback_manager.as_trace("templating"):
                with callback_manager.event(
                    CBEventType.TEMPLATING,
                    payload={EventPayload.TEMPLATE: sample_questions}
                ) as templating_event:
                    for i, sample_question in enumerate(sample_questions):
                        sample_q_str += f"{i+1}: {sample_question}\n"
                    prompt = DATA_GENERATION_PROMPT.partial_format(
                        example_questions=sample_q_str,
                        num_questions=INT_NUMBER_TO_STRING[questions_per_iter]
                    )
                    documents = SimpleDirectoryReader(input_files=[source_path]).load_data()
                    doc_texts = [document.text for document in documents]
                    context = "".join(prompt_helper.repack(prompt, doc_texts))
                    prompt = prompt.format(context=context)
                    templating_event.on_end(
                        payload={EventPayload.PROMPT: prompt}
                    )

            with callback_manager.as_trace("query"):
                with callback_manager.event(
                    CBEventType.QUERY,
                    payload={EventPayload.QUERY_STR: prompt}
                ) as query_event:
                    response = llm.complete(prompt).text
                    result = response.strip().split("\n")
                    _cleaned_questions = [
                        re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
                    ]
                    _cleaned_questions = [
                      question for question in _cleaned_questions if len(question) > 0
                    ][: questions_per_iter]
                    cleaned_questions.extend(_cleaned_questions)
                    query_event.on_end(
                        payload={EventPayload.COMPLETION: _cleaned_questions}
                    )

    return cleaned_questions, q_sources

def main(dotenv_path: str,
         example_files: List[str],
         data_folder: str,
         output_folder: str = "data/test/Bot",
         output_file: str = "output.csv",
         extension: str = ".pdf",
         num_iter: int = 10,
         questions_per_iter: int = 2):
    initialize(dotenv_path)
    prompt_helper = Settings.prompt_helper
    llm = Settings.llm
    callback_manager = Settings.callback_manager

    sample_questions_mapping = {}
    sources = []
    for example_file in example_files:
        dataframe = pd.read_csv(example_file)
        for i in range(len(dataframe) - 1):
            source = dataframe.iloc[i]["Source"]
            if source not in sources:
                sources.append(source)
            q = dataframe.iloc[i]["Question (English)"]
            if source not in sample_questions_mapping.keys():
                sample_questions_mapping[source] = [q]
            else:
                sample_questions_mapping[source].append(q)

    cleaned_questions, q_sources = generate_test_samples(
        callback_manager,
        llm,
        prompt_helper,
        sources,
        sample_questions_mapping,
        data_folder,
        extension,
        num_iter,
        questions_per_iter)

    with open(f"{output_folder}/{output_file}", 'a', encoding="utf-8") as f:
        for cleaned_question, q_source in zip(cleaned_questions, q_sources):
            f.write(f"{cleaned_question};{q_source}\n")


if __name__ == '__main__':
    typer.run(main)