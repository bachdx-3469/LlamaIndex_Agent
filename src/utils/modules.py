from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding 
from llama_index.llms.azure_openai import AzureOpenAI

from .settings import Settings as AppSettings


def setup_modules(settings: AppSettings):
    api_key = settings.azure_openai.key
    end_point = settings.azure_openai.endpoint
    version = settings.azure_openai.version
    gpt = settings.azure_openai.gpt_deployment_name
    embed = settings.azure_openai.embed_deployment_name

    llm = AzureOpenAI(
        deployment_name=gpt,
        azure_endpoint=end_point,
        api_key=api_key,
        api_version=version)
    embed_model = AzureOpenAIEmbedding(
        deployment_name=embed,
        azure_endpoint=end_point,
        api_version=version,
        api_key=api_key)

    Settings.llm = llm
    Settings.embed_model = embed_model
