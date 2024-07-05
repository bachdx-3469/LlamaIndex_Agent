from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class TracingConfig(BaseModel):
    trace: bool = True
    public_key: str
    secret_key: str
    user_id: str
    host: str = "https://langfuse.poc.sun-asterisk.ai"


class AzureOpenAIConfig(BaseModel):
    endpoint: str
    key: str
    gpt_deployment_name: str
    embed_deployment_name: str
    version: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter='__')
    azure_openai: AzureOpenAIConfig
    tracing: TracingConfig


def load_settings() -> Settings:
    return Settings()
