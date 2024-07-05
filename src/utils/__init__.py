from dotenv import load_dotenv

from .settings import (
    Settings as AppSettings,
    load_settings
)
from .tracing import setup_tracing
from .modules import setup_modules

from .convert2exel import display2exel
from .evaldisplay import display_eval_df, display_deepeval_df


__all__ = [
    "AppSettings",
    "load_settings",
    "setup_tracing",
    "setup_modules",
    "initialize",
    "display2exel",
    "display_eval_df",
    "display_deepeval_df"
]


def initialize(dotenv_path: str):
    load_dotenv(dotenv_path)
    settings = load_settings()

    setup_tracing(settings)
    setup_modules(settings)