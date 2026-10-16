"""NAT LLM configs -> PraisonAI ``Agent(llm=...)`` specs (wrapper_type is the string "praisonai")."""
import os

from nat.plugin_api import Builder
from nat.plugin_api import get_secret_value
from nat.plugin_api import register_llm_client
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig

from . import FRAMEWORK

NIM_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _common(cfg) -> dict:
    out = {}
    for k in ("temperature", "top_p", "max_tokens", "seed"):
        v = getattr(cfg, k, None)
        if v is not None:
            out[k] = v
    return out


@register_llm_client(config_type=NIMModelConfig, wrapper_type=FRAMEWORK)
async def nim_praisonai(cfg: NIMModelConfig, _builder: Builder):
    yield {
        "model": f"nvidia_nim/{cfg.model_name}",
        "base_url": cfg.base_url or NIM_DEFAULT_BASE_URL,
        "api_key": get_secret_value(cfg.api_key) or os.getenv("NVIDIA_API_KEY"),
        **_common(cfg),
    }


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=FRAMEWORK)
async def openai_praisonai(cfg: OpenAIModelConfig, _builder: Builder):
    spec = {"model": f"openai/{cfg.model_name}", **_common(cfg)}
    if (key := get_secret_value(cfg.api_key) or os.getenv("OPENAI_API_KEY")):
        spec["api_key"] = key
    if (base := cfg.base_url or os.getenv("OPENAI_BASE_URL")):
        spec["base_url"] = base
    yield spec


@register_llm_client(config_type=LiteLlmModelConfig, wrapper_type=FRAMEWORK)
async def litellm_praisonai(cfg: LiteLlmModelConfig, _builder: Builder):
    spec = {"model": cfg.model_name, **_common(cfg)}
    if (key := get_secret_value(cfg.api_key)):
        spec["api_key"] = key
    if cfg.base_url:
        spec["base_url"] = cfg.base_url
    yield spec
