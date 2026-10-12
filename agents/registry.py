"""
Model registry — maps friendly model names to agent classes + defaults.
"""

from typing import Dict, List, Optional, Tuple, Type

from .base import BaseAgent


def _qwen_inferred_defaults(model: str) -> dict:
    """
    Infer sensible defaults for Qwen model IDs that are not explicitly registered.

    Hugging Face-style Qwen3.5 IDs are typically served behind an OpenAI-compatible
    endpoint rather than DashScope-native APIs, so default them to `api_backend=openai`.
    """
    model_lower = model.lower()
    if "qwen3.5" in model_lower or "qwen35" in model_lower:
        return {
            "api_backend": "openai",
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": 32768,
            "history_n": 100,
        }
    return {}

# ── Model registry ──────────────────────────────────────────────────────
# Each entry: model_alias -> (agent_class_path, default_kwargs)
# agent_class_path is lazy-imported to avoid pulling in all SDKs at once.

_REGISTRY: Dict[str, dict] = {'gpt-5.5': {'cls': 'agents.chatgpt_agent.ChatGPTAgent', 'defaults': {'model': 'gpt-5.5'}}, 'gpt-5.6-sol': {'cls': 'agents.chatgpt_agent.ChatGPTAgent', 'defaults': {'model': 'gpt-5.6-sol', 'environment': 'linux'}}, 'claude-sonnet-5': {'cls': 'agents.claude_agent.ClaudeAgent', 'defaults': {'model': 'claude-sonnet-5'}}, 'claude-opus-5-5': {'cls': 'agents.claude_agent.ClaudeAgent', 'defaults': {'model': 'claude-opus-5-5'}}, 'qwen3.8-27b': {'cls': 'agents.qwen_agent.QwenAgent', 'defaults': {'model': 'Qwen/Qwen3.8-27B', 'api_backend': 'openai', 'enable_thinking': True}}, 'evocua-32b': {'cls': 'agents.evocua_agent.EvoCUAAgent', 'defaults': {'model': 'EvoCUA-32B', 'prompt_style': 'S2'}}, 'kimi-k3': {'cls': 'agents.kimi_agent.KimiAgent', 'defaults': {'model': 'kimi-k3', 'coordinate_type': 'auto', 'structured_output': True}}}


def _import_class(dotted_path: str) -> Type[BaseAgent]:
    """Import a class from a dotted path like 'agents.claude_agent.ClaudeAgent'."""
    import importlib

    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_agent(model: str, **kwargs) -> BaseAgent:
    """
    Create an agent by model name.

    Args:
        model: A model alias (e.g. "claude-sonnet-4-5", "kimi-k2.5", "qwen3-vl",
               "evocua-s1", "mano", "opencua", "dart")
               or any full model ID supported by an agent class.
        **kwargs: Override any default parameter.

    Returns:
        An initialized agent ready for predict() calls.

    Example:
        agent = create_agent("claude-sonnet-4-5", platform="ubuntu")
        agent.reset()
        reasoning, actions = agent.predict("Open Firefox", {"screenshot": img_bytes})
    """
    if model in _REGISTRY:
        entry = _REGISTRY[model]
        cls = _import_class(entry["cls"])
        merged = {**entry["defaults"], **kwargs}
        return cls(**merged)

    # If not in registry, try to infer the agent family from the name
    model_lower = model.lower()
    if "claude" in model_lower:
        cls = _import_class("agents.claude_agent.ClaudeAgent")
        return cls(model=model, **kwargs)
    elif "kimi" in model_lower:
        cls = _import_class("agents.kimi_agent.KimiAgent")
        return cls(model=model, **kwargs)
    elif "qwen" in model_lower:
        cls = _import_class("agents.qwen_agent.QwenAgent")
        inferred = _qwen_inferred_defaults(model)
        merged = {**inferred, **kwargs}
        return cls(model=model, **merged)
    elif "evocua" in model_lower:
        cls = _import_class("agents.evocua_agent.EvoCUAAgent")
        return cls(model=model, **kwargs)
    elif "mano" in model_lower:
        cls = _import_class("agents.mano_agent.ManoAgent")
        return cls(model=model, **kwargs)
    elif "opencua" in model_lower:
        cls = _import_class("agents.opencua_agent.OpenCUAAgent")
        return cls(model=model, **kwargs)
    elif "dart" in model_lower:
        cls = _import_class("agents.dart_agent.DartAgent")
        return cls(model=model, **kwargs)
    elif "owl" in model_lower:
        cls = _import_class("agents.owl15_agent.Owl15Agent")
        return cls(model=model, **kwargs)
    elif "mai-ui" in model_lower or model_lower.startswith("mai_ui") or "mai-ui" in model:
        cls = _import_class("agents.mai_ui_agent.MAIUIAgent")
        return cls(model=model, **kwargs)
    elif "ui-tars" in model_lower or "ui_tars" in model_lower or "uitars" in model_lower:
        cls = _import_class("agents.ui_tars_agent.UITarsAgent")
        return cls(model=model, **kwargs)
    elif "gemini" in model_lower:
        cls = _import_class("agents.gemini_agent.GeminiAgent")
        return cls(model=model, **kwargs)
    elif model_lower.startswith("azure-") or model_lower.startswith("azure/"):
        bare = model.split("-", 1)[1] if model_lower.startswith("azure-") else model.split("/", 1)[1]
        cls = _import_class("agents.chatgpt_agent.ChatGPTAgent")
        merged = {"api_backend": "azure", **kwargs}
        return cls(model=bare, **merged)
    elif model_lower.startswith("bltcy-") or model_lower.startswith("bltcy/"):
        bare = model.split("-", 1)[1] if model_lower.startswith("bltcy-") else model.split("/", 1)[1]
        if "claude" in bare.lower():
            cls = _import_class("agents.claude_agent.ClaudeAgent")
            merged = {"api_backend": "bltcy", **kwargs}
            return cls(model=bare, **merged)
        cls = _import_class("agents.chatgpt_agent.ChatGPTAgent")
        merged = {"api_backend": "bltcy", **kwargs}
        return cls(model=bare, **merged)
    elif (
        "chatgpt" in model_lower
        or model_lower.startswith("gpt-")
        or "computer-use-preview" in model_lower
    ):
        cls = _import_class("agents.chatgpt_agent.ChatGPTAgent")
        return cls(model=model, **kwargs)

    raise ValueError(
        f"Unknown model '{model}'. Available: {list(_REGISTRY.keys())}. "
        f"Or use a full model ID containing one of: claude, kimi, qwen, evocua, mano, opencua, dart, owl, mai-ui, ui-tars, gemini, chatgpt/gpt-."
    )


def list_models() -> List[str]:
    """Return all registered model aliases."""
    return sorted(_REGISTRY.keys())
