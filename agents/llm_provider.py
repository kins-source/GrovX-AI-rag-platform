import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

DEFAULT_NVIDIA_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"
RETIRED_NVIDIA_MODELS = {
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-405b-instruct",
    "qwen/qwen2.5-72b-instruct",
}
DEFAULT_NVIDIA_MODELS = (
    DEFAULT_NVIDIA_MODEL,
    "nvidia/llama-3.1-nemotron-51b-instruct",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/nemotron-3-super-120b-a12b",
    "deepseek-ai/deepseek-v4.1-flash",
)


def get_nvidia_models():
    configured_models = os.getenv("NVIDIA_MODELS", "")
    models = configured_models.split(",") if configured_models else DEFAULT_NVIDIA_MODELS
    return tuple(model.strip() for model in models if model.strip() and model.strip() not in RETIRED_NVIDIA_MODELS)


def get_llm(provider=None, api_key=None, model=None):
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

    def require_key(value, environment_name):
        key = value or os.getenv(environment_name)
        if not key:
            raise ValueError(f"Missing API key for {provider}. Add it in Settings or set {environment_name}.")
        return key

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=require_key(api_key, "OPENAI_API_KEY"),
            temperature=temperature,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            google_api_key=require_key(api_key, "GEMINI_API_KEY"),
            temperature=temperature,
        )

    if provider == "grok":
        from langchain_xai import ChatXAI

        return ChatXAI(
            model=os.getenv("GROK_MODEL", "grok-3-mini"),
            xai_api_key=require_key(api_key, "XAI_API_KEY"),
            temperature=temperature,
        )

    if provider == "nvidia":
        from langchain_openai import ChatOpenAI

        nvidia_key = api_key or os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
        if not nvidia_key:
            raise ValueError("Missing API key for nvidia. Add it in Settings or set NVIDIA_NIM_API_KEY.")
        nvidia_model = model or os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)
        if nvidia_model in RETIRED_NVIDIA_MODELS:
            nvidia_model = DEFAULT_NVIDIA_MODEL
        return ChatOpenAI(
            model=nvidia_model,
            api_key=nvidia_key,
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            temperature=temperature,
        )

    if provider != "ollama":
        raise ValueError("LLM_PROVIDER must be ollama, openai, gemini, grok, or nvidia")

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
    )