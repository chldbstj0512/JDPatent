import os


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def openai_chat_options(component: str, **metadata: object) -> dict:
    if not _env_enabled("OPENAI_STORE_COMPLETIONS", default=True):
        return {}

    safe_metadata = {
        "service": "jdpatent",
        "component": str(component)[:512],
    }
    for key, value in metadata.items():
        if value is None:
            continue
        safe_key = str(key)[:64]
        safe_metadata[safe_key] = str(value)[:512]

    return {
        "store": True,
        "metadata": safe_metadata,
    }
