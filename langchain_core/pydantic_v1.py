# compatibility shim for "from langchain_core.pydantic_v1 import BaseModel, Field"
# Works with either Pydantic v1 (installed) or Pydantic v2's v1 compatibility layer.
try:
    # If Pydantic v2 is installed, use its v1 compatibility module
    from pydantic.v1 import BaseModel, Field  # type: ignore
except Exception:
    # Otherwise fall back to pydantic v1
    from pydantic import BaseModel, Field  # type: ignore

__all__ = ["BaseModel", "Field"]
