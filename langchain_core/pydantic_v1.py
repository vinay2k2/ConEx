# Local fallback shim for "langchain_core.pydantic_v1"
# Ensures BaseModel & Field imports work even if LangChain package changed.
from pydantic import BaseModel, Field

__all__ = ["BaseModel", "Field"]
