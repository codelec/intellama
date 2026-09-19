"""Request schemas (loose on purpose - we only require what we actually use)."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    stream: bool = True
    options: Dict[str, Any] = {}
    system: Optional[str] = None
    think: Optional[Union[bool, str]] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage] = []
    stream: bool = True
    options: Dict[str, Any] = {}
    think: Optional[Union[bool, str]] = None
