"""
Response formatting utilities
"""

from __future__ import annotations

from ..config import Config


def _truncate_response_text(text: str) -> str:
    """Clamp pathological model responses before they blow up the UI."""
    max_chars = max(0, int(getattr(Config, "LLM_MAX_RESPONSE_CHARS", 24000)))
    if not max_chars or len(text) <= max_chars:
        return text
    marker = "\n\n[Response truncated in UI after reaching safety limit]\n"
    keep = max_chars - len(marker)
    if keep <= 0:
        return text[:max_chars]
    return text[:keep].rstrip() + marker


def extract_response_text(response) -> str:
    """
    Extract text content from various response formats.
    
    LLM responses can come in different formats:
    - Plain strings
    - Objects with .content attribute
    - Structured content with multiple blocks
    
    Args:
        response: Response from LLM
    
    Returns:
        Extracted text as string
    """
    if hasattr(response, 'content'):
        if isinstance(response.content, str):
            return _truncate_response_text(response.content)
        elif isinstance(response.content, list):
            # Handle structured content (list of content blocks)
            text_parts = []
            for block in response.content:
                if isinstance(block, dict) and 'text' in block:
                    text_parts.append(block['text'])
                elif isinstance(block, str):
                    text_parts.append(block)
            if text_parts:
                return _truncate_response_text('\n'.join(text_parts))
            # Fallback: some providers return non-text blocks (tool calls, metadata, etc.)
            # Return a string representation so the UI doesn't show a blank message.
            try:
                return _truncate_response_text(str(response.content))
            except Exception:
                return ""
    
    # Fallback: convert to string
    return _truncate_response_text(str(response))
