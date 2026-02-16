"""
Response formatting utilities
"""


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
            return response.content
        elif isinstance(response.content, list):
            # Handle structured content (list of content blocks)
            text_parts = []
            for block in response.content:
                if isinstance(block, dict) and 'text' in block:
                    text_parts.append(block['text'])
                elif isinstance(block, str):
                    text_parts.append(block)
            if text_parts:
                return '\n'.join(text_parts)
            # Fallback: some providers return non-text blocks (tool calls, metadata, etc.)
            # Return a string representation so the UI doesn't show a blank message.
            try:
                return str(response.content)
            except Exception:
                return ""
    
    # Fallback: convert to string
    return str(response)