"""Provider-specific SDK wrappers.

This package contains wrappers for:
- OpenAI (openai.py)
- Azure OpenAI (azure_openai.py)
- Anthropic (anthropic.py)
- Vertex AI (vertexai.py)

Each provider module handles:
- SDK method patching
- Usage extraction from responses
- Streaming support
- Region inference from base URLs
"""

__all__ = ["openai", "azure_openai", "anthropic", "vertexai"]
