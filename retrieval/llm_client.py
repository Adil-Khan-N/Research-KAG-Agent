"""
Unified Gemini LLM Client.
Provides a single interface that works with both:
1. Modern google-genai SDK (genai.Client)
2. Legacy google-generativeai SDK (genai.GenerativeModel)
Provides lazy initialization so importing modules never crashes when GEMINI_API_KEY is not yet configured.
"""

import os
import logging
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

class _ModelAdapter:
    """Adapts different SDK response objects to have a common .text property."""
    def __init__(self, backend: str, instance, model_name: str):
        self.backend = backend
        self.instance = instance
        self.model_name = model_name

    def generate_content(self, prompt: str):
        if self.backend == "genai":
            response = self.instance.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response
        else:
            return self.instance.generate_content(prompt)

_cached_models = {}

def get_gemini_model(model_name: Optional[str] = None):
    """
    Get a Gemini model instance.
    Uses GEMINI_MODEL env var if model_name is not specified (default: gemini-2.5-flash).
    Lazily loads client so import-time failures do not occur.
    """
    selected_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if selected_model in _cached_models:
        return _cached_models[selected_model]

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add GEMINI_API_KEY=<your-key> to your .env file."
        )

    # Try modern google.genai first
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        adapter = _ModelAdapter("genai", client, selected_model)
        _cached_models[selected_model] = adapter
        logger.info(f"Initialized Gemini model '{selected_model}' via google-genai SDK")
        return adapter
    except ImportError:
        pass

    # Try google.generativeai
    try:
        import google.generativeai as legacy_genai
        legacy_genai.configure(api_key=api_key)
        model = legacy_genai.GenerativeModel(selected_model)
        adapter = _ModelAdapter("google.generativeai", model, selected_model)
        _cached_models[selected_model] = adapter
        logger.info(f"Initialized Gemini model '{selected_model}' via google-generativeai SDK")
        return adapter
    except ImportError:
        raise ImportError(
            "Neither 'google-genai' nor 'google-generativeai' package is installed. "
            "Please install via: pip install google-genai"
        )
