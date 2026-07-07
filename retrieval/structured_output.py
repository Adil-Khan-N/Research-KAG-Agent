"""
Structured Output Mode.

Accepts an optional JSON schema in the /query request body.
When present:
1. Instructs LLM to format answer matching the schema
2. Validates response against the schema
3. Returns clean JSON instead of prose

Use cases:
- Comparison queries → structured table object
- Paper listing → [{title, year, method}]
- Metric extraction → {paper: str, accuracy: float, dataset: str}

CV talking point:
"Implemented structured output mode with JSON schema validation,
enabling programmatic consumption of research answers"
"""

import os
import re
import json
import logging
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-2.5-flash")


STRUCTURED_PROMPT = """You are a research assistant that answers questions
about scientific papers.

Answer the question using ONLY the provided evidence.
Format your answer as a JSON object matching this schema:
{schema}

EVIDENCE:
{evidence_text}

QUESTION: {query}

Return ONLY the JSON object — no explanation, no markdown,
no code fences. The JSON must be valid and match the schema exactly.
Use null for fields where evidence is insufficient."""


def validate_against_schema(
    data: dict,
    schema: dict,
) -> tuple[bool, list[str]]:
    """
    Validate a dict against a simple JSON schema.
    Supports: type, properties, required, items.

    Returns (is_valid, list_of_errors)
    """
    errors = []

    schema_type = schema.get("type", "object")

    if schema_type == "object":
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in data:
                errors.append(f"Missing required field: '{field}'")

        for field, field_schema in properties.items():
            if field in data and data[field] is not None:
                field_type = field_schema.get("type")
                if field_type == "string" and not isinstance(data[field], str):
                    errors.append(
                        f"Field '{field}' should be string, "
                        f"got {type(data[field]).__name__}"
                    )
                elif field_type == "number" and not isinstance(
                    data[field], (int, float)
                ):
                    errors.append(
                        f"Field '{field}' should be number, "
                        f"got {type(data[field]).__name__}"
                    )
                elif field_type == "array" and not isinstance(
                    data[field], list
                ):
                    errors.append(
                        f"Field '{field}' should be array, "
                        f"got {type(data[field]).__name__}"
                    )

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"Expected array, got {type(data).__name__}")

    return len(errors) == 0, errors


def generate_structured_answer(
    query: str,
    chunks: list,
    output_schema: dict,
) -> dict:
    """
    Generate a structured JSON answer matching the provided schema.

    Args:
        query: user question
        chunks: retrieved evidence chunks
        output_schema: JSON schema dict defining the output format

    Returns:
        {
            "data": <structured answer matching schema>,
            "schema_valid": bool,
            "validation_errors": list,
            "raw": str,
        }
    """
    # Format evidence
    evidence_lines = []
    for i, chunk in enumerate(chunks[:6], 1):

        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            title = chunk.get("title", "")
            year = chunk.get("year", "")
        else:
            text = getattr(chunk, "text", "")
            title = getattr(chunk, "title", "")
            year = getattr(chunk, "year", "")

        evidence_lines.append(
            f"[{i}] {title} ({year}):\n{text[:400]}"
        )
    evidence_text = "\n\n".join(evidence_lines)

    schema_str = json.dumps(output_schema, indent=2)

    prompt = STRUCTURED_PROMPT.format(
        schema=schema_str,
        evidence_text=evidence_text[:3000],
        query=query,
    )

    try:
        response = _model.generate_content(prompt)
        raw = response.text.strip()

        # Strip markdown
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*',     '', raw)
        raw = re.sub(r'\s*```$',     '', raw)
        raw = re.sub(r',\s*([}\]])', r'\1', raw)
        raw = raw.strip()

        data = json.loads(raw)

        # Validate
        is_valid, errors = validate_against_schema(data, output_schema)

        logger.info(
            f"Structured output: valid={is_valid}, "
            f"errors={len(errors)}"
        )

        return {
            "data":              data,
            "schema_valid":      is_valid,
            "validation_errors": errors,
            "raw":               raw,
        }

    except json.JSONDecodeError as e:
        logger.error(f"Structured output JSON parse failed: {e}")
        return {
            "data":              None,
            "schema_valid":      False,
            "validation_errors": [f"JSON parse error: {e}"],
            "raw":               raw if "raw" in dir() else "",
        }
    except Exception as e:
        logger.error(f"Structured output failed: {e}")
        return {
            "data":              None,
            "schema_valid":      False,
            "validation_errors": [str(e)],
            "raw":               "",
        }


# ── Preset schemas for common query types ─────────────────────

COMPARISON_SCHEMA = {
    "type": "object",
    "required": ["comparison_topic", "items"],
    "properties": {
        "comparison_topic": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":         {"type": "string"},
                    "year":         {"type": "number"},
                    "key_feature":  {"type": "string"},
                    "advantage":    {"type": "string"},
                    "limitation":   {"type": "string"},
                    "datasets":     {"type": "array"},
                },
            },
        },
        "summary": {"type": "string"},
    },
}

PAPER_LIST_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["title", "year"],
        "properties": {
            "title":        {"type": "string"},
            "year":         {"type": "number"},
            "arxiv_id":     {"type": "string"},
            "key_method":   {"type": "string"},
            "dataset":      {"type": "string"},
            "contribution": {"type": "string"},
        },
    },
}

METRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "paper":    {"type": "string"},
        "year":     {"type": "number"},
        "metric":   {"type": "string"},
        "value":    {"type": "number"},
        "dataset":  {"type": "string"},
        "notes":    {"type": "string"},
    },
}

TIMELINE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["year", "title"],
        "properties": {
            "year":         {"type": "number"},
            "title":        {"type": "string"},
            "contribution": {"type": "string"},
            "extends":      {"type": "string"},
            "method":       {"type": "string"},
        },
    },
}

PRESET_SCHEMAS = {
    "comparison": COMPARISON_SCHEMA,
    "paper_list": PAPER_LIST_SCHEMA,
    "metric":     METRIC_SCHEMA,
    "timeline":   TIMELINE_SCHEMA,
}