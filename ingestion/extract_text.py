"""
Extract text from PDFs using PyMuPDF.
Tags sections using heading heuristics.
Cleans noise: headers, footers, equations, references.
"""

import fitz  # PyMuPDF
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Section heading patterns for scientific papers
SECTION_PATTERNS = [
    r"^(abstract)$",
    r"^(\d+\.?\s+introduction)$",
    r"^(\d+\.?\s+related work)$",
    r"^(\d+\.?\s+background)$",
    r"^(\d+\.?\s+method(ology|s)?)$",
    r"^(\d+\.?\s+approach)$",
    r"^(\d+\.?\s+model)$",
    r"^(\d+\.?\s+architecture)$",
    r"^(\d+\.?\s+experiment(s|al( setup)?)?)$",
    r"^(\d+\.?\s+result(s)?)$",
    r"^(\d+\.?\s+evaluation)$",
    r"^(\d+\.?\s+discussion)$",
    r"^(\d+\.?\s+conclusion(s)?)$",
    r"^(\d+\.?\s+limitation(s)?)$",
    r"^(references)$",
    r"^(appendix)$",
]

COMPILED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in SECTION_PATTERNS
]

# Noise patterns to filter out
NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),                    # standalone page numbers
    re.compile(r"^(arXiv|preprint|under review)", re.IGNORECASE),
    re.compile(r"^\s*figure\s+\d+", re.IGNORECASE), # figure captions
    re.compile(r"^\s*table\s+\d+", re.IGNORECASE),  # table captions
    re.compile(r"^https?://\S+$"),                   # bare URLs
]

def is_section_heading(text: str) -> str | None:
    """Return section name if line looks like a section heading."""
    text = text.strip()
    if len(text) > 80:
        return None
    if len(text) < 3:
        return None

    text_lower = text.lower()

    # Direct keyword match — most reliable for academic PDFs
    direct_matches = [
        "abstract", "introduction", "related work", "background",
        "method", "methods", "methodology", "approach", "model",
        "architecture", "experiments", "experimental setup",
        "evaluation", "results", "discussion", "conclusion",
        "conclusions", "limitations", "references", "appendix",
    ]
    for keyword in direct_matches:
        if text_lower == keyword:
            return keyword
        # "1. Introduction" or "1 Introduction" patterns
        import re
        if re.match(
            rf'^\d+\.?\s*{keyword}s?$', text_lower
        ):
            return keyword

    # Numbered section: "2. Related Work", "3.1 Architecture"
    import re
    numbered = re.match(
        r'^(\d+\.?\d*\.?\s+)([a-z].{3,40})$', text_lower
    )
    if numbered:
        return numbered.group(2).strip()

    return None

def is_noise(text: str) -> bool:
    """Return True if this line should be discarded."""
    text = text.strip()
    if len(text) < 3:
        return True
    for pattern in NOISE_PATTERNS:
        if pattern.match(text):
            return True
    return False

def clean_text(text: str) -> str:
    """Clean the text by removing extra whitespace and unwanted characters."""
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Remove lines that are just symbols/numbers (table/equation artifacts)
    lines = text.split("\n")

    cleaned = []
    for line in lines:
        stripped = line.strip()

        alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
        if alpha_ratio < 0.3 and len(stripped) > 5:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract full text from a PDF with section tagging.
    
    Returns:
        {
            "full_text": str,
            "sections": {
                "abstract": str,
                "introduction": str,
                "methods": str,
                "results": str,
                ...
            },
            "page_count": int
        }
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    all_lines = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        # Sort blocks top-to-bottom, left-to-right
        blocks.sort(key=lambda b: (round(b[1] / 10), b[0]))
        
        for block in blocks:
            text = block[4].strip()
            if text and not is_noise(text):
                all_lines.append(text)
    page_count = len(doc)
    doc.close()

    raw_text = "\n".join(all_lines)
    cleaned_text = clean_text(raw_text)
    
    sections = {}
    current_section = "preamble"
    section_buffer = []

    for line in cleaned_text.split("\n"):
        heading = is_section_heading(line)
        if(heading):
            if section_buffer:
                sections[current_section] = "\n".join(section_buffer).strip()
                section_buffer = []
            current_section = heading

        else:
            if line.strip():  # Avoid adding empty lines
                section_buffer.append(line)

    if section_buffer:
        sections[current_section] = "\n".join(section_buffer).strip()

        # Stop at references — we don't want to embed bibliographies
    if "references" in sections:
        del sections["references"]
    if "appendix" in sections:
        del sections["appendix"]

    return {
        "full_text": cleaned_text,
        "sections": sections,
        "page_count": page_count,
    }

if __name__ == "__main__":
    # Quick test on one paper
    result = extract_text_from_pdf("data/raw/2010.11929.pdf")
    print(f"Pages: {result['page_count']}")
    print(f"Sections found: {list(result['sections'].keys())}")
    print(f"\nAbstract preview:\n{result['sections'].get('abstract', '')[:300]}")
    print(f"\nTotal text length: {len(result['full_text'])} characters")
