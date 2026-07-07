"""
Adaptive chunking strategy — section-aware chunk sizes.

Reasoning:
- Abstract: dense, information-rich → small chunks (150 tokens)
  High precision needed: each sentence is independently meaningful
- Introduction: narrative context → medium chunks (200 tokens)  
- Methods/Architecture: technical detail → standard chunks (300 tokens)
- Results/Experiments: needs surrounding context → larger chunks (400 tokens)
- Discussion/Conclusion: synthesis → medium chunks (250 tokens)
- Preamble (title, authors, headers): noise → tiny chunks (100 tokens)
  These rarely contribute to good answers

Compare with fixed-size (300 tokens everywhere):
- Abstract gets over-chunked: related sentences split across chunks
- Results get under-chunked: context for a finding is cut off
- Preamble wastes slots with author names and headers

CV claim: "Implemented adaptive chunking strategy with section-aware
sizing; evaluated impact via RAGAS context precision/recall metrics"
"""

import logging
import tiktoken

logger = logging.getLogger(__name__)

TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Section → target tokens mapping
SECTION_CHUNK_SIZES = {
    # Very small: abstract sentences are self-contained
    "abstract":           {"target": 150, "overlap": 20, "max": 200},

    # Small: preamble is usually title/author noise
    "preamble":           {"target": 100, "overlap": 10, "max": 150},

    # Medium: introduction provides narrative context
    "introduction":       {"target": 200, "overlap": 30, "max": 280},
    "1. introduction":    {"target": 200, "overlap": 30, "max": 280},
    "2. introduction":    {"target": 200, "overlap": 30, "max": 280},

    # Standard: related work is reference-dense
    "related work":       {"target": 250, "overlap": 40, "max": 320},
    "2. related work":    {"target": 250, "overlap": 40, "max": 320},
    "background":         {"target": 250, "overlap": 40, "max": 320},

    # Standard: methods need full context
    "method":             {"target": 300, "overlap": 50, "max": 400},
    "methods":            {"target": 300, "overlap": 50, "max": 400},
    "methodology":        {"target": 300, "overlap": 50, "max": 400},
    "approach":           {"target": 300, "overlap": 50, "max": 400},
    "model":              {"target": 300, "overlap": 50, "max": 400},
    "architecture":       {"target": 300, "overlap": 50, "max": 400},
    "3. method":          {"target": 300, "overlap": 50, "max": 400},
    "4. method":          {"target": 300, "overlap": 50, "max": 400},

    # Larger: results need surrounding numbers for context
    "results":            {"target": 400, "overlap": 60, "max": 500},
    "experiments":        {"target": 400, "overlap": 60, "max": 500},
    "experimental setup": {"target": 350, "overlap": 50, "max": 450},
    "evaluation":         {"target": 350, "overlap": 50, "max": 450},
    "ablation":           {"target": 350, "overlap": 50, "max": 450},

    # Medium: discussion synthesizes findings
    "discussion":         {"target": 250, "overlap": 40, "max": 330},
    "conclusion":         {"target": 200, "overlap": 30, "max": 280},
    "conclusions":        {"target": 200, "overlap": 30, "max": 280},
    "limitations":        {"target": 200, "overlap": 30, "max": 280},
}

# Default for unknown sections
DEFAULT_CHUNK_SIZE = {"target": 300, "overlap": 50, "max": 400}


def get_chunk_config(section: str) -> dict:
    """
    Get chunk size configuration for a section.
    Case-insensitive, partial match.
    """
    if not section:
        return DEFAULT_CHUNK_SIZE

    section_lower = section.lower().strip()

    # Exact match first
    if section_lower in SECTION_CHUNK_SIZES:
        return SECTION_CHUNK_SIZES[section_lower]

    # Partial match
    for key, config in SECTION_CHUNK_SIZES.items():
        if key in section_lower or section_lower in key:
            return config

    return DEFAULT_CHUNK_SIZE


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    import re
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def adaptive_chunk_text(
    text: str,
    section: str = "unknown",
    config: dict = None,
) -> list[str]:
    """
    Chunk text using section-aware sizing.

    Args:
        text: text to chunk
        section: section name (determines chunk size)
        config: override config dict

    Returns:
        list of chunk strings
    """
    if config is None:
        config = get_chunk_config(section)

    target = config["target"]
    overlap = config["overlap"]
    max_tokens = config["max"]

    paragraphs = split_into_paragraphs(text)
    chunks = []
    current_sentences = []
    current_count = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > max_tokens:
            sentences = split_into_sentences(para)
        else:
            sentences = [para]

        for sentence in sentences:
            sent_tokens = count_tokens(sentence)

            if sent_tokens > max_tokens:
                sentence = sentence[:1500]
                sent_tokens = count_tokens(sentence)

            if (current_count + sent_tokens > target
                    and current_sentences):
                # Finalize chunk
                chunks.append(" ".join(current_sentences))

                # Overlap: take tail sentences
                overlap_sents = []
                overlap_count = 0
                for prev in reversed(current_sentences):
                    pt = count_tokens(prev)
                    if overlap_count + pt > overlap:
                        break
                    overlap_sents.insert(0, prev)
                    overlap_count += pt

                current_sentences = overlap_sents + [sentence]
                current_count = overlap_count + sent_tokens
            else:
                current_sentences.append(sentence)
                current_count += sent_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def _infer_section_from_text(text: str) -> str:
    """
    Infer section type from chunk text content.
    Used when section metadata is missing.
    """
    text_lower = text.lower()[:200]

    if any(w in text_lower for w in [
        "we propose", "in this paper", "we present",
        "this paper", "we introduce"
    ]):
        return "abstract"

    if any(w in text_lower for w in [
        "in this section", "we describe", "our method",
        "the architecture", "consists of", "we use"
    ]):
        return "methods"

    if any(w in text_lower for w in [
        "table", "accuracy", "performance", "outperforms",
        "achieves", "top-1", "imagenet", "coco", "compared to"
    ]):
        return "results"

    if any(w in text_lower for w in [
        "in conclusion", "we have shown", "future work",
        "limitation", "we summarize"
    ]):
        return "conclusion"

    return "unknown"


def adaptive_chunk_paper(paper_data: dict) -> list[dict]:
    """
    Chunk paper with adaptive sizing.
    Falls back to content-based section inference if no sections found.
    """
    arxiv_id = paper_data["arxiv_id"]
    sections = paper_data.get("sections", {})
    existing_chunks = paper_data.get("chunks", [])

    all_chunks = []
    chunk_index = 0

    # ── Path A: sections exist ────────────────────────────────
    if sections:
        section_order = [
            "abstract", "preamble", "introduction",
            "related work", "background", "method", "methods",
            "methodology", "approach", "model", "architecture",
            "experiments", "experimental setup", "evaluation",
            "results", "ablation", "discussion",
            "conclusion", "conclusions", "limitations",
        ]

        processed = set()
        for section_name in section_order:
            for key in sections:
                if section_name in key.lower() and key not in processed:
                    text = sections[key]
                    if not text or len(text) < 30:
                        continue
                    config = get_chunk_config(key)
                    raw_chunks = adaptive_chunk_text(
                        text, section=key, config=config
                    )
                    for chunk_str in raw_chunks:
                        if count_tokens(chunk_str) < 15:
                            continue
                        all_chunks.append({
                            "chunk_id": (
                                f"{arxiv_id}_adaptive_chunk_{chunk_index:04d}"
                            ),
                            "arxiv_id":      arxiv_id,
                            "chunk_index":   chunk_index,
                            "section":       key,
                            "text":          chunk_str,
                            "token_count":   count_tokens(chunk_str),
                            "chunking_strategy": "adaptive",
                            "target_tokens": config["target"],
                        })
                        chunk_index += 1
                    processed.add(key)

        for key, text in sections.items():
            if key not in processed and text and len(text) > 30:
                config = get_chunk_config(key)
                raw_chunks = adaptive_chunk_text(
                    text, section=key, config=config
                )
                for chunk_str in raw_chunks:
                    if count_tokens(chunk_str) < 15:
                        continue
                    all_chunks.append({
                        "chunk_id": (
                            f"{arxiv_id}_adaptive_chunk_{chunk_index:04d}"
                        ),
                        "arxiv_id":      arxiv_id,
                        "chunk_index":   chunk_index,
                        "section":       key,
                        "text":          chunk_str,
                        "token_count":   count_tokens(chunk_str),
                        "chunking_strategy": "adaptive",
                        "target_tokens": config["target"],
                    })
                    chunk_index += 1

    # ── Path B: no sections — infer from existing chunks ──────
    elif existing_chunks:
        logger.info(
            f"{arxiv_id}: no sections found, "
            f"re-chunking {len(existing_chunks)} existing chunks "
            f"with content-based inference"
        )

        for orig_chunk in existing_chunks:
            text = orig_chunk.get("text", "")
            if not text or len(text) < 30:
                continue

            # Infer section from content
            inferred_section = _infer_section_from_text(text)
            config = get_chunk_config(inferred_section)

            # Re-chunk with adaptive size
            raw_chunks = adaptive_chunk_text(
                text,
                section=inferred_section,
                config=config,
            )

            for chunk_str in raw_chunks:
                if count_tokens(chunk_str) < 15:
                    continue
                all_chunks.append({
                    "chunk_id": (
                        f"{arxiv_id}_adaptive_chunk_{chunk_index:04d}"
                    ),
                    "arxiv_id":        arxiv_id,
                    "chunk_index":     chunk_index,
                    "section":         inferred_section,
                    "text":            chunk_str,
                    "token_count":     count_tokens(chunk_str),
                    "chunking_strategy": "adaptive_inferred",
                    "target_tokens":   config["target"],
                })
                chunk_index += 1

    # ── Path C: nothing works — use full_text ─────────────────
    else:
        full_text = paper_data.get("full_text", "")
        if full_text:
            raw_chunks = adaptive_chunk_text(
                full_text, section="unknown"
            )
            for chunk_str in raw_chunks:
                all_chunks.append({
                    "chunk_id": (
                        f"{arxiv_id}_adaptive_chunk_{chunk_index:04d}"
                    ),
                    "arxiv_id":        arxiv_id,
                    "chunk_index":     chunk_index,
                    "section":         "unknown",
                    "text":            chunk_str,
                    "token_count":     count_tokens(chunk_str),
                    "chunking_strategy": "adaptive_fulltext",
                    "target_tokens":   300,
                })
                chunk_index += 1

    logger.info(
        f"{arxiv_id}: {len(all_chunks)} adaptive chunks "
        f"(path: {'sections' if sections else 'inferred' if existing_chunks else 'fulltext'})"
    )
    return all_chunks

def compare_chunking_strategies(
    paper_data: dict,
) -> dict:
    """
    Compare fixed vs adaptive chunking for one paper.
    Returns stats for both strategies.
    """
    from ingestion.chunker import chunk_paper as fixed_chunk_paper

    fixed_chunks = fixed_chunk_paper(paper_data)
    adaptive_chunks = adaptive_chunk_paper(paper_data)

    def stats(chunks):
        if not chunks:
            return {}
        token_counts = [c["token_count"] for c in chunks]
        by_section = {}
        for c in chunks:
            sec = c.get("section", "unknown")
            if sec not in by_section:
                by_section[sec] = []
            by_section[sec].append(c["token_count"])
        return {
            "total_chunks": len(chunks),
            "avg_tokens": round(
                sum(token_counts) / len(token_counts), 1
            ),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "by_section": {
                k: {
                    "count": len(v),
                    "avg": round(sum(v) / len(v), 1),
                }
                for k, v in by_section.items()
            },
        }

    return {
        "arxiv_id": paper_data["arxiv_id"],
        "title": paper_data["title"],
        "fixed": stats(fixed_chunks),
        "adaptive": stats(adaptive_chunks),
    }