"""
Semantic chunking: split paper text into ~300 token chunks,
breaking on paragraph/sentence boundaries, with overlap.
"""

import tiktoken
import re
import logging

logger = logging.getLogger(__name__)

# Use cl100k_base tokenizer (same as GPT-4, close enough for counting)
TOKENIZER = tiktoken.get_encoding("cl100k_base")

TARGET_CHUNK_TOKENS = 300
OVERLAP_TOKENS = 50
MAX_CHUNK_TOKENS = 400  # hard ceiling


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences — respects abbreviations reasonably well."""
    # Split on period/exclamation/question followed by space + capital
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on double newlines."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(
    text: str,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[str]:
    """
    Split text into overlapping chunks of ~target_tokens.
    
    Strategy:
    1. Split into paragraphs first
    2. If a paragraph fits in the chunk budget, add it
    3. If a paragraph is too large, split it into sentences
    4. Add overlap from the previous chunk's tail
    
    Returns: list of chunk strings
    """
    paragraphs = split_into_paragraphs(text)
    
    chunks = []
    current_chunk_sentences = []
    current_token_count = 0
    
    for para in paragraphs:
        para_tokens = count_tokens(para)
        
        if para_tokens > max_tokens:
            # Paragraph too big — split into sentences
            sentences = split_into_sentences(para)
        else:
            sentences = [para]
        
        for sentence in sentences:
            sent_tokens = count_tokens(sentence)
            
            if sent_tokens > max_tokens:
                # Single sentence exceeds max — truncate (rare edge case)
                sentence = sentence[:1500]  # rough char limit
                sent_tokens = count_tokens(sentence)
            
            if current_token_count + sent_tokens > target_tokens and current_chunk_sentences:
                # Finalize this chunk
                chunk_text_str = " ".join(current_chunk_sentences)
                chunks.append(chunk_text_str)
                
                # Build overlap: take sentences from the tail of current chunk
                # until we've accumulated overlap_tokens worth
                overlap_sentences = []
                overlap_count = 0
                for prev_sent in reversed(current_chunk_sentences):
                    prev_tokens = count_tokens(prev_sent)
                    if overlap_count + prev_tokens > overlap_tokens:
                        break
                    overlap_sentences.insert(0, prev_sent)
                    overlap_count += prev_tokens
                
                # Start new chunk with overlap
                current_chunk_sentences = overlap_sentences + [sentence]
                current_token_count = overlap_count + sent_tokens
            else:
                current_chunk_sentences.append(sentence)
                current_token_count += sent_tokens
    
    # Don't forget the last chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    
    return chunks


def chunk_paper(paper_data: dict) -> list[dict]:
    """
    Chunk a full paper into structured chunk objects.
    
    Input: paper_data with keys: arxiv_id, title, sections, full_text
    Output: list of chunk dicts ready for JSON storage
    """
    arxiv_id = paper_data["arxiv_id"]
    sections = paper_data.get("sections", {})
    
    # Priority: chunk section by section so we know which section each chunk came from
    # If no sections detected, fall back to full_text
    
    all_chunks = []
    chunk_index = 0
    
    if sections:
        # Define section priority order for chunking
        section_order = [
            "abstract",
            "introduction",
            "related work",
            "background",
            "method", "methods", "methodology", "approach", "model", "architecture",
            "experiments", "experimental setup", "evaluation",
            "results",
            "discussion",
            "conclusion", "conclusions",
            "limitations",
            "preamble",
        ]
        
        # Process sections in order
        processed = set()
        for section_name in section_order:
            for key in sections:
                if section_name in key.lower() and key not in processed:
                    text = sections[key]
                    if not text or len(text) < 50:
                        continue
                    
                    raw_chunks = chunk_text(text)
                    for chunk_str in raw_chunks:
                        if count_tokens(chunk_str) < 20:
                            continue  # skip tiny fragments
                        
                        all_chunks.append({
                            "chunk_id": f"{arxiv_id}_chunk_{chunk_index:04d}",
                            "arxiv_id": arxiv_id,
                            "chunk_index": chunk_index,
                            "section": key,
                            "text": chunk_str,
                            "token_count": count_tokens(chunk_str),
                        })
                        chunk_index += 1
                    processed.add(key)
        
        # Any sections not in our order list
        for key, text in sections.items():
            if key not in processed and text and len(text) > 50:
                raw_chunks = chunk_text(text)
                for chunk_str in raw_chunks:
                    if count_tokens(chunk_str) < 20:
                        continue
                    all_chunks.append({
                        "chunk_id": f"{arxiv_id}_chunk_{chunk_index:04d}",
                        "arxiv_id": arxiv_id,
                        "chunk_index": chunk_index,
                        "section": key,
                        "text": chunk_str,
                        "token_count": count_tokens(chunk_str),
                    })
                    chunk_index += 1
    
    else:
        # Fallback: chunk the full text
        logger.warning(f"{arxiv_id}: No sections detected, chunking full text")
        raw_chunks = chunk_text(paper_data.get("full_text", ""))
        for chunk_str in raw_chunks:
            if count_tokens(chunk_str) < 20:
                continue
            all_chunks.append({
                "chunk_id": f"{arxiv_id}_chunk_{chunk_index:04d}",
                "arxiv_id": arxiv_id,
                "chunk_index": chunk_index,
                "section": "unknown",
                "text": chunk_str,
                "token_count": count_tokens(chunk_str),
            })
            chunk_index += 1
    
    return all_chunks


if __name__ == "__main__":
    # Quick test
    sample_text = """
    Vision Transformers have shown remarkable performance on image classification tasks.
    The key innovation is treating image patches as tokens, similar to words in NLP.
    
    This approach allows the model to capture long-range dependencies across the entire image.
    Unlike convolutional networks, there is no inductive bias toward local features.
    
    The architecture consists of a patch embedding layer followed by standard transformer blocks.
    Each block contains multi-head self-attention and a feed-forward network.
    """
    
    chunks = chunk_text(sample_text, target_tokens=50, overlap_tokens=10)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} ({count_tokens(chunk)} tokens): {chunk[:80]}...")