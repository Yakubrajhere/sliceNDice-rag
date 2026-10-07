"""Recursive character text splitter (no external dependency)."""
from typing import Dict, List

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_text(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Split text into pieces no longer than chunk_size, preserving the separators."""
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        # Final fallback: split by characters if no separators are left
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Find the highest priority separator present in the text
    chosen_sep = ""
    next_separators: List[str] = []
    for i, s in enumerate(separators):
        if s == "":
            chosen_sep = s
            next_separators = separators[i + 1:]
            break
        if s in text:
            chosen_sep = s
            next_separators = separators[i + 1:]
            break

    # Split text while retaining the separator attached to its preceding piece
    pieces: List[str] = []
    if chosen_sep == "":
        pieces = list(text)
    else:
        splits = text.split(chosen_sep)
        for i, split in enumerate(splits):
            if i < len(splits) - 1:
                pieces.append(split + chosen_sep)
            else:
                if split:  # Handle dangling trailing strings
                    pieces.append(split)

    # Recursively break down pieces that exceed the max chunk size
    out: List[str] = []
    for piece in pieces:
        if len(piece) > chunk_size:
            out.extend(_split_text(piece, chunk_size, next_separators))
        else:
            out.append(piece)
    return out


def _merge(pieces: List[str], chunk_size: int, overlap: int) -> List[str]:
    """Greedily pack text pieces together while preserving accurate rolling context overlap."""
    chunks: List[str] = []
    current_pieces: List[str] = []
    current_len = 0

    for piece in pieces:
        if not piece.strip():
            continue
        
        # If adding this piece exceeds limits, finalize the current chunk
        if current_len + len(piece) > chunk_size and current_pieces:
            chunks.append("".join(current_pieces))
            
            # Backtrack pieces to build a clean sliding window overlap
            overlap_pieces: List[str] = []
            overlap_len = 0
            for p in reversed(current_pieces):
                if overlap_len + len(p) <= overlap:
                    overlap_pieces.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            
            current_pieces = overlap_pieces
            current_len = overlap_len

        current_pieces.append(piece)
        current_len += len(piece)

    if current_pieces:
        chunks.append("".join(current_pieces))
    return chunks


def split_documents(documents: List[Dict], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict]:
    """Split loaded documents into chunks, keeping source/page metadata + chunk id."""
    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks: List[Dict] = []
    for doc in documents:
        text = doc.get("text", "")
        if not text.strip():
            continue
            
        pieces = _split_text(text, chunk_size, _SEPARATORS)
        merged_texts = _merge(pieces, chunk_size, chunk_overlap)
        
        for idx, chunk_text in enumerate(merged_texts):
            meta = dict(doc.get("metadata", {}))
            meta["chunk"] = idx
            chunks.append({"text": chunk_text, "metadata": meta})
    return chunks
