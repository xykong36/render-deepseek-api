"""
Text splitting utilities for paragraph processing.

Provides functions to split paragraphs into individual sentences.
"""

import re
from typing import List


def split_into_sentences(text: str, split_by: str = "period") -> List[str]:
    """
    Split a paragraph into individual sentences.

    Args:
        text: Input paragraph text
        split_by: Splitting method:
            - "period": Split by periods, question marks, exclamation marks
            - "newline": Split by newline characters
            - "auto": Smart splitting (tries period first, falls back to newline)

    Returns:
        List of sentence strings (stripped and non-empty)
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    if split_by == "newline":
        sentences = text.split('\n')
    elif split_by == "period" or split_by == "auto":
        # Split by sentence-ending punctuation
        # Handles: . ! ? followed by space or end of string
        # Preserves abbreviations like "Mr." "Dr." "U.S."
        sentences = re.split(r'(?<=[.!?])\s+', text)
    else:
        # Default: treat as period
        sentences = re.split(r'(?<=[.!?])\s+', text)

    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    # If auto mode and only got 1 sentence, try newline split
    if split_by == "auto" and len(sentences) <= 1 and '\n' in text:
        sentences = text.split('\n')
        sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for text.

    Simple approximation: ~3 characters per token (average for English).

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return len(text) // 3
