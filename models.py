"""
Pydantic models for request and response validation.

Defines all data structures used in the Translation API endpoints.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


# ===== Shared Models =====

class HighlightEntry(BaseModel):
    """Model for a single highlight entry."""
    slug: str = Field(..., description="Kebab-case slug identifier")
    display_text: str = Field(..., description="Original word/phrase")
    translation_zh: str = Field(..., description="Chinese translation")


# ===== Use Case 1: Paragraph Translation + Highlights =====

class ParagraphTranslateRequest(BaseModel):
    """Request model for paragraph translation with highlights."""
    text: str = Field(..., description="English paragraph text to translate")
    extract_highlights: bool = Field(True, description="Whether to extract highlight entries")


class ParagraphTranslateResponse(BaseModel):
    """Response model for paragraph translation."""
    original_text: str = Field(..., description="Original English text")
    translation: str = Field(..., description="Chinese translation")
    highlights: list[HighlightEntry] = Field(default_factory=list, description="Extracted highlight entries")


# ===== Use Case 2: Paragraph to Sentences Generation =====

class ParagraphGenerateSentencesRequest(BaseModel):
    """Request model for generating sentences from paragraph."""
    text: str = Field(..., description="English paragraph text")
    episode_id: Optional[int] = Field(None, description="Episode ID for sentences")
    split_by: str = Field("period", description="Sentence splitting method: period | newline | auto")


class EnhancedSentence(BaseModel):
    """Model for an enhanced sentence with all metadata."""
    sentence_id: Optional[str] = None
    episode_id: Optional[int] = None
    episode_sequence: Optional[int] = None
    en: str
    zh: str
    phonetic_us: str
    highlight_entries: list[HighlightEntry] = Field(default_factory=list)
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    duration: Optional[float] = None
    sentence_hash: Optional[str] = None


class ParagraphGenerateSentencesResponse(BaseModel):
    """Response model for paragraph sentence generation."""
    sentences: list[EnhancedSentence]
    total_count: int = Field(..., description="Total number of sentences generated")


# ===== Use Case 3: Single Sentence Enhancement =====

class SentenceEnhanceRequest(BaseModel):
    """Request model for single sentence enhancement."""
    en: str = Field(..., description="English sentence text")
    sentence_id: Optional[str] = Field(None, description="Sentence ID")
    episode_id: Optional[int] = Field(None, description="Episode ID")
    episode_sequence: Optional[int] = Field(None, description="Episode sequence number")
    start_ts: Optional[float] = Field(None, description="Start timestamp")
    end_ts: Optional[float] = Field(None, description="End timestamp")
    duration: Optional[float] = Field(None, description="Duration")
    sentence_hash: Optional[str] = Field(None, description="Sentence hash")


class SentenceEnhanceResponse(BaseModel):
    """Response model for single sentence enhancement."""
    sentence: EnhancedSentence


# ===== Use Case 4: Expression Generation =====

class ExpressionGenerateRequest(BaseModel):
    """Request model for expression generation from sentences."""
    sentences: list[dict[str, Any]] = Field(..., description="List of sentence dictionaries with 'en', 'zh', etc.")
    episode_id: int = Field(1, description="Episode ID for the expressions")
    max_input_tokens: int = Field(2500, description="Maximum input tokens per batch")
    max_workers: int = Field(4, description="Maximum parallel workers")


class ExpressionTranslations(BaseModel):
    """Translations for an expression meaning."""
    zh: str
    en: str


class ExpressionExample(BaseModel):
    """Example usage of an expression."""
    zh: str
    en: str


class ExpressionMeaning(BaseModel):
    """Meaning definition for an expression."""
    translations: ExpressionTranslations
    examples: list[ExpressionExample] = Field(default_factory=list)


class PhraseTranslation(BaseModel):
    """A phrase with translation."""
    phrase: str
    translation: str


class WordRelations(BaseModel):
    """Word relations for an expression."""
    synonyms: list[PhraseTranslation] = Field(default_factory=list)
    antonyms: list[PhraseTranslation] = Field(default_factory=list)
    similar: list[PhraseTranslation] = Field(default_factory=list)


class RelatedExpression(BaseModel):
    """Related expression model."""
    type: str
    phrase: str
    translation: str
    examples: list[ExpressionExample] = Field(default_factory=list)


class Analysis(BaseModel):
    """Analysis item for an expression."""
    type: str
    content: str


class MatchedSentence(BaseModel):
    """A sentence that matches an expression."""
    sentence_id: Optional[str] = None
    episode_id: Optional[int] = None
    episode_sequence: Optional[int] = None
    en: str
    zh: str
    phonetic_us: str
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    duration: Optional[float] = None
    sentence_hash: Optional[str] = None
    highlight_entries: list[HighlightEntry] = Field(default_factory=list)
    matched_highlight: Optional[str] = None


class Expression(BaseModel):
    """Complete expression data model."""
    expression_id: str
    phrase: str
    phonetic: str
    type: str
    meanings: list[ExpressionMeaning]
    wordRelations: WordRelations
    relatedExpressions: list[RelatedExpression] = Field(default_factory=list)
    analysis: list[Analysis] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    episodes: list[str]
    matched_sentences: list[MatchedSentence]
    slug: str
    forms: list[str]


class ExpressionGenerateResponse(BaseModel):
    """Response model for expression generation."""
    expressions: list[Expression]
    total_count: int = Field(..., description="Total number of expressions generated")


# ===== Error Response Model =====

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
