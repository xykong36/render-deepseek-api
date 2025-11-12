"""
Highlight Service - Handles highlight entry extraction logic.

Business logic layer for extracting important words/phrases as highlights.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from .deepseek_client import DeepseekClient, DeepseekAPIError
from .prompts import get_highlight_system_prompt, get_highlight_user_prompt


logger = logging.getLogger(__name__)


class HighlightService:
    """
    Service for extracting highlight entries from sentences.

    Responsibilities:
    - Validate input text
    - Call Deepseek API for highlight extraction
    - Parse JSON response
    - Validate highlight entries
    - Filter invalid highlights
    """

    def __init__(self, client: DeepseekClient):
        """
        Initialize highlight service.

        Args:
            client: DeepseekClient instance for API calls
        """
        self.client = client
        self.logger = logger

    def extract_highlights(
        self,
        text: str,
        chinese_translation: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Extract highlight entries from English text.

        Args:
            text: English text to analyze
            chinese_translation: Chinese translation of the text (optional)

        Returns:
            List of highlight dictionaries with keys:
            - slug: Kebab-case identifier
            - display_text: Original word/phrase
            - translation_zh: Chinese translation

        Raises:
            DeepseekAPIError: If highlight extraction fails
        """
        # Input validation
        if not text or not text.strip():
            self.logger.warning("Empty text provided for highlight extraction")
            return []

        text = text.strip()
        has_translation = bool(chinese_translation and chinese_translation.strip())

        try:
            self.logger.debug(f"Extracting highlights from: {text[:50]}...")

            # Get appropriate prompts
            system_prompt = get_highlight_system_prompt(has_translation)
            user_prompt = get_highlight_user_prompt(text, chinese_translation)

            # Call API
            raw_result = self.client.simple_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=500
            )

            # Parse JSON response
            highlights = self._parse_json_response(raw_result)

            # Validate highlights
            validated_highlights = self._validate_highlights(
                highlights,
                chinese_translation if has_translation else None
            )

            self.logger.info(f"✅ Extracted {len(validated_highlights)} highlights from: '{text[:30]}...'")
            return validated_highlights

        except DeepseekAPIError as e:
            self.logger.error(f"Highlight extraction failed for text '{text[:30]}...': {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during highlight extraction: {e}")
            raise DeepseekAPIError(f"Highlight extraction failed: {e}")

    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse JSON response from API.

        Args:
            response: Raw API response string

        Returns:
            List of highlight dictionaries

        Raises:
            ValueError: If JSON parsing fails
        """
        try:
            highlights = json.loads(response)

            if not isinstance(highlights, list):
                self.logger.warning(f"Invalid highlight format (not a list): {response[:100]}")
                return []

            return highlights

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse highlight JSON: {e}")
            self.logger.debug(f"Raw response: {response}")
            return []

    def _validate_highlights(
        self,
        highlights: List[Dict[str, Any]],
        chinese_translation: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Validate and filter highlight entries.

        Args:
            highlights: List of raw highlight dictionaries
            chinese_translation: Chinese translation for validation (optional)

        Returns:
            List of validated highlight dictionaries
        """
        validated = []

        for entry in highlights:
            # Check required fields
            if not isinstance(entry, dict):
                self.logger.warning(f"Skipping non-dict highlight entry: {entry}")
                continue

            required_fields = ['slug', 'display_text', 'translation_zh']
            if not all(k in entry for k in required_fields):
                self.logger.warning(f"Skipping highlight entry with missing fields: {entry}")
                continue

            # Additional validation: if we have Chinese translation, verify translation_zh is a substring
            if chinese_translation:
                translation_zh = entry.get('translation_zh', '').strip()
                if translation_zh and translation_zh not in chinese_translation:
                    self.logger.warning(
                        f"⚠️ translation_zh '{translation_zh}' not found in sentence '{chinese_translation}'. "
                        f"Skipping highlight entry: {entry}"
                    )
                    continue

            # Add validated entry
            validated.append({
                'slug': entry['slug'],
                'display_text': entry['display_text'],
                'translation_zh': entry['translation_zh']
            })

        return validated
