"""
Phonetic Service - Handles US phonetic transcription logic.

Business logic layer for phonetic transcription operations.
"""

import logging
import re
from typing import Optional
from .deepseek_client import DeepseekClient, DeepseekAPIError
from .prompts import PHONETIC_SYSTEM_PROMPT, get_phonetic_user_prompt


logger = logging.getLogger(__name__)


class PhoneticService:
    """
    Service for handling US phonetic transcription operations.

    Responsibilities:
    - Validate input text
    - Call Deepseek API for phonetic transcription
    - Parse and clean phonetic output
    - Ensure proper IPA format with slashes
    """

    def __init__(self, client: DeepseekClient):
        """
        Initialize phonetic service.

        Args:
            client: DeepseekClient instance for API calls
        """
        self.client = client
        self.logger = logger

    def get_phonetic(self, text: str) -> str:
        """
        Generate US phonetic transcription for English text.

        Args:
            text: English text to transcribe

        Returns:
            US phonetic transcription in IPA notation with slashes (e.g., /həˈloʊ/)

        Raises:
            DeepseekAPIError: If phonetic generation fails
        """
        # Input validation
        if not text or not text.strip():
            self.logger.warning("Empty text provided for phonetic transcription")
            return ""

        text = text.strip()

        try:
            self.logger.debug(f"Generating phonetic for: {text[:50]}...")

            # Call API
            raw_result = self.client.simple_completion(
                system_prompt=PHONETIC_SYSTEM_PROMPT,
                user_prompt=get_phonetic_user_prompt(text),
                temperature=0.1,
                max_tokens=500
            )

            # Parse and clean the response
            phonetic = self._parse_phonetic_response(raw_result)

            self.logger.info(f"✅ Phonetic generated: '{text[:30]}...' -> '{phonetic}'")
            return phonetic

        except DeepseekAPIError as e:
            self.logger.error(f"Phonetic generation failed for text '{text[:30]}...': {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during phonetic generation: {e}")
            raise DeepseekAPIError(f"Phonetic generation failed: {e}")

    def _parse_phonetic_response(self, response: str) -> str:
        """
        Parse and clean phonetic response from API.

        Ensures the output is properly formatted with slashes.

        Args:
            response: Raw API response

        Returns:
            Cleaned phonetic transcription with slashes (e.g., /həˈloʊ/)
        """
        if not response:
            return ""

        # Already properly formatted with slashes
        if response.startswith('/') and response.endswith('/'):
            return response

        # Extract phonetic transcription from response (look for text enclosed in slashes)
        phonetic_match = re.search(r'(/[^/]+/)', response)
        if phonetic_match:
            return phonetic_match.group(1)

        # Remove common prefixes/suffixes
        cleaned_response = response.strip()
        prefixes_to_remove = [
            "Phonetic transcription:",
            "IPA:",
            "US pronunciation:",
            "Transcription:"
        ]

        for prefix in prefixes_to_remove:
            if cleaned_response.lower().startswith(prefix.lower()):
                cleaned_response = cleaned_response[len(prefix):].strip()

        # Add slashes if missing
        if not cleaned_response.startswith('/'):
            cleaned_response = f"/{cleaned_response}"
        if not cleaned_response.endswith('/'):
            cleaned_response = f"{cleaned_response}/"

        return cleaned_response
