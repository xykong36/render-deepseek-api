"""
Deepseek API Client - Pure API wrapper.

This module provides a lightweight wrapper around the Deepseek API.
It only handles API communication and does not contain business logic.
"""

import logging
from typing import List, Dict, Any
from openai import OpenAI


class DeepseekAPIError(Exception):
    """Custom exception for Deepseek API errors."""
    pass


class DeepseekClient:
    """
    Lightweight client for Deepseek API communication.

    Responsibilities:
    - Create and manage OpenAI client instance
    - Send requests to Deepseek API
    - Return raw API responses
    - Handle connection errors

    Does NOT handle:
    - Business logic
    - Data validation
    - Complex response parsing
    """

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        """
        Initialize the Deepseek API client.

        Args:
            api_key: Deepseek API key
            base_url: API base URL (default: https://api.deepseek.com)

        Raises:
            DeepseekAPIError: If API key is invalid or missing
        """
        if not api_key or not api_key.strip():
            raise DeepseekAPIError("API key is required and cannot be empty")

        self.api_key = api_key
        self.base_url = base_url
        self.client = self._create_client()
        self.logger = logging.getLogger(__name__)

    def _create_client(self) -> OpenAI:
        """Create and configure the OpenAI client for Deepseek."""
        try:
            return OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except Exception as e:
            raise DeepseekAPIError(f"Failed to create API client: {e}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1000,
        model: str = "deepseek-chat"
    ) -> str:
        """
        Send a chat completion request to Deepseek API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 - 2.0)
            max_tokens: Maximum tokens in response
            model: Model name (default: deepseek-chat)

        Returns:
            Raw response content string from API

        Raises:
            DeepseekAPIError: If API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )

            content = response.choices[0].message.content
            if content is None:
                raise DeepseekAPIError("API returned empty response")

            return content.strip()

        except Exception as e:
            self.logger.error(f"Deepseek API call failed: {e}")
            raise DeepseekAPIError(f"API call failed: {e}")

    def simple_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> str:
        """
        Simplified chat completion with system and user prompts.

        Args:
            system_prompt: System role prompt
            user_prompt: User message prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Raw response content string from API

        Raises:
            DeepseekAPIError: If API call fails
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
