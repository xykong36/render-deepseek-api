"""
Expression Service - Handles expression generation from sentences.

Business logic layer for extracting phrasal verbs, idioms, collocations, etc.
This is the most complex service with batch processing and deduplication.
"""

import json
import re
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from bson import ObjectId

from .deepseek_client import DeepseekClient, DeepseekAPIError
from .prompts import EXPRESSION_SYSTEM_PROMPT, get_expression_user_prompt


logger = logging.getLogger(__name__)


class ExpressionService:
    """
    Service for generating expressions from sentences.

    Responsibilities:
    - Split sentences into batches by token limit
    - Process batches in parallel with retry logic
    - Parse and validate expression JSON
    - Deduplicate and merge expressions
    - Match expressions to sentences
    """

    def __init__(self, client: DeepseekClient):
        """
        Initialize expression service.

        Args:
            client: DeepseekClient instance for API calls
        """
        self.client = client
        self.logger = logger

    def generate_expressions(
        self,
        sentences: List[Dict[str, Any]],
        episode_id: int = 1,
        max_input_tokens: int = 2500,
        max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Generate expressions from a list of sentences with parallel batch processing.

        Args:
            sentences: List of sentence dictionaries with 'en', 'zh', etc.
            episode_id: Episode ID for the expressions
            max_input_tokens: Maximum input tokens per batch (default: 2500)
            max_workers: Maximum parallel workers (default: 4)

        Returns:
            List of expression dictionaries

        Raises:
            DeepseekAPIError: If expression generation fails
        """
        if not sentences:
            self.logger.warning("No sentences provided for expression generation")
            return []

        # Split into batches
        batches = self._split_sentences_by_token_limit(sentences, max_input_tokens)

        if len(batches) > 1:
            self.logger.info(f"Split {len(sentences)} sentences into {len(batches)} batches")

        all_expressions = []

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {}
            for batch_idx, batch_sentences in enumerate(batches, 1):
                future = executor.submit(
                    self._process_batch_with_retry,
                    batch_sentences,
                    episode_id,
                    batch_idx
                )
                future_to_batch[future] = batch_idx

            # Collect results
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_expressions = future.result()
                    all_expressions.extend(batch_expressions)
                    self.logger.info(f"✅ Batch {batch_idx}/{len(batches)}: {len(batch_expressions)} expressions")
                except Exception as e:
                    self.logger.error(f"❌ Batch {batch_idx}/{len(batches)} failed: {e}")

        # Deduplicate and merge
        deduplicated = self._deduplicate_expressions(all_expressions)

        # Assign ObjectIds
        for expr in deduplicated:
            expr['expression_id'] = str(ObjectId())

        self.logger.info(
            f"Generated {len(deduplicated)} unique expressions from {len(sentences)} sentences "
            f"({len(all_expressions)} before deduplication)"
        )

        return deduplicated

    def _split_sentences_by_token_limit(
        self,
        sentences: List[Dict[str, Any]],
        max_input_tokens: int
    ) -> List[List[Dict[str, Any]]]:
        """Split sentences into batches by token limit."""
        batches = []
        current_batch = []
        current_tokens = 0

        for sentence in sentences:
            en_text = sentence.get('en', '')
            sentence_tokens = len(en_text) // 3  # Rough estimate

            if current_tokens + sentence_tokens > max_input_tokens and current_batch:
                batches.append(current_batch)
                current_batch = [sentence]
                current_tokens = sentence_tokens
            else:
                current_batch.append(sentence)
                current_tokens += sentence_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def _process_batch_with_retry(
        self,
        sentences: List[Dict[str, Any]],
        episode_id: int,
        batch_idx: int,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """Process a batch with retry logic."""
        import time

        last_error = None

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Processing batch {batch_idx} (attempt {attempt + 1}/{max_retries})")

                if attempt > 0:
                    delay = 2 ** attempt
                    self.logger.info(f"Retrying after {delay}s delay...")
                    time.sleep(delay)

                expressions = self._generate_expressions_from_batch(sentences, episode_id)

                if expressions:
                    return expressions
                else:
                    self.logger.warning(f"Batch {batch_idx} returned no expressions")

            except Exception as e:
                last_error = e
                self.logger.warning(f"Batch {batch_idx} attempt {attempt + 1} failed: {e}")

        if last_error:
            raise last_error
        else:
            return []

    def _generate_expressions_from_batch(
        self,
        sentences: List[Dict[str, Any]],
        episode_id: int
    ) -> List[Dict[str, Any]]:
        """Generate expressions from a single batch."""
        # Combine sentences for context
        sentences_text = "\n".join([
            f"{i+1}. {s.get('en', '')}"
            for i, s in enumerate(sentences)
            if s.get('en', '').strip()
        ])

        # Call API
        raw_result = self.client.simple_completion(
            system_prompt=EXPRESSION_SYSTEM_PROMPT,
            user_prompt=get_expression_user_prompt(sentences_text),
            temperature=0.3,
            max_tokens=6000
        )

        # Parse JSON
        expressions = self._robust_json_parse(raw_result)
        if not expressions:
            self.logger.warning("No valid expressions parsed from API response")
            return []

        # Process each expression
        processed = []
        for expr in expressions:
            if not isinstance(expr, dict):
                continue

            normalized = self._normalize_expression_data(expr)
            phrase = normalized.get('phrase', '')
            if not phrase:
                continue

            # Generate slug
            slug = phrase.lower().replace(' ', '-').replace("'", '')

            # Match sentences
            matched_sentences = []
            matched_indices = normalized.get('matched_sentence_indices', [])

            for sent_idx in matched_indices:
                if 1 <= sent_idx <= len(sentences):
                    sentence = sentences[sent_idx - 1]
                    en_text = sentence.get('en', '')

                    matched_highlight = self._find_phrase_in_sentence(
                        phrase, en_text, normalized.get('forms', [])
                    )

                    matched_sentences.append({
                        'sentence_id': sentence.get('sentence_id'),
                        'episode_id': sentence.get('episode_id', episode_id),
                        'episode_sequence': sentence.get('episode_sequence', sent_idx),
                        'en': en_text,
                        'zh': sentence.get('zh', ''),
                        'phonetic_us': sentence.get('phonetic_us', ''),
                        'start_ts': sentence.get('start_ts', 0),
                        'end_ts': sentence.get('end_ts', 0),
                        'duration': sentence.get('duration', 0),
                        'sentence_hash': sentence.get('sentence_hash', ''),
                        'highlight_entries': sentence.get('highlight_entries', []),
                        'matched_highlight': matched_highlight
                    })

            # Build complete expression
            processed.append({
                'expression_id': str(ObjectId()),
                'phrase': phrase,
                'phonetic': normalized.get('phonetic', ''),
                'type': normalized.get('type', 'expression'),
                'meanings': normalized.get('meanings', []),
                'wordRelations': normalized.get('wordRelations', {
                    'synonyms': [], 'antonyms': [], 'similar': []
                }),
                'relatedExpressions': normalized.get('relatedExpressions', []),
                'analysis': normalized.get('analysis', []),
                'tags': normalized.get('tags', []),
                'episodes': [str(episode_id)],
                'matched_sentences': matched_sentences,
                'slug': slug,
                'forms': normalized.get('forms', [phrase])
            })

        return processed

    def _robust_json_parse(self, json_str: str) -> List[Dict[str, Any]]:
        """Robustly parse JSON with error recovery."""
        # Extract JSON from markdown
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        elif not json_str.strip().startswith('['):
            array_match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if array_match:
                json_str = array_match.group(0)

        try:
            result = json.loads(json_str)
            if isinstance(result, list):
                return result
            else:
                self.logger.warning(f"JSON result is not a list: {type(result)}")
                return []
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parse failed: {e}")
            return []

    def _normalize_expression_data(self, expr: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize expression data to match TypeScript interface."""
        normalized = {}

        normalized['phrase'] = expr.get('phrase', '')
        normalized['phonetic'] = expr.get('phonetic', '')
        normalized['type'] = expr.get('type', 'expression')
        normalized['tags'] = expr.get('tags', [])
        normalized['forms'] = expr.get('forms', [normalized['phrase']])
        normalized['matched_sentence_indices'] = expr.get('matched_sentence_indices', [])

        # Normalize meanings
        meanings = expr.get('meanings', [])
        normalized_meanings = []
        for meaning in meanings:
            if isinstance(meaning, dict) and 'translations' in meaning:
                normalized_meanings.append({
                    'translations': meaning.get('translations', {'zh': '', 'en': ''}),
                    'examples': meaning.get('examples', [])
                })

        normalized['meanings'] = normalized_meanings

        # Normalize wordRelations
        wr = expr.get('wordRelations', {})
        normalized['wordRelations'] = {
            'synonyms': self._normalize_phrase_list(wr.get('synonyms', [])),
            'antonyms': self._normalize_phrase_list(wr.get('antonyms', [])),
            'similar': self._normalize_phrase_list(wr.get('similar', []))
        }

        # Normalize relatedExpressions
        related = expr.get('relatedExpressions', [])
        normalized_related = []
        for item in related:
            if isinstance(item, dict) and 'type' in item and 'phrase' in item:
                normalized_related.append({
                    'type': item.get('type', 'related'),
                    'phrase': item.get('phrase', ''),
                    'translation': item.get('translation', ''),
                    'examples': item.get('examples', [])
                })

        normalized['relatedExpressions'] = normalized_related

        # Normalize analysis
        analysis = expr.get('analysis', [])
        normalized_analysis = []
        for item in analysis:
            if isinstance(item, dict) and 'type' in item and 'content' in item:
                normalized_analysis.append(item)

        normalized['analysis'] = normalized_analysis

        return normalized

    def _normalize_phrase_list(self, items: list) -> list:
        """Normalize phrase list to {phrase, translation} format."""
        normalized = []
        for item in items:
            if isinstance(item, dict) and 'phrase' in item:
                normalized.append({
                    'phrase': item.get('phrase', ''),
                    'translation': item.get('translation', '')
                })
            elif isinstance(item, str):
                normalized.append({'phrase': item, 'translation': ''})

        return normalized

    def _find_phrase_in_sentence(
        self,
        phrase: str,
        sentence: str,
        forms: list
    ) -> str:
        """Find the best matching form of a phrase in a sentence."""
        sentence_lower = sentence.lower()

        # Try each form
        for form in forms:
            if form.lower() in sentence_lower:
                start_idx = sentence_lower.index(form.lower())
                return sentence[start_idx:start_idx + len(form)]

        # Try base phrase
        if phrase.lower() in sentence_lower:
            start_idx = sentence_lower.index(phrase.lower())
            return sentence[start_idx:start_idx + len(phrase)]

        return phrase

    def _deduplicate_expressions(
        self,
        expressions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate expressions based on phrase."""
        if not expressions:
            return []

        # Group by phrase
        phrase_to_expressions = {}
        for expr in expressions:
            phrase = expr.get('phrase', '').lower().strip()
            if not phrase:
                continue

            if phrase not in phrase_to_expressions:
                phrase_to_expressions[phrase] = []
            phrase_to_expressions[phrase].append(expr)

        # Merge duplicates
        deduplicated = []
        for phrase, expr_list in phrase_to_expressions.items():
            if len(expr_list) == 1:
                deduplicated.append(expr_list[0])
            else:
                merged = self._merge_expressions(expr_list)
                deduplicated.append(merged)

        self.logger.info(f"Deduplicated {len(expressions)} -> {len(deduplicated)} expressions")
        return deduplicated

    def _merge_expressions(self, expr_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple expressions with same phrase."""
        if not expr_list:
            return {}

        merged = expr_list[0].copy()

        # Merge matched_sentences
        all_matched = []
        seen_ids = set()
        for expr in expr_list:
            for sent in expr.get('matched_sentences', []):
                sent_id = sent.get('sentence_id')
                if sent_id and sent_id not in seen_ids:
                    all_matched.append(sent)
                    seen_ids.add(sent_id)

        merged['matched_sentences'] = all_matched

        # Merge tags
        all_tags = set()
        for expr in expr_list:
            all_tags.update(expr.get('tags', []))
        merged['tags'] = list(all_tags)

        # Merge forms
        all_forms = set()
        for expr in expr_list:
            all_forms.update(expr.get('forms', []))
        merged['forms'] = list(all_forms)

        return merged
