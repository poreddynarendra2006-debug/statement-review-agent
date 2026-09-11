"""3-Tier Semantic Column Mapping Pipeline.

Prioritizes:
1. Deterministic alias dictionary (0 cost, instantaneous)
2. Embedding-based semantic matching (TF-IDF subword/token cosine similarity or custom vectorizer)
3. LLM fallback for ambiguous columns (targeted prompt on remaining unmapped columns)

"Deterministic first, embeddings second, LLM last — cheapest reliable method wins."
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import (
    CANONICAL_COLUMNS,
    DETERMINISTIC_ALIAS_INDEX,
    ColumnSpec,
    clean_header_candidate,
    is_amount_to_ratio_violation,
    normalize_column_name,
)

logger = logging.getLogger("ingestion.mapper")


@dataclass
class ColumnMappingResult:
    """Outcome of mapping a single raw column to a canonical column."""

    raw_name: str
    canonical_name: Optional[str]
    method: str  # "deterministic", "embedding", "llm", "unmapped", "collision", "ambiguous"
    confidence: float
    reason: str
    candidates: List[str] = None

    def __post_init__(self):
        if self.candidates is None:
            self.candidates = []


class SemanticColumnMapper:
    """Maps raw CSV column names to canonical schema using the 3-tier strategy with collision protection."""

    def __init__(
        self,
        embedding_threshold: float = 0.65,
        ambiguity_margin: float = 0.05,
        llm_resolver: Optional[Callable[[str, List[Any], List[str]], Optional[str]]] = None,
        custom_embedding_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    ):
        """Initializes the mapper.

        Args:
            embedding_threshold: Minimum cosine similarity score for Tier 2 (0.0 - 1.0).
            ambiguity_margin: Score margin between top 2 candidates below which column is deemed ambiguous.
            llm_resolver: Optional callable (raw_col, sample_values, candidate_cols) -> target_col.
            custom_embedding_fn: Optional dense embedding function (e.g. OpenAI or sentence-transformers).
        """
        self.embedding_threshold = embedding_threshold
        self.ambiguity_margin = ambiguity_margin
        self.llm_resolver = llm_resolver or self._default_env_llm_resolver
        self.custom_embedding_fn = custom_embedding_fn
        self._init_embedding_index()

    def _init_embedding_index(self) -> None:
        """Precomputes vector representations for all canonical columns."""
        self.canonical_keys: List[str] = list(CANONICAL_COLUMNS.keys())

        # Build rich textual profile for each canonical column
        # includes canonical name, description, and aliases
        self.canonical_profiles: List[str] = []
        for key in self.canonical_keys:
            spec = CANONICAL_COLUMNS[key]
            alias_text = " ".join(spec.aliases).replace("_", " ")
            profile = f"{key.replace('_', ' ')} {spec.description} {alias_text}"
            self.canonical_profiles.append(profile)

        if self.custom_embedding_fn:
            try:
                self.canonical_vectors = self.custom_embedding_fn(self.canonical_profiles)
                self.vectorizer = None
                return
            except Exception as e:
                logger.warning(
                    f"Custom embedding function failed, falling back to TF-IDF vectorizer: {e}"
                )

        # Scikit-learn subword/character + word n-gram vectorizer
        # Combines word n-grams and character n-grams to capture typos, compound words, and semantic stems
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            lowercase=True,
            strip_accents="unicode",
        )
        self.canonical_vectors = self.vectorizer.fit_transform(self.canonical_profiles)

    def _default_env_llm_resolver(
        self,
        raw_col: str,
        sample_values: List[Any],
        candidate_cols: List[str],
    ) -> Optional[str]:
        """Default LLM fallback resolver using Gemini or OpenAI REST API when keys are configured."""
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                import requests
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
                prompt = (
                    f"You are an expert financial statement analyst and schema mapper.\n"
                    f"Map the column name '{raw_col}' (sample data: {sample_values}) to the best matching canonical field from this list:\n"
                    f"{candidate_cols}\n"
                    f"If no canonical field matches, respond with 'NONE'.\n"
                    f"Respond ONLY with the exact canonical name from the list or 'NONE'."
                )
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 20},
                }
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip().lower()
                        if text in candidate_cols:
                            return text
            except Exception as e:
                logger.warning(f"Gemini LLM resolver error: {e}")

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                import requests
                url = "https://api.openai.com/v1/chat/completions"
                prompt = (
                    f"Map the financial column '{raw_col}' (samples: {sample_values}) to one canonical field from:\n"
                    f"{candidate_cols}\n"
                    f"Respond ONLY with the exact canonical field name, or NONE."
                )
                headers = {
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 20,
                }
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        text = choices[0].get("message", {}).get("content", "").strip().lower()
                        if text in candidate_cols:
                            return text
            except Exception as e:
                logger.warning(f"OpenAI LLM resolver error: {e}")

        return None

    def map_columns(
        self,
        raw_columns: List[str],
        sample_data: Optional[Dict[str, List[Any]]] = None,
    ) -> Tuple[Dict[str, str], List[ColumnMappingResult], List[Dict[str, Any]]]:
        """Maps a list of raw column names to canonical schema names with duplicate collision protection.

        Args:
            raw_columns: List of raw column header strings from the CSV.
            sample_data: Optional dictionary of raw_col -> list of sample values for Tier 3 LLM fallback.

        Returns:
            Tuple of:
            - column_rename_map: Dict[raw_name, canonical_name]
            - mapping_results: List of ColumnMappingResult
            - collisions: List of collision dictionaries (when multiple columns matched same canonical field)
        """
        column_rename_map: Dict[str, str] = {}
        mapping_results: List[ColumnMappingResult] = []
        assigned_canonical: Dict[str, str] = {}  # canonical_name -> first_raw_col
        collisions: List[Dict[str, Any]] = []
        unmapped_indices: List[int] = []

        # ----------------------------------------------------
        # TIER 1: Deterministic Alias Dictionary (Exact Match First)
        # ----------------------------------------------------
        for idx, raw_col in enumerate(raw_columns):
            # Try 1: standard snake_case
            norm_direct = normalize_column_name(raw_col)
            # Try 2: snake_case after stripping unit parentheticals like ($M), (%), etc.
            norm_cleaned = normalize_column_name(clean_header_candidate(raw_col))

            canonical_match = DETERMINISTIC_ALIAS_INDEX.get(norm_direct) or DETERMINISTIC_ALIAS_INDEX.get(norm_cleaned)

            # Type Guard check: prevent amount headers from mapping to ratio fields
            if canonical_match and is_amount_to_ratio_violation(raw_col, canonical_match):
                canonical_match = None

            if canonical_match:
                if canonical_match not in assigned_canonical:
                    column_rename_map[raw_col] = canonical_match
                    assigned_canonical[canonical_match] = raw_col
                    mapping_results.append(
                        ColumnMappingResult(
                            raw_name=raw_col,
                            canonical_name=canonical_match,
                            method="deterministic",
                            confidence=1.0,
                            reason=f"Matched deterministic alias '{norm_cleaned}' -> '{canonical_match}'",
                        )
                    )
                    logger.debug(f"[Tier 1 Deterministic] '{raw_col}' -> '{canonical_match}'")
                else:
                    # DUPLICATE COLUMN COLLISION DETECTED
                    primary_col = assigned_canonical[canonical_match]
                    collision_info = {
                        "canonical_name": canonical_match,
                        "primary_column": primary_col,
                        "conflicting_column": raw_col,
                        "candidates": [primary_col, raw_col],
                        "action": "preserved_as_secondary",
                        "reason": f"Both '{primary_col}' and '{raw_col}' matched canonical field '{canonical_match}'",
                    }
                    collisions.append(collision_info)
                    logger.warning(
                        f"Column collision: '{raw_col}' and '{primary_col}' both match '{canonical_match}'. "
                        f"Preserving '{raw_col}' as secondary."
                    )
                    # Preserve secondary column under its own normalized name
                    sec_name = f"{norm_cleaned}_secondary" if norm_cleaned == canonical_match else norm_cleaned
                    column_rename_map[raw_col] = sec_name
                    mapping_results.append(
                        ColumnMappingResult(
                            raw_name=raw_col,
                            canonical_name=None,
                            method="collision",
                            confidence=0.99,
                            reason=collision_info["reason"],
                            candidates=[canonical_match],
                        )
                    )
            else:
                unmapped_indices.append(idx)

        # If all columns mapped deterministically or handled, return early
        if not unmapped_indices:
            return column_rename_map, mapping_results, collisions

        # ----------------------------------------------------
        # TIER 2: Embedding-based Semantic Matching (Fuzzy Match Last Resort)
        # ----------------------------------------------------
        remaining_indices_for_tier3: List[int] = []
        available_canonical = [k for k in self.canonical_keys if k not in assigned_canonical]

        if available_canonical:
            for idx in unmapped_indices:
                raw_col = raw_columns[idx]
                cleaned_query = clean_header_candidate(raw_col).replace("_", " ").lower().strip()

                # Vectorize query
                if self.custom_embedding_fn:
                    try:
                        q_vec = self.custom_embedding_fn([cleaned_query])
                        sims = cosine_similarity(q_vec, self.canonical_vectors)[0]
                    except Exception:
                        sims = np.zeros(len(self.canonical_keys))
                else:
                    q_vec = self.vectorizer.transform([cleaned_query])
                    sims = cosine_similarity(q_vec, self.canonical_vectors)[0]

                # Score all currently unassigned canonical columns, filtering out Type Guard violations
                scored_candidates: List[Tuple[float, str]] = []
                for c_idx, key in enumerate(self.canonical_keys):
                    if key in available_canonical:
                        if not is_amount_to_ratio_violation(raw_col, key):
                            scored_candidates.append((float(sims[c_idx]), key))

                scored_candidates.sort(key=lambda x: x[0], reverse=True)

                is_ambiguous = False
                qualifying_candidates = [
                    (s, k) for s, k in scored_candidates if s >= self.embedding_threshold
                ]

                # If 2 or more candidates qualify above threshold, or top 2 are within ambiguity margin
                if len(scored_candidates) >= 2:
                    top_sim, top_col = scored_candidates[0]
                    sec_sim, sec_col = scored_candidates[1]
                    if (top_sim - sec_sim) < self.ambiguity_margin and sec_sim >= 0.35:
                        is_ambiguous = True
                        logger.info(
                            f"Ambiguity detected for '{raw_col}': '{top_col}' ({top_sim:.4f}) vs "
                            f"'{sec_col}' ({sec_sim:.4f}) within margin {self.ambiguity_margin}."
                        )

                if len(qualifying_candidates) > 1:
                    is_ambiguous = True

                if scored_candidates and not is_ambiguous and qualifying_candidates:
                    best_sim, best_canonical = qualifying_candidates[0]
                    column_rename_map[raw_col] = best_canonical
                    assigned_canonical[best_canonical] = raw_col
                    available_canonical.remove(best_canonical)
                    mapping_results.append(
                        ColumnMappingResult(
                            raw_name=raw_col,
                            canonical_name=best_canonical,
                            method="embedding",
                            confidence=round(best_sim, 4),
                            reason=f"Cosine similarity {best_sim:.4f} >= {self.embedding_threshold}",
                            candidates=[best_canonical],
                        )
                    )
                    logger.info(
                        f"[Tier 2 Embedding] '{raw_col}' -> '{best_canonical}' (similarity: {best_sim:.4f})"
                    )
                    continue

                remaining_indices_for_tier3.append(idx)
        else:
            remaining_indices_for_tier3 = unmapped_indices

        # If all remaining columns mapped by embedding, return
        if not remaining_indices_for_tier3:
            return column_rename_map, mapping_results, collisions

        # ----------------------------------------------------
        # TIER 3: LLM Fallback for Ambiguous / Unresolved Columns
        # ----------------------------------------------------
        available_canonical = [k for k in self.canonical_keys if k not in assigned_canonical]

        for idx in remaining_indices_for_tier3:
            raw_col = raw_columns[idx]
            samples = []
            if sample_data and raw_col in sample_data:
                samples = [s for s in sample_data[raw_col][:3] if s is not None and str(s).strip()]

            # Determine if column was ambiguous (had multiple close candidates)
            cleaned_query = clean_header_candidate(raw_col).replace("_", " ").lower().strip()
            if self.vectorizer is not None:
                q_vec = self.vectorizer.transform([cleaned_query])
                sims = cosine_similarity(q_vec, self.canonical_vectors)[0]
                scored = [
                    (float(sims[c_idx]), key)
                    for c_idx, key in enumerate(self.canonical_keys)
                    if key in available_canonical and not is_amount_to_ratio_violation(raw_col, key)
                ]
                scored.sort(key=lambda x: x[0], reverse=True)
                competing_candidates = [k for s, k in scored[:3] if s >= 0.30]
            else:
                competing_candidates = []

            resolved_col = None
            if self.llm_resolver and available_canonical:
                try:
                    resolved_col = self.llm_resolver(raw_col, samples, available_canonical)
                    if resolved_col and is_amount_to_ratio_violation(raw_col, resolved_col):
                        resolved_col = None
                except Exception as e:
                    logger.warning(f"LLM resolver failed for column '{raw_col}': {e}")
                    resolved_col = None

            if resolved_col and resolved_col in available_canonical:
                column_rename_map[raw_col] = resolved_col
                assigned_canonical[resolved_col] = raw_col
                available_canonical.remove(resolved_col)
                mapping_results.append(
                    ColumnMappingResult(
                        raw_name=raw_col,
                        canonical_name=resolved_col,
                        method="llm",
                        confidence=0.85,
                        reason="Resolved via Tier 3 LLM fallback",
                        candidates=[resolved_col],
                    )
                )
                logger.info(f"[Tier 3 LLM] '{raw_col}' -> '{resolved_col}'")
            else:
                # Retain raw column with normalized snake_case name as an extra/unmapped column
                norm_extra = normalize_column_name(raw_col)
                column_rename_map[raw_col] = norm_extra
                is_amb = len(competing_candidates) >= 2
                method_name = "ambiguous" if is_amb else "unmapped"
                reason_str = (
                    f"Ambiguous column; competing candidates: {competing_candidates}"
                    if is_amb
                    else "No match across Tier 1, Tier 2, or Tier 3"
                )
                mapping_results.append(
                    ColumnMappingResult(
                        raw_name=raw_col,
                        canonical_name=None,
                        method=method_name,
                        confidence=0.0,
                        reason=reason_str,
                        candidates=competing_candidates,
                    )
                )
                logger.info(f"[{method_name.capitalize()} Column] '{raw_col}' preserved as '{norm_extra}'.")

        return column_rename_map, mapping_results, collisions
