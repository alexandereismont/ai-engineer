"""
spacy_extractor.py — spaCy-based entity extractor for pension-domain text.

Recognises five custom entity types on top of spaCy's standard NER:
    PENSION_FUND   — named pension funds (Dutch and general)
    REGULATION     — legislative instruments (IORP II/III, FTK, WTP)
    METRIC         — quantitative/coverage metrics
    REQUIREMENT    — regulatory obligations

Usage
-----
    from spacy_extractor import PensionEntityExtractor

    extractor = PensionEntityExtractor()
    entities = extractor.extract("The FTK requires a coverage ratio above 105%.")
    batch    = extractor.batch_extract(["...", "..."])
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc, Span

# ---------------------------------------------------------------------------
# Entity ruler patterns
# ---------------------------------------------------------------------------

#: All custom patterns loaded into the EntityRuler.
#: Each dict follows spaCy's {"label": ..., "pattern": ...} format.
PENSION_PATTERNS: list[dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # PENSION_FUND
    # -------------------------------------------------------------------------
    {"label": "PENSION_FUND", "pattern": "pension fund"},
    {"label": "PENSION_FUND", "pattern": "pension funds"},
    {"label": "PENSION_FUND", "pattern": "pensioenfonds"},
    {"label": "PENSION_FUND", "pattern": "bedrijfstakpensioenfonds"},
    {"label": "PENSION_FUND", "pattern": "ABP"},
    {"label": "PENSION_FUND", "pattern": "PFZW"},
    {"label": "PENSION_FUND", "pattern": "PMT"},
    {"label": "PENSION_FUND", "pattern": "BpfBOUW"},
    {"label": "PENSION_FUND", "pattern": "Pensioenfonds Zorg en Welzijn"},
    {"label": "PENSION_FUND", "pattern": "Algemeen Burgerlijk Pensioenfonds"},
    # -------------------------------------------------------------------------
    # REGULATION
    # -------------------------------------------------------------------------
    {"label": "REGULATION", "pattern": "IORP II"},
    {"label": "REGULATION", "pattern": "IORP III"},
    {"label": "REGULATION", "pattern": "IORP"},
    {"label": "REGULATION", "pattern": "FTK"},
    {"label": "REGULATION", "pattern": "Financieel Toetsingskader"},
    {"label": "REGULATION", "pattern": "Wet toekomst pensioenen"},
    {"label": "REGULATION", "pattern": "WTP"},
    {"label": "REGULATION", "pattern": "SFDR"},
    {"label": "REGULATION", "pattern": "Sustainable Finance Disclosure Regulation"},
    {"label": "REGULATION", "pattern": "Pension Schemes Act"},
    # -------------------------------------------------------------------------
    # METRIC
    # -------------------------------------------------------------------------
    {"label": "METRIC", "pattern": "coverage ratio"},
    {"label": "METRIC", "pattern": "coverage ratios"},
    {"label": "METRIC", "pattern": "dekkingsgraad"},
    {"label": "METRIC", "pattern": "beleidsdekkingsgraad"},
    {"label": "METRIC", "pattern": "funding ratio"},
    {"label": "METRIC", "pattern": "funding level"},
    {"label": "METRIC", "pattern": "solvency ratio"},
    {"label": "METRIC", "pattern": "technical provisions"},
    {"label": "METRIC", "pattern": "vereist eigen vermogen"},
    # Token-pattern for "X% coverage ratio" or "coverage ratio of X%"
    {
        "label": "METRIC",
        "pattern": [
            {"LIKE_NUM": True},
            {"ORTH": "%", "OP": "?"},
            {"LOWER": "coverage"},
            {"LOWER": "ratio"},
        ],
    },
    # -------------------------------------------------------------------------
    # REQUIREMENT
    # -------------------------------------------------------------------------
    {"label": "REQUIREMENT", "pattern": "recovery plan"},
    {"label": "REQUIREMENT", "pattern": "herstelplan"},
    {"label": "REQUIREMENT", "pattern": "ORSA"},
    {"label": "REQUIREMENT", "pattern": "Own Risk and Solvency Assessment"},
    {"label": "REQUIREMENT", "pattern": "prudent person principle"},
    {"label": "REQUIREMENT", "pattern": "prudent person rule"},
    {"label": "REQUIREMENT", "pattern": "prudent person"},
    {"label": "REQUIREMENT", "pattern": "ESG integration"},
    {"label": "REQUIREMENT", "pattern": "ESG reporting"},
    {"label": "REQUIREMENT", "pattern": "fit and proper"},
    {"label": "REQUIREMENT", "pattern": "fit-and-proper"},
    {"label": "REQUIREMENT", "pattern": "governance review"},
    {"label": "REQUIREMENT", "pattern": "annual ORSA"},
]

# ---------------------------------------------------------------------------
# Dataclass for extracted entities
# ---------------------------------------------------------------------------


@dataclass
class ExtractedEntity:
    """A single entity extracted from text."""

    text: str
    label: str
    start_char: int
    end_char: int
    sentence: str


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------


class PensionEntityExtractor:
    """spaCy pipeline with a custom pension-domain EntityRuler.

    The pipeline loads ``en_core_web_sm`` and prepends a ``pension_ruler``
    EntityRuler component that takes priority over the statistical NER model.

    Args:
        model: spaCy model name (default ``en_core_web_sm``).
        include_spacy_ner: If True, also include entities from the statistical
            NER component (PERSON, ORG, DATE, etc.).
    """

    def __init__(self, model: str = "en_core_web_sm", include_spacy_ner: bool = True) -> None:
        self.include_spacy_ner = include_spacy_ner
        self.nlp = self._build_pipeline(model)

    def _build_pipeline(self, model: str) -> Language:
        """Load the spaCy model and add the pension EntityRuler."""
        try:
            nlp = spacy.load(model)
        except OSError:
            raise OSError(
                f"spaCy model '{model}' not found. "
                f"Install it with:  python -m spacy download {model}"
            ) from None

        # Add the ruler BEFORE the ner component so custom patterns take priority
        ruler: EntityRuler = nlp.add_pipe(
            "entity_ruler",
            name="pension_ruler",
            before="ner",
            config={"overwrite_ents": False},
        )
        ruler.add_patterns(PENSION_PATTERNS)
        return nlp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _span_to_entity(self, span: Span) -> ExtractedEntity:
        """Convert a spaCy Span to an ExtractedEntity."""
        # Find the sentence that contains this span
        sent_text = span.sent.text if span.sent else ""
        return ExtractedEntity(
            text=span.text,
            label=span.label_,
            start_char=span.start_char,
            end_char=span.end_char,
            sentence=sent_text.strip(),
        )

    def _filter_entities(self, doc: Doc) -> list[Span]:
        """Return entities of interest from the processed doc."""
        target_labels = {"PENSION_FUND", "REGULATION", "METRIC", "REQUIREMENT"}
        if self.include_spacy_ner:
            target_labels |= {"PERSON", "ORG", "DATE", "GPE", "MONEY", "PERCENT"}
        return [ent for ent in doc.ents if ent.label_ in target_labels]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Extract entities from a single text string.

        Args:
            text: Input document text.

        Returns:
            List of entity dicts with keys:
                text, label, start_char, end_char, sentence.
        """
        doc = self.nlp(text)
        return [asdict(self._span_to_entity(span)) for span in self._filter_entities(doc)]

    def batch_extract(self, texts: list[str]) -> list[dict[str, Any]]:
        """Extract entities from multiple texts efficiently using nlp.pipe.

        Args:
            texts: List of input document strings.

        Returns:
            List of result dicts, one per input text, each containing:
                {
                    "text":     original input text (first 120 chars),
                    "entities": list of entity dicts (same format as extract()),
                }
        """
        results: list[dict[str, Any]] = []
        for doc, text in zip(self.nlp.pipe(texts, batch_size=32), texts):
            entities = [
                asdict(self._span_to_entity(span)) for span in self._filter_entities(doc)
            ]
            results.append(
                {
                    "text": text[:120] + ("..." if len(text) > 120 else ""),
                    "entities": entities,
                }
            )
        return results

    def extract_with_context(self, text: str) -> dict[str, Any]:
        """Extract entities and return them grouped by label.

        Args:
            text: Input document text.

        Returns:
            Dict mapping label → list of unique entity texts found.
        """
        doc = self.nlp(text)
        grouped: dict[str, list[str]] = {}
        for span in self._filter_entities(doc):
            grouped.setdefault(span.label_, [])
            if span.text not in grouped[span.label_]:
                grouped[span.label_].append(span.text)
        return grouped


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_extractor: PensionEntityExtractor | None = None


def _get_default_extractor() -> PensionEntityExtractor:
    global _default_extractor  # noqa: PLW0603
    if _default_extractor is None:
        _default_extractor = PensionEntityExtractor()
    return _default_extractor


def extract(text: str) -> list[dict[str, Any]]:
    """Module-level convenience wrapper around PensionEntityExtractor.extract."""
    return _get_default_extractor().extract(text)


def batch_extract(texts: list[str]) -> list[dict[str, Any]]:
    """Module-level convenience wrapper around PensionEntityExtractor.batch_extract."""
    return _get_default_extractor().batch_extract(texts)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

DEMO_TEXTS = [
    (
        "Under IORP III Article 19, pension funds must maintain a coverage ratio above 100%. "
        "ABP and PFZW are the largest Dutch pension funds and must submit a recovery plan "
        "within 12 weeks if the dekkingsgraad falls below the minimum threshold."
    ),
    (
        "The FTK requires beleidsdekkingsgraad calculations over a rolling 12-month period. "
        "The prudent person principle mandates diversification across asset classes."
    ),
    (
        "IORP II Article 28 introduced the ORSA requirement, which is expanded under IORP III. "
        "The Own Risk and Solvency Assessment must be conducted at least annually."
    ),
]


if __name__ == "__main__":
    print("Loading pension entity extractor ...")
    extractor = PensionEntityExtractor()

    print("\n--- Single extract demo ---")
    for i, text in enumerate(DEMO_TEXTS, 1):
        print(f"\nText {i}: {text[:80]}...")
        entities = extractor.extract(text)
        for ent in entities:
            print(f"  [{ent['label']:15s}] '{ent['text']}'")

    print("\n--- Batch extract demo ---")
    batch_results = extractor.batch_extract(DEMO_TEXTS)
    for result in batch_results:
        print(f"\n  Input: {result['text']}")
        for ent in result["entities"]:
            print(f"    [{ent['label']:15s}] {ent['text']!r} @ {ent['start_char']}-{ent['end_char']}")
