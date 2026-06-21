from __future__ import annotations
from ragpeek.session import GenerationSpan
from ragpeek.config import TracerConfig

# phrases that correlate with the model answering from training weights
# rather than the provided context
_HEDGING_PHRASES = [
    "i believe",
    "i think",
    "i'm not sure",
    "generally speaking",
    "typically",
    "usually",
    "in general",
    "it's possible that",
    "you might want to",
    "based on my knowledge",
    "as far as i know",
    "i'm not certain",
    "this may vary",
    "it depends",
]


def analyze_generation(span: GenerationSpan, config: TracerConfig) -> None:
    """
    Looks for signals that the LLM ignored the provided context.
    Uses heuristics only — no embedding model needed.
    """
    response_lower = span.response.lower()

    # Heuristic 1: hedging language
    # LLMs hedge when they're uncertain — which often means they're
    # not using the context and falling back to general knowledge
    hedging_found = [p for p in _HEDGING_PHRASES if p in response_lower]

    if hedging_found:
        span.diagnosis.append(
            f"Hedging language detected ({len(hedging_found)} phrases: "
            f"{', '.join(repr(p) for p in hedging_found[:3])}{'...' if len(hedging_found) > 3 else ''}). "
            f"The model may be answering from general knowledge rather than "
            f"the retrieved context."
        )

    # Heuristic 2: empty or very short response
    if len(span.response.strip()) < 20:
        span.diagnosis.append(
            "Response is very short or empty — possible generation failure. "
            "Check model availability and prompt formatting."
        )

    # no issues found
    if not span.diagnosis:
        span.diagnosis.append("Generation looks healthy — no obvious issues detected.")
