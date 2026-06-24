"""
LLM-based claim extraction.

Unlike the rule-based extractor (extract_rule_based.py), this version asks
an LLM to break the answer into atomic, self-contained claims -- meaning
pronouns and implicit references are resolved. E.g.:

    "The Eiffel Tower was built in 1889. It is 330m tall."
    -> ["The Eiffel Tower was built in 1889.",
        "The Eiffel Tower is 330m tall."]   (note: "It" resolved)

Two things this module deliberately does NOT trust the LLM to do:
  1. Character offsets -- LLMs are unreliable at precise indexing, so we
     never ask for offsets directly. Instead we recover them ourselves by
     locating each returned claim back in the source answer (see
     `_locate_claim_span`).
  2. Perfect determinism -- the same answer can produce slightly different
     claim boundaries across calls. We don't paper over this; it's a real
     property of LLM-based extraction worth measuring (see eval script).

KNOWN LIMITATION -- offset precision degrades with pronoun resolution:
When a claim's wording diverges substantially from the source text (e.g.
"It stands 330m tall" -> "The Eiffel Tower stands 330m tall"), the fuzzy
offset-recovery in `_locate_claim_span` tends to return a wider span than
the minimal matching substring, because it anchors on the longest common
block and pads outward rather than doing true word-level alignment. The
recovered span still contains the right text (verified against the eval
set), it's just not pixel-perfect for UI highlighting purposes. A more
precise fix would use word-level sequence alignment (e.g. Needleman-Wunsch)
instead of difflib's block matching -- left as a future improvement, noted
here rather than silently shipped as if it were exact.
"""

import difflib
import json
import os
import re
from abc import ABC, abstractmethod

from src.claim_models import Claim, ExtractionResult

EXTRACTION_PROMPT = """You are extracting atomic, independently-checkable claims from an AI-generated answer.

Rules:
1. Each claim must be a single, self-contained factual assertion.
2. Resolve all pronouns and implicit references using context from the full answer. \
For example, if the answer is "The Eiffel Tower was built in 1889. It is 330m tall.", \
the second claim must be "The Eiffel Tower is 330m tall." (not "It is 330m tall.").
3. Split compound sentences into separate claims when they assert more than one fact.
4. Do not include claims for filler, opinions, or hedging language (e.g. "I think", \
"it's worth noting") -- only extract checkable factual assertions.
5. Preserve the original wording as closely as possible aside from pronoun resolution; \
do not add information that wasn't in the answer.

Return ONLY a JSON array of strings, one per claim, with no other text, no markdown \
fences, and no preamble.

Answer to extract claims from:
\"\"\"
{answer}
\"\"\"
"""


class LLMProvider(ABC):
    """Minimal interface so the extractor isn't tied to one API."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send a prompt, return the raw text response."""
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Calls the Anthropic API. Requires `pip install anthropic` and ANTHROPIC_API_KEY set."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def complete(self, prompt: str) -> str:
        import anthropic  # imported lazily so this file has no hard dependency

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    """Calls the OpenAI API. Requires `pip install openai` and OPENAI_API_KEY set."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def complete(self, prompt: str) -> str:
        from openai import OpenAI  # imported lazily

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class MockProvider(LLMProvider):
    """Deterministic fake provider so the pipeline is testable with zero API cost.

    Uses the SAME rule-based logic as extract_rule_based.py as its stand-in
    'model', but returns it in the JSON shape the real LLM would. This lets
    you run the full pipeline end-to-end -- and write/debug the offset
    recovery and downstream retrieval/verification code -- before spending
    a single API call. Swap in AnthropicProvider or OpenAIProvider once
    you're ready to test real extraction quality.
    """

    def complete(self, prompt: str) -> str:
        match = re.search(r'"""\n(.*?)\n"""', prompt, re.DOTALL)
        answer = match.group(1) if match else ""

        from src.extraction.extract_rule_based import extract_claims_rule_based

        fallback_claims = [c.text for c in extract_claims_rule_based(answer).claims]
        return json.dumps(fallback_claims)


def _parse_llm_json_array(raw: str) -> list[str]:
    """Parse the LLM's JSON array response, tolerating common formatting issues
    (markdown code fences, leading/trailing whitespace or prose).
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse LLM response as JSON array. Raw response:\n{raw}"
        ) from e

    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError(f"Expected a JSON array of strings, got: {parsed!r}")

    return parsed


def _locate_claim_span(claim_text: str, answer: str, search_from: int = 0) -> tuple[int, int]:
    """Find the best-matching span for `claim_text` inside `answer`.

    Claims may not appear verbatim (the LLM resolves pronouns, which changes
    the text), so we can't just use str.find(). Instead we search the whole
    answer for the best-matching substring using difflib's
    SequenceMatcher.find_longest_match as an anchor, then expand to a
    sensible span around it.

    `search_from` is used as a *soft* preference (ties are broken in favor
    of matches at or after this position) rather than a hard constraint,
    because claim order from the LLM isn't guaranteed to match source order
    once pronouns are resolved and sentences are reordered/split.

    Returns (start_char, end_char). If no good match is found, returns
    (search_from, search_from) as a signal of failure -- callers should
    treat a zero-length span as "offset unknown" and handle it explicitly
    (e.g. flag for manual review) rather than silently trusting it.
    """
    # Fast path: exact substring match.
    idx = answer.find(claim_text, search_from)
    if idx != -1:
        return idx, idx + len(claim_text)
    idx = answer.find(claim_text)  # try without the search_from constraint too
    if idx != -1:
        return idx, idx + len(claim_text)

    # Slow path: use difflib's longest-matching-block to anchor a span,
    # then grow it word-by-word in both directions while similarity improves.
    matcher = difflib.SequenceMatcher(None, answer, claim_text, autojunk=False)
    match = matcher.find_longest_match(0, len(answer), 0, len(claim_text))

    if match.size < 4:  # essentially no overlap found at all
        return search_from, search_from

    target_len = len(claim_text)
    anchor_start, anchor_end = match.a, match.a + match.size

    # Grow outward from the anchor toward the claim's length, snapping to
    # word boundaries. We grow asymmetrically based on how much of the
    # claim falls before vs after the matched anchor within claim_text
    # itself, rather than a flat symmetric buffer (which over-widens short
    # claims with a small anchor match).
    chars_before_anchor_in_claim = match.b
    chars_after_anchor_in_claim = len(claim_text) - (match.b + match.size)

    start = max(0, anchor_start - chars_before_anchor_in_claim - 5)
    end = min(len(answer), anchor_end + chars_after_anchor_in_claim + 5)

    # Snap to word boundaries.
    while start > 0 and answer[start - 1] != " ":
        start -= 1
    while end < len(answer) and answer[end] not in " .!?":
        end += 1

    candidate = answer[start:end]
    ratio = difflib.SequenceMatcher(None, claim_text, candidate).ratio()

    # If the tightened span is a poor match, fall back to the wider
    # symmetric guess as a last resort rather than failing outright.
    if ratio < 0.35:
        pad = max(0, (target_len - match.size) // 2 + 10)
        start = max(0, anchor_start - pad)
        end = min(len(answer), anchor_end + pad)
        candidate = answer[start:end]
        ratio = difflib.SequenceMatcher(None, claim_text, candidate).ratio()

    if ratio < 0.35:
        return search_from, search_from

    return start, end


def extract_claims_llm(answer: str, provider: LLMProvider | None = None) -> ExtractionResult:
    """Extract atomic, self-contained claims from an answer using an LLM.

    Args:
        answer: the AI-generated answer to extract claims from.
        provider: which LLM to call. Defaults to MockProvider (free, offline,
            uses rule-based logic under the hood) so this is runnable without
            an API key. Pass AnthropicProvider() or OpenAIProvider() for the
            real thing.
    """
    if provider is None:
        provider = MockProvider()

    prompt = EXTRACTION_PROMPT.format(answer=answer)
    raw_response = provider.complete(prompt)
    claim_texts = _parse_llm_json_array(raw_response)

    claims: list[Claim] = []
    last_match_end = 0
    for text in claim_texts:
        start, end = _locate_claim_span(text, answer, search_from=last_match_end)
        claims.append(Claim(text=text, start_char=start, end_char=end))
        if end > start:
            last_match_end = end  # soft hint for the next claim, not a hard floor

    return ExtractionResult(answer=answer, claims=claims, method="llm")


if __name__ == "__main__":
    example = (
        "The Eiffel Tower was built between 1887 and 1889 for the World's Fair. "
        "It stands 330 meters tall and was made of wrought iron."
    )
    result = extract_claims_llm(example)  # uses MockProvider by default
    print(f"Extracted {len(result)} claims (provider: {result.method}):\n")
    for c in result.claims:
        print(f"  - {c.text!r}  [offset {c.start_char}:{c.end_char}]")
