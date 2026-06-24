"""
Rule-based claim extraction.

Approach: split the answer into sentences, then split compound sentences
on coordinating conjunctions ("and", "but") when both halves look like
independent clauses (each has its own subject + verb-ish structure).

This is intentionally simple and has known weaknesses, documented below
in KNOWN_LIMITATIONS. The point of building this version first is to have
a free, fast, zero-dependency baseline to compare the LLM-based extractor
against -- not to make this perfect.
"""

import re

from src.claim_models import Claim, ExtractionResult

KNOWN_LIMITATIONS = """
1. Pronoun resolution: "It was built in 1889" extracted as a standalone
   claim loses the referent of "It". The LLM-based extractor resolves this;
   the rule-based one does not.
2. Compound splitting is naive: it only splits on " and " / " but " at the
   top level and can incorrectly split phrases like "salt and pepper" or
   "research and development" that aren't actually two claims.
3. Nested/subordinate clauses (e.g. "...which triggers an immune response")
   are not separated into their own claims even when they assert something
   independently checkable.
4. No semantic understanding -- this only operates on surface punctuation
   and connector words.
"""

# Conjunctions we attempt to split compound sentences on.
_SPLIT_CONJUNCTIONS = [" and ", " but "]

# A short blocklist of common noun-phrase collocations that should NOT be
# split even though they contain " and ". Not exhaustive -- this is a
# deliberately simple heuristic, see KNOWN_LIMITATIONS.
_NO_SPLIT_PHRASES = [
    "salt and pepper",
    "research and development",
    "trial and error",
    "back and forth",
    "more and more",
    "command and control",
]


def _sentence_split(text: str) -> list[tuple[str, int, int]]:
    """Split text into sentences, returning (sentence, start_char, end_char)."""
    # Split on '.', '!', '?' followed by whitespace and a capital letter,
    # while being careful not to split on common abbreviations.
    pattern = re.compile(r"(?<!\b[A-Z])(?<=[.!?])\s+(?=[A-Z])")
    sentences = []
    pos = 0
    for part in pattern.split(text):
        part = part.strip()
        if not part:
            continue
        start = text.index(part, pos)
        end = start + len(part)
        pos = end
        sentences.append((part, start, end))
    return sentences


def _looks_like_independent_clause(fragment: str) -> bool:
    """Very rough heuristic: does this fragment have something verb-like in it?

    We don't have a POS tagger here (kept dependency-free), so we just check
    the fragment has more than 2 words -- short fragments after a split are
    usually noun phrases, not independent clauses.
    """
    return len(fragment.split()) > 2


# Matches "between <number/year> and <number/year>" so we don't split
# numeric or date ranges, e.g. "between 1887 and 1889" or "between 10 and 20".
_RANGE_PATTERN = re.compile(r"\bbetween\s+\S+\s+and\s+\S+", re.IGNORECASE)


def _mask_ranges(sentence: str) -> tuple[str, dict[str, str]]:
    """Temporarily replace 'between X and Y' spans so the conjunction
    splitter doesn't see the 'and' inside them. Returns the masked
    sentence and a map to restore the original text afterward.
    """
    replacements: dict[str, str] = {}

    def _replacer(match: re.Match) -> str:
        token = f"__RANGE_{len(replacements)}__"
        replacements[token] = match.group(0)
        return token

    masked = _RANGE_PATTERN.sub(_replacer, sentence)
    return masked, replacements


def _unmask_ranges(text: str, replacements: dict[str, str]) -> str:
    for token, original in replacements.items():
        text = text.replace(token, original)
    return text


def _split_compound(sentence: str) -> list[str]:
    """Attempt to split a compound sentence into independent clause-like parts."""
    for phrase in _NO_SPLIT_PHRASES:
        if phrase in sentence.lower():
            return [sentence]

    masked, replacements = _mask_ranges(sentence)

    for conj in _SPLIT_CONJUNCTIONS:
        if conj in masked:
            left, _, right = masked.partition(conj)
            left, right = left.strip(), right.strip()
            if _looks_like_independent_clause(left) and _looks_like_independent_clause(right):
                left = _unmask_ranges(left, replacements)
                right = _unmask_ranges(right, replacements)
                # Capitalize the right half since it's now its own sentence.
                if right and right[0].islower():
                    right = right[0].upper() + right[1:]
                return [left, right]

    return [sentence]


def extract_claims_rule_based(answer: str) -> ExtractionResult:
    """Extract atomic claims from an answer using sentence + conjunction splitting.

    No external dependencies, no API calls. See KNOWN_LIMITATIONS above for
    where this falls short compared to the LLM-based extractor.
    """
    claims: list[Claim] = []

    for sentence, sent_start, _sent_end in _sentence_split(answer):
        parts = _split_compound(sentence)
        search_pos = sent_start
        for part in parts:
            # Find this part's actual offset within the original answer text
            # (search starting from where we left off, to handle repeated text).
            idx = answer.find(part.rstrip(".!? "), search_pos)
            if idx == -1:
                # Fallback: couldn't locate exact offset (e.g. due to the
                # capitalization rewrite above) -- use sentence start.
                idx = sent_start
            end = idx + len(part)
            claims.append(Claim(text=part.strip(), start_char=idx, end_char=end))
            search_pos = end

    return ExtractionResult(answer=answer, claims=claims, method="rule_based")


if __name__ == "__main__":
    example = (
        "The Eiffel Tower was built between 1887 and 1892 for the World's Fair. "
        "It stands 324 meters tall and was the tallest structure until 1950."
    )
    result = extract_claims_rule_based(example)
    print(f"Extracted {len(result)} claims:\n")
    for c in result.claims:
        print(f"  - {c.text}")
