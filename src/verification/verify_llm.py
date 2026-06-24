"""
Verification via LLM-as-judge: given a claim and a piece of retrieved
evidence, classify whether the evidence supports, contradicts, or is
silent on (unsupported) the claim.

DESIGN NOTE -- the main risk with LLM-as-judge is sycophancy/leniency bias:
LLMs tend to default toward "supported" or "plausible" unless explicitly
instructed otherwise, because most training data rewards agreeable,
helpful-sounding answers. Left unchecked, this would make the verifier
nearly useless (a verifier that calls everything "supported" has 0 value).
The prompt below counters this by:
  1. Explicitly defining UNSUPPORTED as a real, expected, frequent outcome
     -- not an edge case -- so the model doesn't avoid it.
  2. Requiring the model to quote the specific part of the evidence it's
     relying on, which forces it to actually ground its judgment instead
     of pattern-matching on topic similarity.
  3. Giving a concrete example of each label so "unsupported" isn't an
     abstract category the model glosses over.

This is a real, testable hypothesis -- the eval script measures whether
this prompt actually reduces over-prediction of "supported" compared to a
naive prompt, see eval_verification.py.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from src.verification.verification_models import VerificationLabel, VerificationResult

JUDGE_PROMPT = """You are a strict fact-checker. Given a CLAIM and a piece of EVIDENCE \
(a sentence from a source document), decide whether the evidence SUPPORTS, \
CONTRADICTS, or is UNSUPPORTED BY (silent on) the claim.

Definitions -- read carefully, "unsupported" is a common and expected answer, not a rare edge case:
- SUPPORTED: the evidence directly confirms the claim's specific facts (same entities, numbers, dates).
- CONTRADICTED: the evidence directly conflicts with the claim (different number, different entity, opposite statement).
- UNSUPPORTED: the evidence does not confirm OR deny the claim -- it's simply about something \
else, or doesn't contain the specific detail the claim asserts. This includes claims that ADD \
information not present in the evidence at all, even if it sounds plausible. \
Do NOT mark something as SUPPORTED just because it's on the same general topic -- \
the evidence must contain the SPECIFIC fact in the claim.

Example: claim = "The company raised $50 million", evidence = "The company raised $12 million \
in a Series A round." -> CONTRADICTED (specific number conflicts).
Example: claim = "The company is backed by Sequoia Capital", evidence = "The company raised a \
$12 million round led by Greenfield Ventures." -> CONTRADICTED (different investor named).
Example: claim = "The company plans to expand to Europe next year", evidence = "The company \
raised a $12 million round led by Greenfield Ventures." -> UNSUPPORTED (evidence says nothing \
about expansion plans at all -- don't assume, don't guess).

You must quote the exact relevant span of the evidence (or write "none" if UNSUPPORTED) to \
justify your answer -- this forces you to ground the judgment in the actual text rather than \
general plausibility.

Return ONLY a JSON object with this exact shape, no other text, no markdown fences:
{{"label": "supported" | "contradicted" | "unsupported", "quoted_evidence_span": "...", "reasoning": "one sentence"}}

CLAIM: {claim}

EVIDENCE: {evidence}
"""


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def complete(self, prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def complete(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class GeminiProvider(LLMProvider):
    """Calls Google's Gemini API. Requires GEMINI_API_KEY set."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = os.environ.get("GEMINI_MODEL", model)
        self.min_interval_seconds = float(os.environ.get("GEMINI_MIN_INTERVAL_SECONDS", "13"))
        self._last_request_at = 0.0

    def complete(self, prompt: str) -> str:
        api_key = os.environ["GEMINI_API_KEY"]
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "1200")),
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        for attempt in range(5):
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)

            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    self._last_request_at = time.monotonic()
                    parsed = json.loads(response.read().decode("utf-8"))
                    break
            except urllib.error.HTTPError as e:
                self._last_request_at = time.monotonic()
                body = e.read().decode("utf-8", errors="replace")
                if e.code == 429 and attempt < 4:
                    retry_after = e.headers.get("Retry-After")
                    wait_seconds = float(retry_after) if retry_after else self.min_interval_seconds
                    time.sleep(wait_seconds)
                    continue
                raise RuntimeError(f"Gemini API request failed ({e.code}): {body}") from e

        try:
            return parsed["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected Gemini API response: {parsed!r}") from e
class GroqProvider(LLMProvider):
    """Calls Groq's API using the official SDK. Requires GROQ_API_KEY env var.

    Uses llama-3.3-70b-versatile by default — strong enough for LLM-as-judge,
    fast, and generous free tier (30 req/min, 14400/day).
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = os.environ.get("GROQ_MODEL", model)
        self._min_interval = 2.1
        self._last_request_at = 0.0

    def complete(self, prompt: str) -> str:
        from groq import Groq

        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
        )
        self._last_request_at = time.monotonic()
        return response.choices[0].message.content

class RuleBasedMockProvider(LLMProvider):
    """Deterministic, free, offline stand-in for testing pipeline plumbing.

    IMPORTANT: this is NOT a real verifier. It uses crude word-overlap /
    negation heuristics, only good enough to exercise the JSON parsing and
    downstream code paths without an API key. It is expected to perform far
    worse than a real LLM judge -- that gap IS the point: eval_verification.py
    reports both so you can see how much value the real LLM call adds versus
    this naive baseline.
    """

    _NEGATION_MARKERS = [" not ", " no ", " never ", " doesn't ", " does not ", " isn't "]

    def complete(self, prompt: str) -> str:
        claim_match = re.search(r"CLAIM:\s*(.+?)\n\nEVIDENCE:", prompt, re.DOTALL)
        evidence_match = re.search(r"EVIDENCE:\s*(.+?)\n*$", prompt, re.DOTALL)
        claim = (claim_match.group(1) if claim_match else "").strip().lower()
        evidence = (evidence_match.group(1) if evidence_match else "").strip().lower()

        claim_words = set(re.findall(r"\b\w+\b", claim)) - _STOPWORDS
        evidence_words = set(re.findall(r"\b\w+\b", evidence)) - _STOPWORDS
        overlap = claim_words & evidence_words
        overlap_ratio = len(overlap) / max(1, len(claim_words))

        if overlap_ratio < 0.25:
            label = "unsupported"
            reasoning = "Low word overlap between claim and evidence."
            span = "none"
        else:
            # Extremely crude: look for a number mismatch as a proxy for contradiction.
            claim_numbers = set(re.findall(r"\b\d+\b", claim))
            evidence_numbers = set(re.findall(r"\b\d+\b", evidence))
            if claim_numbers and evidence_numbers and claim_numbers != evidence_numbers:
                label = "contradicted"
                reasoning = "Numbers in claim and evidence differ."
            else:
                label = "supported"
                reasoning = "Sufficient word overlap, no obvious numeric conflict."
            span = evidence[:60]

        return json.dumps({"label": label, "quoted_evidence_span": span, "reasoning": reasoning})


_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "of", "in", "on",
    "at", "to", "for", "and", "or", "it", "its", "that", "this", "with", "as",
}


def _parse_judge_response(raw: str) -> dict:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse judge response as JSON. Raw:\n{raw}") from e

    if "label" not in parsed or parsed["label"] not in {"supported", "contradicted", "unsupported"}:
        raise ValueError(f"Judge response missing valid 'label' field: {parsed!r}")

    return parsed


def verify_claim_llm(
    claim_text: str, evidence_text: str, provider: LLMProvider | None = None
) -> VerificationResult:
    """Classify one (claim, evidence) pair using an LLM judge.

    Defaults to RuleBasedMockProvider (free, offline, NOT a real quality
    test -- see its docstring) so this is runnable without an API key.
    Pass AnthropicProvider() or OpenAIProvider() for real verification.
    """
    if provider is None:
        provider = RuleBasedMockProvider()

    prompt = JUDGE_PROMPT.format(claim=claim_text, evidence=evidence_text)
    raw = provider.complete(prompt)
    parsed = _parse_judge_response(raw)

    return VerificationResult(
        claim_text=claim_text,
        evidence_text=evidence_text,
        label=VerificationLabel(parsed["label"]),
        reasoning=parsed.get("reasoning", ""),
        method="llm_judge",
    )


if __name__ == "__main__":
    examples = [
        ("The company raised $50 million", "The company raised a $12 million Series A round led by Greenfield Ventures."),
        ("The company raised $12 million", "The company raised a $12 million Series A round led by Greenfield Ventures."),
        ("The company plans to expand to Europe next year", "The company raised a $12 million Series A round led by Greenfield Ventures."),
    ]
    for claim, evidence in examples:
        result = verify_claim_llm(claim, evidence)  # RuleBasedMockProvider
        print(f"Claim:    {claim}")
        print(f"Evidence: {evidence}")
        print(f"-> {result.label.value}  ({result.reasoning})\n")
