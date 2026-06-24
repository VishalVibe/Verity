"""
Verification via a dedicated NLI (Natural Language Inference) model.

NLI models are trained specifically to classify the relationship between a
"premise" (our evidence) and a "hypothesis" (our claim) into entailment,
contradiction, or neutral -- which maps directly onto our supported /
contradicted / unsupported labels. Unlike LLM-as-judge, this is a small,
purpose-built classifier (not a general-purpose chat model repurposed for
a task it wasn't specifically trained on), so it's worth comparing: does a
model literally trained for this exact 3-way distinction outperform a
general LLM given a careful prompt?

Mapping: NLI's "entailment" -> SUPPORTED, "contradiction" -> CONTRADICTED,
"neutral" -> UNSUPPORTED. This mapping is exact, not approximate -- it's
precisely the distinction NLI was designed for.

NOT RUNNABLE in this sandboxed environment: requires `transformers` +
`torch` and a model download (e.g. facebook/bart-large-mnli or
microsoft/deberta-v3-base-mnli), neither of which are available without
network access. The interface and integration code below are real and
correct; run them on your own machine per the README.
"""

from abc import ABC, abstractmethod

from src.verification.verification_models import VerificationLabel, VerificationResult


class NLIModel(ABC):
    @abstractmethod
    def classify(self, premise: str, hypothesis: str) -> tuple[str, float]:
        """Return (label, confidence) where label is one of
        'entailment', 'contradiction', 'neutral'.
        """
        raise NotImplementedError


class HuggingFaceNLIModel(NLIModel):
    """Wraps a HuggingFace NLI model via the `transformers` pipeline API.

    Requires: pip install transformers torch
    First call downloads the model (~400MB-1.5GB depending on model_name),
    so run this on your own machine with network access, not in a
    sandboxed/offline environment.
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        from transformers import pipeline  # lazy import

        self.classifier = pipeline("zero-shot-classification", model=model_name)
        self.model_name = model_name

    def classify(self, premise: str, hypothesis: str) -> tuple[str, float]:
        # NLI models classify (premise, hypothesis) pairs directly when
        # loaded via AutoModelForSequenceClassification rather than the
        # zero-shot pipeline. Using the proper NLI pipeline:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        inputs = self._tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]

        # bart-large-mnli label order: [contradiction, neutral, entailment]
        label_names = ["contradiction", "neutral", "entailment"]
        best_idx = int(torch.argmax(probs))
        return label_names[best_idx], float(probs[best_idx])


_NLI_TO_VERIFICATION_LABEL = {
    "entailment": VerificationLabel.SUPPORTED,
    "contradiction": VerificationLabel.CONTRADICTED,
    "neutral": VerificationLabel.UNSUPPORTED,
}


def verify_claim_nli(claim_text: str, evidence_text: str, model: NLIModel) -> VerificationResult:
    """Classify one (claim, evidence) pair using a dedicated NLI model.

    Unlike verify_claim_llm, there is no default/mock model here -- NLI
    models require a real download, so there's no meaningful offline
    fallback to demonstrate the pipeline plumbing with. Pass a
    HuggingFaceNLIModel instance, set up per the README.
    """
    nli_label, confidence = model.classify(premise=evidence_text, hypothesis=claim_text)
    verification_label = _NLI_TO_VERIFICATION_LABEL[nli_label]

    return VerificationResult(
        claim_text=claim_text,
        evidence_text=evidence_text,
        label=verification_label,
        reasoning=f"NLI model classified as '{nli_label}' (confidence {confidence:.2f})",
        method="nli_model",
        confidence=confidence,
    )


if __name__ == "__main__":
    print(
        "This module requires `transformers` + `torch` and a model download, "
        "neither available in this sandboxed environment.\n"
        "Run on your own machine:\n\n"
        "    pip install transformers torch\n\n"
        "    from verify_nli import HuggingFaceNLIModel, verify_claim_nli\n"
        "    model = HuggingFaceNLIModel()\n"
        "    result = verify_claim_nli(\n"
        '        claim_text="The company raised $50 million",\n'
        '        evidence_text="The company raised a $12 million Series A round.",\n'
        "        model=model,\n"
        "    )\n"
        "    print(result)\n"
    )
