"""RAGAS evaluator wrapper.

Provides a simple `RagasEvaluator` class with `evaluate(input, response, context=None)`
that runs deepeval metrics for answer relevancy, bias and hallucination and returns
a consolidated score and per-metric details.

This is a thin convenience layer around deepeval metrics:
 - RAGASAnswerRelevancyMetric (from deepeval.metrics.ragas)
 - BiasMetric (from deepeval.metrics.bias)
 - HallucinationMetric (from deepeval.metrics.hallucination)

Notes:
 - These metrics may call an evaluation LLM; set up your environment (API keys,
   GENAI_BASE_URL, etc.) or pass a `model` when constructing `RagasEvaluator`.
 - `context` is a list of strings used as the retrieval/context for hallucination
   and relevancy checks.
"""
from typing import List, Optional, Dict, Any

from deepeval.test_case import LLMTestCase
from deepeval.metrics.bias.bias import BiasMetric
from deepeval.metrics.hallucination.hallucination import HallucinationMetric
from deepeval.metrics.ragas import RAGASAnswerRelevancyMetric


class RagasEvaluator:
    """Evaluate an input/response pair for answer relevancy, bias and hallucination.

    Example:
        evaluator = RagasEvaluator(model="gpt-4o")
        results = evaluator.evaluate("What is X?", "X is ...", context=["doc text ..."])

    Returns a dict with per-metric scores, reasons (when available) and an
    overall average score under the `aggregate` key.
    """

    def __init__(self, model: Optional[Any] = None, embeddings: Optional[Any] = None):
        """Create the evaluator.

        Args:
            model: Optional evaluation model. Can be a deepeval model object or a
                model spec string (e.g., "gpt-3.5-turbo"). If omitted, deepeval will
                attempt to initialize a default evaluation model.
            embeddings: Optional embedding model (passed to RAGAS relevancy metric).
        """
        self.model = model
        self.embeddings = embeddings

    def evaluate(self, input_text: str, response_text: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the selected metrics and return results.

        Args:
            input_text: the original user query / prompt.
            response_text: the generated model response to evaluate.
            context: optional list of context strings (retrieval context / documents).

        Returns:
            dict: {
                'metrics': { metric_name: { 'score': float, 'reason': Optional[str]} },
                'aggregate': float
            }
        """
        context = context or []
        tc = LLMTestCase(input=input_text, actual_output=response_text, context=context, retrieval_context=context)

        metrics = []
        # instantiate metrics with provided model where supported
        try:
            relevancy = RAGASAnswerRelevancyMetric(model=self.model, embeddings=self.embeddings, _track=False)
            metrics.append(relevancy)
        except Exception:
            # fallback to generic AnswerRelevancy if RAGAS metric unavailable
            try:
                from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric

                metrics.append(AnswerRelevancyMetric(model=self.model, async_mode=False))
            except Exception:
                pass

        try:
            bias = BiasMetric(model=self.model, include_reason=True, async_mode=False)
            metrics.append(bias)
        except Exception:
            pass

        try:
            hall = HallucinationMetric(model=self.model, include_reason=True, async_mode=False)
            metrics.append(hall)
        except Exception:
            pass

        results = {}
        scores = []

        for metric in metrics:
            name = getattr(metric, "__name__", metric.__class__.__name__)
            try:
                score = metric.measure(tc, _show_indicator=False, _log_metric_to_confident=False)
                reason = getattr(metric, "reason", None)
            except Exception as e:
                score = None
                reason = f"metric error: {e}"

            results[name] = {"score": score, "reason": reason}
            if isinstance(score, (int, float)):
                scores.append(float(score))

        aggregate = sum(scores) / len(scores) if scores else None

        return {"metrics": results, "aggregate": aggregate}


__all__ = ["RagasEvaluator"]
