import os
import time

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# OpenRouter's embeddings endpoint can be flaky for some models (it intermittently
# returns an empty body, which the OpenAI SDK turns into "No embedding data received").
# Override the model if yours isn't served reliably, e.g.:
#   EMBEDDING_MODEL=openai/text-embedding-3-small uv run uvicorn server:app --reload
DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "google/gemini-embedding-001")

_MAX_RETRIES = 4


class TextEmbedder:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    def _create(self, inp):
        """Call the embeddings endpoint, retrying transient empty responses.

        OpenRouter intermittently returns no ``data``; we retry a few times with
        backoff before giving up with an actionable message. The empty case is caught
        by the ``if response.data`` check below — the SDK only raises its own
        ``ValueError("No embedding data received")`` when ``encoding_format`` is
        omitted, which we no longer do (see the comment on the call).
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                # `encoding_format` MUST be passed explicitly. When it is omitted the
                # OpenAI SDK quietly sends `encoding_format="base64"` as an optimisation
                # (it decodes the vectors again on the way out), and providers that do
                # not accept base64 — Google AI Studio, which serves the default
                # `google/gemini-embedding-001` — reject the request with a 400.
                response = self._client.embeddings.create(
                    model=self.model, input=inp, encoding_format="float"
                )
                if response.data:
                    return response.data
            except ValueError as exc:  # SDK's "No embedding data received"
                last_exc = exc
            time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(
            f"Embedding model '{self.model}' returned no data after {_MAX_RETRIES} attempts. "
            "OpenRouter may not be serving embeddings for this model reliably — set a different "
            "one via the EMBEDDING_MODEL env var, e.g. 'openai/text-embedding-3-small'."
        ) from last_exc

    def embed(self, text: str) -> list[float]:
        return self._create(text)[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = self._create(texts)
        # Happy path: the provider returned one vector per input (order via .index).
        if len(data) == len(texts):
            return [item.embedding for item in sorted(data, key=lambda x: x.index)]
        # Some providers return partial/empty batches — fall back to one call per text.
        return [self.embed(text) for text in texts]
