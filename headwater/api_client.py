"""OpenAI-compatible chat client for auditors queried through an API.

The endpoint comes from OPENAI_BASE_URL and the key from OPENAI_API_KEY. The prompt is
sent as the only user message. Transient errors are retried with exponential backoff, and
errors that no retry can fix stop the run.
"""

import os
import random
import sys
import threading
import time


class APIClient:
    def __init__(self, model: str, max_tokens: int = 128, temperature: float = 0.0,
                 timeout_s: float = 180.0, max_retries: int = 8):
        import openai
        base_url, api_key = os.environ.get("OPENAI_BASE_URL"), os.environ.get("OPENAI_API_KEY")
        if not base_url or not api_key:
            raise SystemExit("set OPENAI_BASE_URL and OPENAI_API_KEY")
        self.model, self.max_tokens, self.temperature, self.max_retries = model, max_tokens, temperature, max_retries
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)
        self.retryable = (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError,
                          openai.InternalServerError)
        self.usage = {"requests": 0, "retries": 0, "input_tokens": 0, "output_tokens": 0}
        self._lock = threading.Lock()

    def complete(self, prompt: str) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens, temperature=self.temperature)
            except self.retryable as exc:
                if attempt == self.max_retries:
                    raise
                print(f"[api] retry {attempt + 1}/{self.max_retries} after {type(exc).__name__}",
                      file=sys.stderr, flush=True)
                with self._lock:
                    self.usage["retries"] += 1
                time.sleep(min(120.0, 2.0 ** attempt) + random.random())
                continue
            with self._lock:
                self.usage["requests"] += 1
                if resp.usage is not None:
                    self.usage["input_tokens"] += resp.usage.prompt_tokens or 0
                    self.usage["output_tokens"] += resp.usage.completion_tokens or 0
            return resp.choices[0].message.content or ""
        raise AssertionError("unreachable")
