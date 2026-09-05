"""
local_model.py
--------------
Optional layer that uses a small local model (served by Ollama, e.g.
`llama3.2:1b` or `qwen2.5:1.5b`) to do the harder, semantic compression:
rewriting a verbose prompt into a shorter one that preserves meaning.

This is deliberately isolated behind a small interface and OFF by default,
because it's the riskiest piece: it costs local latency, needs Ollama
installed and running, and can (rarely) drop nuance a mechanical trim
would have kept. Mechanical trimming + caching should do most of the work;
this is the extra lever for teams with heavier prompts and capable hardware.

If Ollama isn't reachable, `compress()` fails soft and returns the
original text unchanged -- the pipeline should never break because the
optional local model wasn't available.
"""

from __future__ import annotations
import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

COMPRESSION_INSTRUCTION = (
    "Rewrite the following text to be as short as possible while preserving "
    "every fact, instruction, and constraint. Do not add commentary. "
    "Output only the rewritten text.\n\n---\n{text}"
)


class LocalCompressor:
    def __init__(self, model: str = "llama3.2:1b", ollama_url: str = DEFAULT_OLLAMA_URL, timeout: float = 8.0):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            resp = requests.get(self.ollama_url.replace("/api/generate", "/api/tags"), timeout=1.5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def compress(self, text: str) -> tuple[str, bool]:
        """Returns (possibly_compressed_text, was_compressed)."""
        try:
            resp = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": COMPRESSION_INSTRUCTION.format(text=text),
                    "stream": False,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            compressed = resp.json().get("response", "").strip()
            # Guard rail: only accept the compression if it's actually shorter
            # and not suspiciously tiny (which usually means the local model
            # misfired rather than genuinely compressed).
            if compressed and len(compressed) < len(text) and len(compressed) > len(text) * 0.15:
                return compressed, True
            return text, False
        except requests.RequestException:
            return text, False
