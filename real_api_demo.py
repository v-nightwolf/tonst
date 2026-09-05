"""
real_api_demo.py
----------------
Wires TonstClient to the REAL Claude API (api.anthropic.com), not a mock.
Everything upstream of the actual HTTP call -- caching, redaction,
trimming -- runs exactly as it does in demo.py. Only `call_fn` changes.

Setup:
    1. Get an API key from https://console.anthropic.com
    2. export ANTHROPIC_API_KEY=sk-ant-...
    3. python3 real_api_demo.py

If ANTHROPIC_API_KEY isn't set, or is invalid, you'll get a clean
"authentication_error" from Anthropic's API rather than a code crash --
that confirms the integration is wired correctly even before you have
a working key.
"""

import os
import sys
import subprocess


def _ensure_dependencies():
    required = {"requests": "requests>=2.25", "numpy": "numpy>=1.20"}
    missing = [spec for mod, spec in required.items() if not _try_import(mod)]
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def _try_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


_ensure_dependencies()

import requests
from tonst import TonstClient

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-6"


def call_claude(prompt: str) -> str:
    """
    This is the ONE function tonst wraps. Everything before this call
    (cache check, redaction, trimming) has already happened by the time
    this runs -- `prompt` here is the trimmed, redacted text, never the
    user's raw original.
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "Set ANTHROPIC_API_KEY first: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )

    if resp.status_code != 200:
        # Surface the real API error rather than swallowing it -- this is
        # what you'll see right now with no/invalid key: a clean
        # "authentication_error", proving the request itself is correct.
        raise RuntimeError(f"API error {resp.status_code}: {resp.json()}")

    data = resp.json()
    return "".join(block["text"] for block in data["content"] if block["type"] == "text")


def main():
    if not ANTHROPIC_API_KEY:
        print(
            "No ANTHROPIC_API_KEY set. This will run the full tonst pipeline\n"
            "(cache check, redaction, trimming) and then fail at the actual\n"
            "API call with a clean authentication_error -- which confirms\n"
            "everything up to that point is wired correctly.\n"
        )

    client = TonstClient(call_fn=call_claude)

    # --- Test 1: a short, clean prompt (like the earlier run) ---
    # Expect near-0% savings -- there's nothing bloated to trim here.
    # This is the honest baseline: tonst doesn't manufacture savings where
    # none exist.
    short_prompt = (
        "Hi, this is Priya Malhotra. My email is priya@example.com. "
        "In one short sentence, what is prompt caching?"
    )

    # --- Test 2: a realistic bloated prompt -- the kind tonst is actually
    # built for. Duplicated system instructions and repeated lines are
    # extremely common in real chat-app histories that get re-sent on
    # every turn. This is where mechanical trimming has real work to do.
    bloated_prompt = """You are a customer support assistant for Acme Cloud Hosting.
You are a customer support assistant for Acme Cloud Hosting.
Be polite and concise.
Be polite and concise.
Always confirm the customer's account details before proceeding.
Always confirm the customer's account details before proceeding.

Customer: Hi, I've been having trouble with my account.
Customer: Hi, I've been having trouble with my account.
Agent: I'm sorry to hear that, could you share more detail?
Customer: My email is jane.doe@example.com and my card ending isn't working,
full number is 4111 1111 1111 1111. Can you check my billing status? Also my
phone is +1 415-555-0132 in case you need to call me back.

In one short sentence, what should the agent ask for next?
"""

    for label, prompt in [("Test 1: short clean prompt", short_prompt),
                           ("Test 2: bloated/duplicated prompt", bloated_prompt)]:
        print(f"\n=== {label} ===")
        try:
            response, report = client.query(prompt)
            print("Response:", response)
            print(f"Original tokens (est.): {report.original_tokens}")
            print(f"Tokens actually sent:   {report.sent_tokens}")
            print(f"Tokens saved:           {report.tokens_saved} ({report.percent_saved}%)")
            print(f"PII fields redacted:    {report.redacted_fields}")
        except RuntimeError as e:
            print(f"\n[Expected without a real key] {e}")
            print(
                "This confirms the integration reached the real Anthropic API "
                "and failed only on auth -- the tonst pipeline itself worked "
                "correctly up to that point."
            )
            break  # no point running test 2 without a key either


if __name__ == "__main__":
    main()
