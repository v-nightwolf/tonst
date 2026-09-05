"""
demo.py
-------
Runs the SDK against a MOCK paid-API function (no real network call, no
API key needed) so you can see the pipeline and the reported savings.

Swap `mock_paid_api_call` for a real call to Claude/OpenAI/etc. and this
becomes a working integration -- nothing else about the SDK changes.

Run: python3 demo.py
(No setup needed -- this bootstraps its own missing dependencies below.)
"""

import subprocess
import sys


def _ensure_dependencies():
    """
    Self-healing dependency check: if requests/numpy aren't installed,
    install them automatically rather than making the user run a separate
    setup step first. This only installs the two packages this project
    declares in requirements.txt -- nothing else, nothing silent beyond
    a printed notice.
    """
    required = {"requests": "requests>=2.25", "numpy": "numpy>=1.20"}
    missing = []
    for module_name, pip_spec in required.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_spec)

    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        except subprocess.CalledProcessError:
            print(
                "\nAutomatic install failed (no internet connection, or pip is "
                "blocked on this machine).\n"
                "Please install manually and re-run:\n"
                f"    {sys.executable} -m pip install {' '.join(missing)}"
            )
            sys.exit(1)
        print()


_ensure_dependencies()

from tonst import TonstClient


def mock_paid_api_call(prompt: str) -> str:
    """
    Stands in for `anthropic.messages.create(...)` or similar.
    Just proves the SDK only ever hands this function the trimmed/redacted text.
    """
    print(f"    [PAID API RECEIVED] ({len(prompt)} chars): {prompt[:120]}...")
    return (
        "Thanks for reaching out. We've noted your account and will follow up "
        "with next steps shortly regarding your request."
    )


CHAT_HISTORY_BLOB = """You are a customer support assistant for Acme Cloud Hosting.
You are a customer support assistant for Acme Cloud Hosting.
Be polite and concise.
Be polite and concise.

Customer: Hi, I've been having trouble with my account.
Customer: Hi, I've been having trouble with my account.
Agent: I'm sorry to hear that, could you share more detail?
Customer: My email is jane.doe@example.com and my card ending isn't working,
full number is 4111 1111 1111 1111. Can you check my billing status? Also my
phone is +1 415-555-0132 in case you need to call me back.
"""


def main():
    client = TonstClient(call_fn=mock_paid_api_call, use_local_compression=False)

    print("=== Call 1: first time this question is asked ===")
    response, report = client.query(CHAT_HISTORY_BLOB)
    print(f"Response returned to app: {response}\n")
    print(f"Original tokens (est.):   {report.original_tokens}")
    print(f"Tokens actually sent:     {report.sent_tokens}")
    print(f"Tokens saved:             {report.tokens_saved} ({report.percent_saved}%)")
    print(f"PII fields redacted:      {report.redacted_fields}")
    print(f"Cache hit:                {report.cache_hit}")

    print("\n=== Call 2: a near-duplicate question (should hit cache, $0 spent) ===")
    similar_query = CHAT_HISTORY_BLOB.replace("Hi, I've been having trouble", "Hello, I'm having trouble")
    response2, report2 = client.query(similar_query)
    print(f"Response returned to app: {response2}\n")
    print(f"Tokens sent to paid API:  {report2.sent_tokens}  <-- should be 0")
    print(f"Cache hit:                {report2.cache_hit}")

    print("\n=== Cache stats ===")
    print(client.cache.stats())


if __name__ == "__main__":
    main()
