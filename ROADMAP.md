# Roadmap & Open Discussion

This file tracks bigger decisions that aren't settled yet — things we're
watching for signal on rather than committing to. It's written so it can be
pasted directly into a pinned GitHub issue once the repo has enough traffic
for that issue to actually get seen.

## Open discussion topics

### Java support — interested?

**Status: not planned, actively watching for demand.**

tonst is Python-only today. A Java port isn't ruled out, but it isn't queued
up either — this is a "tell us if you need it" item, not a "someday maybe"
that quietly rots.

**What would trigger it:**
- 2–3 concrete requests for a Java/JVM version, or
- Comments/issues referencing Spring or other enterprise-Java use cases
  where tonst's approach (local redaction + caching in front of a paid LLM
  call) would apply

If you're reading this because you want Java support: comment on the pinned
issue (or open a new one) and say what you're building. That's the signal
we're waiting for.

**If/when it happens:** scope v1 to just the redaction + caching core (the
genuinely differentiated part — see `redact_llm.py` and `cache.py` in the
Python version), not a full port of everything. A smaller, focused Java v1
validates demand faster than porting the whole surface area up front.

---

*(Add future open-discussion topics below this line.)*
