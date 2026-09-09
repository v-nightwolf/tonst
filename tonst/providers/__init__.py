"""
tonst.providers
----------------
One module per LLM provider's caching mechanics. This package exists
because "prompt caching" is not one thing across providers -- Anthropic,
OpenAI, and Gemini each require a different request shape, have
different minimum cacheable lengths, price the discount differently
(and, for OpenAI, differently PER MODEL), and report usage at a
different JSON path. See each module's docstring for specifics, and the
README's "Multi-provider support" section for the cross-provider
comparison.

Deliberately NOT a shared abstract base class / common interface across
providers. The mechanics differ enough (Gemini alone has two distinct
caching mechanisms with different cost shapes) that forcing one
interface would hide real differences a caller needs to know about,
not simplify anything. The one thing providers do share is
tonst.cache_structuring.CacheUsageReport -- a plain data container each
provider's own parse_*_usage() function fills in, so callers get a
consistent shape to read regardless of which provider they used.

Historical note on naming: Anthropic's implementation predates this
package and still lives in tonst/cache_structuring.py rather than
tonst/providers/anthropic.py, for backward compatibility with the
existing public API. See ROADMAP.md for the plan to reconcile this.

Beyond the three named providers (anthropic in cache_structuring.py,
openai.py, gemini.py), generic.py is the actual answer to "any model":
a configurable adapter (GenericCacheConfig) that takes a provider's
minimum token length, pricing multipliers, and usage-JSON field paths
as plain data instead of hardcoded provider-specific code -- for
whatever provider tonst doesn't have (and may never have) a dedicated
module for. presets.py ships a couple of real, verified configs built
with it (AWS Bedrock's Converse API, Azure OpenAI's PTU-M tier) as
worked examples of platforms that need one because they genuinely
change the response shape or pricing versus the direct API they wrap.
"""
