from .client import TonstClient, OptimizationReport, StructuredRedactionResult
from .cache_structuring import (
    PromptParts,
    structure_for_caching,
    build_anthropic_cache_request,
    check_cache_eligibility,
    CacheEligibility,
    parse_anthropic_usage,
    CacheUsageReport,
)
from .compactor import (
    HistoryCompactor,
    compact_history,
    CompactionResult,
    compact_history_rolling,
    RollingSummary,
    RollingCompactionResult,
    FoldJob,
    run_fold_job,
    estimate_fold_payback,
)
from .trim import flatten_messages
from .tool_optimizer import (
    select_tools,
    ToolSelection,
    ToolSession,
    build_anthropic_deferred_tools,
    estimate_tool_tokens,
    DEFERRED_TOOLS_SYSTEM_HINT,
)
from .rag import optimize_chunks, ChunkSelection
from .token_count import AnthropicTokenCounter, GeminiTokenCounter
from .summarizers import AnthropicSummarizer, GeminiSummarizer
from .savings_log import SavingsLog, summarize as summarize_savings, SavingsSummary
from . import providers

__all__ = [
    "TonstClient",
    "OptimizationReport",
    "StructuredRedactionResult",
    "PromptParts",
    "structure_for_caching",
    "build_anthropic_cache_request",
    "check_cache_eligibility",
    "CacheEligibility",
    "parse_anthropic_usage",
    "CacheUsageReport",
    "HistoryCompactor",
    "compact_history",
    "CompactionResult",
    "compact_history_rolling",
    "RollingSummary",
    "RollingCompactionResult",
    "FoldJob",
    "run_fold_job",
    "estimate_fold_payback",
    "flatten_messages",
    "select_tools",
    "ToolSelection",
    "ToolSession",
    "build_anthropic_deferred_tools",
    "estimate_tool_tokens",
    "DEFERRED_TOOLS_SYSTEM_HINT",
    "optimize_chunks",
    "ChunkSelection",
    "AnthropicTokenCounter",
    "GeminiTokenCounter",
    "AnthropicSummarizer",
    "GeminiSummarizer",
    "SavingsLog",
    "summarize_savings",
    "SavingsSummary",
    "providers",
]
