#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures, json, math, random, statistics, subprocess, sys, threading, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from tonst import HistoryCompactor, PromptParts, TonstClient, build_anthropic_cache_request
from tonst.trim import estimate_tokens

# A local-model call (redact_llm's LLMRedactor.redact(), local_model's
# LocalCompressor.compress(), or compactor's HistoryCompactor.summarize())
# times out at 8.0s by default. A real timeout fires within a few ms of
# that ceiling (confirmed directly against a live Ollama instance under
# contention -- see colab-benchmark-findings.md), so treating any stage
# whose reported ms lands within this margin of the ceiling as "likely
# timed out" turns the aggregate local_overhead_ms numbers into an actual
# diagnostic instead of just a latency figure with no failure breakdown.
LOCAL_MODEL_TIMEOUT_S = 8.0
NEAR_TIMEOUT_MS = (LOCAL_MODEL_TIMEOUT_S * 1000) - 250

def check_gpu_status():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True)
        print(f"[GPU DETECTED] {out.strip()}", flush=True)
    except Exception:
        print(
            "[WARNING] No NVIDIA GPU detected! Ollama is running on CPU, which is ~10x slower (0.1 it/s).\n"
            "          In Google Colab, go to Runtime -> Change runtime type -> T4 GPU.",
            flush=True,
        )

def wait_for_ollama_ready(model: str, max_wait_s: float = 240.0, poll_interval_s: float = 2.0) -> None:
    """
    Ollama's /api/tags responds as soon as the server process is up, but
    /api/generate can silently hang for tens of seconds after that while
    the model is loaded into GPU memory (or into RAM/CPU) and initialized
    -- on a freshly (re)started instance this took ~45s in real testing,
    with /api/generate raising a plain ReadTimeout on every attempt in
    between, not a slow-but-real response. Benchmarking before that
    finishes just measures Ollama's one-time startup cost, not the
    model's real steady-state per-call latency, and produces exactly the
    misleading timeout clustering this project chased for a long time
    before finding the real cause (see research/colab-benchmark-findings.md).
    This blocks until a real /api/generate call succeeds (or max_wait_s
    elapses), so every timing number the benchmark records reflects
    steady-state behavior only.

    IMPORTANT: the per-attempt request timeout below must stay well above
    a realistic cold-load time. A prior version used timeout=5, which is
    shorter than gemma2:2b's real cold-load+warmup time under some
    configs (e.g. OLLAMA_NUM_PARALLEL>1 with a larger context). Ollama
    treats an abandoned client connection as a reason to abort the
    in-progress model load server-side ("client connection closed before
    llama-server finished loading, aborting load" in its log) and restart
    a fresh llama-server process for the next attempt -- so a too-short
    timeout doesn't just fail once, it creates a livelock where the model
    can never finish loading because every poll cancels the previous
    attempt's progress. 60s per attempt gives real cold loads room to
    actually complete instead of being repeatedly aborted mid-flight.
    """
    import requests as _requests

    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": "Say hello in one word.", "stream": False}
    per_attempt_timeout_s = 60.0
    t_start = time.perf_counter()
    attempt = 0
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed > max_wait_s:
            print(
                f"[WARNING] Ollama still hasn't answered /api/generate after {max_wait_s:.0f}s -- "
                "proceeding anyway, expect the earliest iterations to be slow or time out.",
                flush=True,
            )
            return
        attempt += 1
        print(
            f"[WAIT] Ollama not ready yet ({elapsed:.1f}s elapsed, attempt {attempt}) -- "
            f"polling /api/generate with a {per_attempt_timeout_s:.0f}s timeout "
            "(long enough that a real cold load isn't cancelled mid-flight)...",
            flush=True,
        )
        try:
            _requests.post(url, json=payload, timeout=per_attempt_timeout_s)
            print(f"[READY] Ollama answered a real /api/generate call after {time.perf_counter() - t_start:.1f}s (attempt {attempt}).", flush=True)
            return
        except _requests.exceptions.RequestException as exc:
            print(
                f"[WAIT] attempt {attempt} did not get an answer ({type(exc).__name__}) -- "
                f"retrying in {poll_interval_s:.0f}s...",
                flush=True,
            )
            time.sleep(poll_interval_s)


INDUSTRY_CONFIGS = {
    "IT": {
        "system_role": "You are a Principal Cloud Infrastructure & SRE Support Assistant.",
        "companies": ["NimbusCloud", "KubeScale Systems", "Apex Datacenters", "VectorNet SRE"],
        "codenames": ["Project KubeShield", "Project ZeroTrust", "Operation MeshGuard"],
        "jargon_bloated": (
            "WARN [kube-apiserver] Etcd leader election latency exceeded threshold.\n"
            "WARN [kube-apiserver] Etcd leader election latency exceeded threshold.\n"
            "INFO [ingress-nginx] TLS handshake retry on upstream pod.\n"
            "INFO [ingress-nginx] TLS handshake retry on upstream pod.\n"
        ),
        "few_shot_examples": [
            "Example 1: Pod CrashLoopBackOff on node-04 -> Check OOMKilled exit code 137.",
            "Example 2: IAM Role AssumeRole AccessDenied -> Verify trust policy externalId.",
        ],
        "domain_context": "production Kubernetes cluster outage and IAM token rotation",
    },
    "Medical": {
        "system_role": "You are a HIPAA-Compliant Clinical Documentation & Triage Assistant.",
        "companies": ["Mercy General Health", "BioGenetics Lab", "Apex Clinical Trials", "Novus Oncology"],
        "codenames": ["Protocol ONCO-204", "Trial CARDIO-9", "Study NEURO-Vanguard"],
        "jargon_bloated": (
            "VITALS: BP 128/82 mmHg | HR 74 bpm | SpO2 98% RA | Temp 36.8 C.\n"
            "VITALS: BP 128/82 mmHg | HR 74 bpm | SpO2 98% RA | Temp 36.8 C.\n"
            "LABS: WBC 6.4 | Hgb 14.2 | Plt 240 | Cr 0.9 mg/dL | ALT 22 U/L.\n"
            "LABS: WBC 6.4 | Hgb 14.2 | Plt 240 | Cr 0.9 mg/dL | ALT 22 U/L.\n"
        ),
        "few_shot_examples": [
            "Example 1: Elevated troponin + ST elevation -> Immediate cardiology consult.",
            "Example 2: Post-op fever day 2 -> Evaluate for atelectasis vs wound infection.",
        ],
        "domain_context": "patient clinical intake note and adverse event billing audit",
    },
    "Electronics": {
        "system_role": "You are a Semiconductor Fabrication & PCB Yield Analysis Assistant.",
        "companies": ["SiliconFab Foundry", "NanoLitho Corp", "OmniChip Design", "QuantumWafer Inc"],
        "codenames": ["ASIC-Kronos-5nm", "Project Gallium-X", "WaferLot-EUV-88"],
        "jargon_bloated": (
            "DRC CHECK: Metal-3 minimum spacing rule violation at reticle (142.4, 88.1).\n"
            "DRC CHECK: Metal-3 minimum spacing rule violation at reticle (142.4, 88.1).\n"
            "TEST LOG: JTAG boundary scan IDCODE mismatch on BGA pin C14.\n"
            "TEST LOG: JTAG boundary scan IDCODE mismatch on BGA pin C14.\n"
        ),
        "few_shot_examples": [
            "Example 1: SerDes eye diagram closure at 112G -> Check PCB trace insertion loss.",
            "Example 2: LDO voltage droop under 4A load -> Inspect decoupling capacitor ESR.",
        ],
        "domain_context": "3nm ASIC tape-out yield excursion and RMA replacement billing",
    },
    "Space": {
        "system_role": "You are an Orbital Flight Dynamics & Satellite Telemetry Assistant.",
        "companies": ["OrbitalVanguard Aero", "AetherSpace Launch", "Helios Propulsion", "StarLinkage Telemetry"],
        "codenames": ["Mission Artemis-IX", "Payload Sentinel-7B", "Vehicle CryoStage-3"],
        "jargon_bloated": (
            "TELEMETRY [S-BAND]: Attitude reaction wheel #2 tach loop jitter > 0.04 rad/s.\n"
            "TELEMETRY [S-BAND]: Attitude reaction wheel #2 tach loop jitter > 0.04 rad/s.\n"
            "PROPULSION [LOX/CH4]: Turbopump inlet cavitation margin nominal at 104% thrust.\n"
            "PROPULSION [LOX/CH4]: Turbopump inlet cavitation margin nominal at 104% thrust.\n"
        ),
        "few_shot_examples": [
            "Example 1: Star tracker blinding near sun exclusion angle -> Switch to IMU gyro propagation.",
            "Example 2: Downlink Ka-band Eb/N0 drop -> Command ground station polarization adjust.",
        ],
        "domain_context": "LEO satellite constellation telemetry anomaly and transponder lease",
    },
    "Finance": {
        "system_role": "You are an Institutional Treasury & Fraud Operations Assistant.",
        "companies": ["Apex Clearing House", "Vanguard Citadel Custody", "Meridian Alpha Desk", "SilverOak FX"],
        "codenames": ["Strategy Alpha-Zero", "Book Liquidity-Prime", "Ledger Settlement-X"],
        "jargon_bloated": (
            "SWIFT MT103 ACK: Correspondent Nostro account reconciliation pending.\n"
            "SWIFT MT103 ACK: Correspondent Nostro account reconciliation pending.\n"
            "FIX 4.4 EXECUTION REPORT: ClOrdID=883921 ExecType=PARTIAL_FILL CumQty=4500.\n"
            "FIX 4.4 EXECUTION REPORT: ClOrdID=883921 ExecType=PARTIAL_FILL CumQty=4500.\n"
        ),
        "few_shot_examples": [
            "Example 1: Duplicate wire debit flag -> Hold settlement and verify beneficiary ABA.",
            "Example 2: Margin utilization > 92% -> Issue automated intraday collateral call.",
        ],
        "domain_context": "high-value wire transfer dispute and institutional card authorization",
    },
    "Legal": {
        "system_role": "You are a Corporate M&A & Regulatory Compliance Assistant.",
        "companies": ["Sterling & Vance LLP", "Global Antitrust Counsel", "OmniCorp Holdings", "Pinnacle IP Trust"],
        "codenames": ["Project Bluebird M&A", "Matter Docket-9921", "Settlement Horizon"],
        "jargon_bloated": (
            "CLAUSE 14.2: Indemnification cap shall not exceed 100% of Escrow Amount.\n"
            "CLAUSE 14.2: Indemnification cap shall not exceed 100% of Escrow Amount.\n"
            "HSR FILING NOTE: Second Request document production privilege log verified.\n"
            "HSR FILING NOTE: Second Request document production privilege log verified.\n"
        ),
        "few_shot_examples": [
            "Example 1: Change-of-control termination trigger -> Require 60-day prior written notice.",
            "Example 2: GDPR DPA cross-border transfer -> Attach Standard Contractual Clauses.",
        ],
        "domain_context": "cross-border acquisition escrow release and retainer fee billing",
    },
}

FIRST_NAMES = ["Priya", "Marcus", "Elena", "David", "Aisha", "Carlos", "Mei", "Liam"]
LAST_NAMES = ["Malhotra", "Vance", "Rostova", "Kim", "Al-Mansoor", "Mendez", "Chen", "O'Connor"]
DOMAINS = ["example.com", "corp-test.org", "client-mail.net", "enterprise-demo.io"]

def generate_synthetic_pii(rng: random.Random, industry: str) -> dict:
    cfg = INDUSTRY_CONFIGS[industry]
    first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
    return {
        "full_name": f"{first} {last}",
        "email": f"{first.lower()}.{last.lower()}{rng.randint(10, 99)}@{rng.choice(DOMAINS)}",
        "card": f"4111 {rng.randint(1000, 9999)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}",
        "phone": f"+1 {rng.randint(200, 999)}-555-{rng.randint(1000, 9999)}",
        "ssn": f"{rng.randint(100, 899)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}",
        "ip": f"192.168.{rng.randint(1, 254)}.{rng.randint(1, 254)}",
        "company": rng.choice(cfg["companies"]),
        "codename": rng.choice(cfg["codenames"]),
    }

def generate_benchmark_case(rng: random.Random, industry: str, mode: str):
    cfg = INDUSTRY_CONFIGS[industry]
    pii = generate_synthetic_pii(rng, industry)
    if mode == "supervised":
        shape = rng.choice(["supervised_few_shot", "supervised_extraction"])
        stable_prefix = (
            f"{cfg['system_role']}\n"
            f"Domain Guidelines for {industry}: Always protect sensitive PII and billing tokens.\n"
            + "\n".join(cfg["few_shot_examples"])
        )
        variable_question = (
            f"[SUPERVISED TASK - {industry}] Lead engineer {pii['full_name']} at {pii['company']} "
            f"(email: {pii['email']}, phone: {pii['phone']}, SSN: {pii['ssn']}) submitted an urgent "
            f"ticket regarding {pii['codename']} connected to node IP {pii['ip']} and corporate card "
            f"{pii['card']}. Classify priority and draft a 1-sentence response."
        )
        parts = PromptParts(system=cfg["system_role"], stable_blocks=[stable_prefix], variable=variable_question)
        return shape, f"{stable_prefix}\n\n{variable_question}", pii, parts, None
    else:
        shape = rng.choice(["unsupervised_bloated_logs", "unsupervised_verbose_dump", "unsupervised_multi_turn"])
        if shape == "unsupervised_bloated_logs":
            flat_prompt = (
                f"{cfg['system_role']}\n{cfg['system_role']}\n{cfg['jargon_bloated']}\n"
                f"UNSTRUCTURED INCIDENT LOG ({industry}):\n"
                f"Contact: {pii['full_name']} ({pii['company']}) | Email: {pii['email']} | "
                f"Tel: {pii['phone']} | Host IP: {pii['ip']} | Billing Card: {pii['card']} | Project: {pii['codename']}.\n"
                f"Contact: {pii['full_name']} ({pii['company']}) | Email: {pii['email']} | "
                f"Tel: {pii['phone']} | Host IP: {pii['ip']} | Billing Card: {pii['card']} | Project: {pii['codename']}.\n\n\n\n"
                f"Please analyze the root cause of this {cfg['domain_context']}."
            )
            return shape, flat_prompt, pii, None, None
        elif shape == "unsupervised_verbose_dump":
            flat_prompt = (
                f"Hello team, I am dumping our raw notes from the {industry} war-room session "
                f"for {pii['codename']} at {pii['company']}. Basically, {pii['full_name']} "
                f"(reachable at {pii['email']} or {pii['phone']}) noticed that whenever telemetry "
                f"packets originate from IP {pii['ip']}, our automated billing gateway charges card "
                f"{pii['card']} repeatedly due to a retry storm during {cfg['domain_context']}. "
                f"{cfg['jargon_bloated']} We need to know what immediate action {pii['full_name']} should take."
            )
            return shape, flat_prompt, pii, None, None
        else:
            messages = [{"role": "system", "content": cfg["system_role"]}]
            for turn in range(6):
                if turn % 2 == 0:
                    messages.append({
                        "role": "user",
                        "content": f"[Turn {turn} - {industry}] {pii['full_name']} ({pii['email']}, {pii['phone']}) at {pii['company']} on {pii['codename']}: host {pii['ip']} and card {pii['card']} impacted."
                    })
                else:
                    messages.append({"role": "assistant", "content": f"[Turn {turn}] Logged diagnostics for {pii['codename']} ({industry})."})
            messages.append({"role": "user", "content": f"Summarize next steps for {pii['email']} in one sentence."})
            return shape, "\n\n".join(f"{m['role']}: {m['content']}" for m in messages), pii, None, messages

def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    return s[int(k)] if f == c else s[int(f)] * (c - k) + s[int(c)] * (k - f)

@dataclass
class SliceMetrics:
    iterations: int = 0
    original_tokens: int = 0
    sent_tokens: int = 0
    redacted_fields: int = 0
    regex_pii_ground_truth_total: int = 0
    regex_pii_redacted_hits: int = 0
    regex_pii_leaks: int = 0
    # NEW: the free-text fields (full_name/company/codename) regex can
    # never catch -- only LLMRedactor's enhanced pass (or a GLiNER-based
    # backend) can. Tracked separately from the regex_pii_* counters
    # above (which now also include ssn, previously computed by regex.py
    # but never actually checked here) so a report can distinguish
    # "structured PII leaked" from "free-text PII leaked" rather than
    # blending them into one number that only ever reflected the
    # regex layer.
    free_text_pii_ground_truth_total: int = 0
    free_text_pii_redacted_hits: int = 0
    free_text_pii_leaks: int = 0
    round_trip_restoration_failures: int = 0
    local_overhead_ms: List[float] = field(default_factory=list)
    # NEW: per-stage near-timeout counters. report.redaction_ms /
    # compression_ms / compaction_ms are already computed by the library
    # (see client.py's OptimizationReport) -- this just flags any stage
    # whose elapsed time landed within NEAR_TIMEOUT_MS of the 8.0s ceiling,
    # which is the same signature _diagnose_local_model_latency.py uses to
    # tell "genuinely slow" apart from "never got serviced at all". Without
    # this, aggregate local_overhead_ms alone can't distinguish a run where
    # every call is contention-limited from one where every call is just
    # slow-but-real.
    redaction_near_timeout: int = 0
    compression_near_timeout: int = 0
    compaction_near_timeout: int = 0
    # NEW: per-stage timing breakdown. OptimizationReport already computes
    # each of these separately per call (client.py) -- previously only their
    # SUM (local_overhead_ms) was kept, which made it impossible to see
    # which stage actually dominates a request's timeline without a
    # separate one-off diagnostic script. Tracking them here means every
    # real benchmark run answers "where did the time go" directly, across
    # the full iteration count and case mix, not a hand-picked sample of 4-5.
    redaction_ms_list: List[float] = field(default_factory=list)
    trim_ms_list: List[float] = field(default_factory=list)
    compression_ms_list: List[float] = field(default_factory=list)
    compaction_ms_list: List[float] = field(default_factory=list)
    call_ms_list: List[float] = field(default_factory=list)
    total_ms_list: List[float] = field(default_factory=list)

    def to_summary_dict(self) -> dict:
        saved = max(0, self.original_tokens - self.sent_tokens)
        pct = round(100.0 * saved / self.original_tokens, 2) if self.original_tokens else 0.0
        recall = round(100.0 * self.regex_pii_redacted_hits / self.regex_pii_ground_truth_total, 2) if self.regex_pii_ground_truth_total else 100.0
        free_text_recall = round(100.0 * self.free_text_pii_redacted_hits / self.free_text_pii_ground_truth_total, 2) if self.free_text_pii_ground_truth_total else 100.0
        return {
            "iterations": self.iterations,
            "original_tokens": self.original_tokens,
            "sent_tokens": self.sent_tokens,
            "tokens_saved": saved,
            "percent_saved": pct,
            "avg_redacted_fields_per_call": round(self.redacted_fields / self.iterations, 2) if self.iterations else 0.0,
            "supervised_pii_recall_percent": recall,
            "pii_leak_count": self.regex_pii_leaks,
            # NEW: full_name/company/codename recall/leak count -- only
            # meaningful when use_enhanced_redaction was on for this run;
            # with it off these fields are EXPECTED to leak (the feature
            # is simply disabled), so a low free_text_pii_recall_percent
            # in that case is not a bug, just an untested-by-design path.
            "free_text_pii_recall_percent": free_text_recall,
            "free_text_pii_leak_count": self.free_text_pii_leaks,
            "round_trip_restoration_failures": self.round_trip_restoration_failures,
            "latency_overhead_ms": {
                "p50": round(percentile(self.local_overhead_ms, 50), 3),
                "p90": round(percentile(self.local_overhead_ms, 90), 3),
                "p95": round(percentile(self.local_overhead_ms, 95), 3),
                "p99": round(percentile(self.local_overhead_ms, 99), 3),
                "mean": round(statistics.mean(self.local_overhead_ms), 3) if self.local_overhead_ms else 0.0,
            },
            # NEW: how many of this slice's calls landed within 250ms of the
            # 8.0s local-model timeout on each stage -- direct evidence of
            # contention/queuing rather than genuine per-call cost, whenever
            # this is a large fraction of `iterations`.
            "near_timeout_counts": {
                "redaction": self.redaction_near_timeout,
                "compression": self.compression_near_timeout,
                "compaction": self.compaction_near_timeout,
            },
            # NEW: per-stage timeline. mean+p50 for each of the 5 sequential
            # steps a request actually goes through (see client.py's docstring
            # for the order: redact -> trim -> compress -> the real call ->
            # restore; compaction sits before all of that, only for
            # query_messages()). These are only non-zero for stages this run
            # actually enabled/exercised (e.g. compression_ms is ~0 unless
            # --use-local-compression or --enable-local-llm was passed).
            "stage_latency_ms": {
                stage: {
                    "mean": round(statistics.mean(values), 3) if values else 0.0,
                    "p50": round(percentile(values, 50), 3),
                    "p90": round(percentile(values, 90), 3),
                }
                for stage, values in (
                    ("redaction", self.redaction_ms_list),
                    ("trim", self.trim_ms_list),
                    ("compression", self.compression_ms_list),
                    ("compaction", self.compaction_ms_list),
                    ("call", self.call_ms_list),
                    ("total", self.total_ms_list),
                )
            },
        }

def run_benchmark(
    iterations: int,
    local_model: str,
    use_enhanced_redaction: bool,
    use_local_compression: bool,
    use_history_compaction: bool,
    workers: int = 4,
    seed: int = 42,
    redaction_backend: Optional[str] = None,
    gliner_model: str = "urchade/gliner_medium-v2.1",
    compression_model: Optional[str] = None,
    compaction_model: Optional[str] = None,
) -> dict:
    check_gpu_status()
    # Explicit redaction_backend wins; otherwise infer from the older
    # boolean for backward compatibility -- same rule TonstClient uses.
    resolved_redaction_backend = redaction_backend if redaction_backend is not None else ("ollama" if use_enhanced_redaction else "regex")
    # Warm up every DISTINCT Ollama model this run will actually call.
    # "gliner" needs no Ollama warmup for redaction at all, and a
    # compression/compaction override (e.g. gemma3:1b while redaction
    # uses gliner) must be warmed under ITS OWN name, not local_model's
    # -- otherwise that model's first real call eats a cold-load penalty
    # the benchmark would misattribute as steady-state latency.
    models_to_warm = set()
    if resolved_redaction_backend == "ollama":
        models_to_warm.add(local_model)
    if use_local_compression:
        models_to_warm.add(compression_model or local_model)
    if use_history_compaction:
        models_to_warm.add(compaction_model or local_model)
    for m in models_to_warm:
        wait_for_ollama_ready(m)
    if resolved_redaction_backend == "gliner":
        # Same reasoning as wait_for_ollama_ready(), for a different cold-
        # start cost: GLiNER's model load happens in-process (no server
        # to pre-warm), and the module-level cache in gliner_redact.py
        # means whichever iteration runs first pays that multi-second
        # load -- found 2026-09-13 via a live run where one supervised/IT
        # case alone accounted for a 6.2s p90 despite every other case
        # landing at 175-500ms. Loading it here, before any timed
        # iteration starts, keeps that one-time cost out of the
        # steady-state numbers entirely.
        print("[GLINER] warming up model (one-time load)...", flush=True)
        from tonst.gliner_redact import GlinerRedactor
        GlinerRedactor(model=gliner_model).is_available()
        print("[GLINER] ready.", flush=True)
    industries, modes = list(INDUSTRY_CONFIGS.keys()), ["supervised", "unsupervised"]
    overall_slice = SliceMetrics()
    by_mode = {m: SliceMetrics() for m in modes}
    by_industry = {ind: SliceMetrics() for ind in industries}
    by_industry_and_mode = {f"{ind}__{m}": SliceMetrics() for ind in industries for m in modes}
    lock = threading.Lock()
    t_start = time.perf_counter()

    def run_single_iteration(idx: int):
        local_rng = random.Random(seed + idx)
        industry = industries[idx % len(industries)]
        mode = modes[(idx // len(industries)) % len(modes)]
        shape, flat_prompt, pii, parts, messages = generate_benchmark_case(local_rng, industry, mode)
        thread_received: List[str] = []

        def mock_paid_api(trimmed_prompt: str) -> str:
            thread_received.append(trimmed_prompt)
            tokens = [w for w in trimmed_prompt.split() if w.startswith("[[") and "]]" in w]
            return f"Processed request successfully. Reference tokens: {' '.join(tokens[:4])}"

        # messages-path cases historically forced the backend down to
        # "regex" whenever ANY enhanced backend was selected. That rule
        # predates GLiNER: per research/colab-benchmark-findings.md, its
        # real purpose was avoiding TWO independent slow Ollama calls
        # with their own 8s timeouts stacking on the same multi-turn
        # iteration (LLMRedactor.redact() per message + HistoryCompactor
        # .summarize() for the dropped turns) -- a genuine Ollama-
        # specific timeout/latency risk. GLiNER has no such risk (one
        # in-process ~200-400ms call, no server round trip, no timeout
        # to stack), so forcing it down to regex-only here was just
        # porting the old ollama-specific workaround too broadly -- it
        # silently zeroed out GLiNER's free-text recall on ~1/6 of every
        # benchmark run for a reason that never applied to it (found
        # 2026-09-13 while explaining a 70-73% vs ~89% in-pipeline
        # recall gap -- see gliner-sanity-check-findings.md). Only
        # "ollama" still gets the forced-regex fallback here; "gliner"
        # and "none" pass through untouched.
        if messages is not None and resolved_redaction_backend == "ollama":
            effective_redaction_backend = "regex"
        else:
            effective_redaction_backend = resolved_redaction_backend
        client = TonstClient(
            call_fn=mock_paid_api,
            use_local_compression=use_local_compression,
            use_history_compaction=use_history_compaction,
            local_model=local_model,
            compaction_token_threshold=100,
            redaction_backend=effective_redaction_backend,
            gliner_model=gliner_model,
            compression_model=compression_model,
            compaction_model=compaction_model,
        )

        if messages is not None:
            response, report = client.query_messages(messages, keep_last_n=3)
        elif parts is not None:
            response, report = client.query_structured(parts)
        else:
            response, report = client.query(flat_prompt)

        sent_prompt = thread_received[-1]
        gt_keys = ("email", "card", "phone", "ip", "ssn")
        gt_hits = sum(1 for k in gt_keys if pii[k] not in sent_prompt)
        leaks = len(gt_keys) - gt_hits
        free_text_keys = ("full_name", "company", "codename")
        free_text_hits = sum(1 for k in free_text_keys if pii[k] not in sent_prompt)
        free_text_leaks = len(free_text_keys) - free_text_hits
        rt_fail = 1 if ("[[" in response and "]]" in response) else 0

        redaction_timed_out = report.redaction_ms >= NEAR_TIMEOUT_MS
        compression_timed_out = report.compression_ms >= NEAR_TIMEOUT_MS
        compaction_timed_out = report.compaction_ms >= NEAR_TIMEOUT_MS

        with lock:
            for target in (overall_slice, by_mode[mode], by_industry[industry], by_industry_and_mode[f"{industry}__{mode}"]):
                target.iterations += 1
                target.original_tokens += report.original_tokens
                target.sent_tokens += report.sent_tokens
                target.redacted_fields += report.redacted_fields
                target.regex_pii_ground_truth_total += len(gt_keys)
                target.regex_pii_redacted_hits += gt_hits
                target.regex_pii_leaks += leaks
                target.free_text_pii_ground_truth_total += len(free_text_keys)
                target.free_text_pii_redacted_hits += free_text_hits
                target.free_text_pii_leaks += free_text_leaks
                target.round_trip_restoration_failures += rt_fail
                target.local_overhead_ms.append(report.local_overhead_ms)
                target.redaction_ms_list.append(report.redaction_ms)
                target.trim_ms_list.append(report.trim_ms)
                target.compression_ms_list.append(report.compression_ms)
                target.compaction_ms_list.append(report.compaction_ms)
                target.call_ms_list.append(report.call_ms)
                target.total_ms_list.append(report.total_ms)
                if redaction_timed_out:
                    target.redaction_near_timeout += 1
                if compression_timed_out:
                    target.compression_near_timeout += 1
                if compaction_timed_out:
                    target.compaction_near_timeout += 1

    pbar = tqdm(total=iterations, desc="Running tonst benchmark", unit="iter") if tqdm is not None else None
    completed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_single_iteration, i) for i in range(iterations)]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()
            completed_count += 1
            if pbar is not None:
                pbar.update(1)
                if completed_count % 5 == 0:
                    saved = max(0, overall_slice.original_tokens - overall_slice.sent_tokens)
                    pct = (100.0 * saved / overall_slice.original_tokens) if overall_slice.original_tokens else 0.0
                    pbar.set_postfix({"saved%": f"{pct:.1f}%", "leaks": overall_slice.regex_pii_leaks})
            elif completed_count % 5 == 0 or completed_count == iterations:
                elapsed = time.perf_counter() - t_start
                saved = max(0, overall_slice.original_tokens - overall_slice.sent_tokens)
                pct = (100.0 * saved / overall_slice.original_tokens) if overall_slice.original_tokens else 0.0
                print(f"[{completed_count}/{iterations}] {(completed_count / elapsed):.1f} it/s | Tokens saved: {pct:.1f}%", flush=True)

    if pbar is not None:
        pbar.close()

    total_wall_sec = time.perf_counter() - t_start
    overall_dict = overall_slice.to_summary_dict()
    overall_dict["estimated_cost_saved_usd"] = round(((overall_slice.original_tokens - overall_slice.sent_tokens) / 1_000_000) * 3.00, 6)
    return {
        "overall_results": overall_dict,
        "by_paradigm": {m: s.to_summary_dict() for m, s in by_mode.items()},
        "by_industry": {ind: s.to_summary_dict() for ind, s in by_industry.items()},
        "by_industry_and_paradigm": {k: s.to_summary_dict() for k, s in by_industry_and_mode.items()},
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=360)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", type=str, default="gemma2:2b")
    parser.add_argument("--enable-local-llm", action="store_true", help="Shorthand for enabling all three local-model flags below at once.")
    parser.add_argument("--use-enhanced-redaction", action="store_true", help="Isolate just LLMRedactor (free-text PII catch). Superseded by --redaction-backend if that's also given.")
    parser.add_argument("--use-local-compression", action="store_true", help="Isolate just LocalCompressor.")
    parser.add_argument("--use-history-compaction", action="store_true", help="Isolate just HistoryCompactor (only affects unsupervised_multi_turn iterations).")
    parser.add_argument("--redaction-backend", type=str, choices=["ollama", "gliner", "regex", "none"], default=None, help="What catches free-text PII beyond regex. 'ollama' = local generative model via redact_llm.py (same as --use-enhanced-redaction). 'gliner' = extractive NER model, no GPU/Ollama needed for this step (see research/gliner-sanity-check-findings.md). 'regex' = structured PII only. 'none' = skip PII redaction entirely. Omit to fall back to --use-enhanced-redaction/--enable-local-llm.")
    parser.add_argument("--gliner-model", type=str, default="urchade/gliner_medium-v2.1", help="Only used when --redaction-backend gliner.")
    parser.add_argument("--compression-model", type=str, default=None, help="Ollama model for the compression step; defaults to --model if not given.")
    parser.add_argument("--compaction-model", type=str, default=None, help="Ollama model for history compaction; defaults to --model if not given.")
    parser.add_argument("--output", type=str, default="report_multi_industry.json")
    args = parser.parse_args()
    use_redaction = args.enable_local_llm or args.use_enhanced_redaction
    use_compression = args.enable_local_llm or args.use_local_compression
    use_compaction = args.enable_local_llm or args.use_history_compaction
    report = run_benchmark(
        args.iterations, args.model, use_redaction, use_compression, use_compaction,
        workers=args.workers,
        redaction_backend=args.redaction_backend,
        gliner_model=args.gliner_model,
        compression_model=args.compression_model,
        compaction_model=args.compaction_model,
    )
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["overall_results"], indent=2), flush=True)
