# JAWA-1606: Case Overview Enrichment Pipeline — Fix Plan

> Revision 2 — 2026-06-03  
> Triggered by dry-run diagnostic from 2026-06-02 identifying 3 root-cause problems.  
> Status: plan updated, awaiting review approval before implementation.

---

## Problem Summary (from dry-run logs)

| # | Problem | Impact | Root Cause in Code |
|---|---------|--------|--------------------|
| 1 | HTTP 524 Cloudflare timeout (126s) | All large court orders fail format step | `_call_llm_opencode` does not set `stream: True` → proxy buffers full response → Cloudflare sees silence → kills connection |
| 2 | Format step prompt = 371k chars | Timeout + wasted tokens | `_discover_court_cases` returns untruncated court order texts; format prompt includes raw `court_order_texts` from NGM discovery without truncation |
| 3 | Model resolved through 3 aliases silently | Unknown capability, silent degradation | Response model field captured but never logged/asserted against requested model |

---

## Fix Plan

### Fix 1: Enable SSE streaming on LLM calls

**File:** `enrich_case_overview.py` → `_call_llm_opencode` (line 1313)

**Change:** Add `"stream": True` to the request body dict.

```python
# Before (line 1313-1321):
body = {
    "model": normalized_model,
    "max_tokens": 6000,
    "messages": [...],
    "temperature": 0.1,
}

# After:
body = {
    "model": normalized_model,
    "max_tokens": 6000,
    "messages": [...],
    "temperature": 0.1,
    "stream": True,
}
```

**Rationale:** `_read_sse_json` (line 401) already handles SSE delta accumulation. The proxy will start sending tokens immediately. Cloudflare resets its 100s gateway timeout on each byte received. The application-level 300s timeout still governs total wait time.

**Risk:** Low. Streaming is already handled by the response parser. The only change is requesting it from the server. If a proxy doesn't support streaming, the response is still valid SSE (it just arrives all at once).

**Test:** Existing tests mock `urllib.request.urlopen` and return pre-built responses. They'll still pass because `_read_sse_json` handles both streaming and non-streaming payloads.

---

### Fix 2: Truncate discovered court order texts in format prompt

**File:** `enrich_case_overview.py` → `_process_case` (lines 1188-1194)

**Change:** Truncate each `court_order_text` entry to max 5000 chars (matching the extraction step's truncation), and add a hard aggregate limit of 15000 chars for all court order texts combined.

```python
# Before (lines 1188-1194):
if discovery.get("court_order_texts"):
    fmt_context["court_order_texts"] = "\n\n---\n\n".join(
        f"Court Order {i+1}:\n{t}"
        for i, t in enumerate(discovery["court_order_texts"])
    )

# After:
_MAX_COURT_ORDER_TEXT_CHARS = 5000
_MAX_AGGREGATE_COURT_ORDER_CHARS = 15000
if discovery.get("court_order_texts"):
    truncated = []
    total = 0
    for i, t in enumerate(discovery["court_order_texts"]):
        chunk = t[:_MAX_COURT_ORDER_TEXT_CHARS]
        truncated.append(chunk)
        total += len(chunk)
        if total >= _MAX_AGGREGATE_COURT_ORDER_CHARS:
            break
    fmt_context["court_order_texts"] = "\n\n---\n\n".join(
        f"Court Order {i+1}:\n{chunk}"
        for i, chunk in enumerate(truncated)
    )
```

**Also:** Add truncation logging at the format step to surface when texts are trimmed:

```python
logger.info(
    "Case %s: step=format prompt_len=%d court_order_texts_total=%d truncated=%s",
    case.case_id,
    len(fmt_prompt),
    sum(len(t) for t in discovery.get("court_order_texts", [])),
    total < sum(len(t) for t in discovery.get("court_order_texts", [])),
)
```

**Rationale:** The extraction step already truncates court orders to 5000 chars (line 778). The discovery phase returns untruncated text because it's a different code path (`_convert_one_source` → `_convert_source_to_markdown`). The format step only needs enough court order text to include relevant verdict/quotation context, not the full document. The extraction JSON already contains all structured data.

**Additional consideration:** The `FORMATTING_USER_PROMPT` template (line 189) includes `{court_order_texts}` section. The format LLM is instructed to "Include court case references and verdict details from the court case metadata." The court case metadata from NGM (small JSON) provides the structured reference data. The raw court order texts are supplementary. Truncating to 5000 chars per entry preserves the ability to quote relevant passages without blowing up the prompt.

**Risk:** Low. The extraction step already captures all structured data. Truncated court orders may lose some trailing detail, but the format LLM has the full extracted JSON to work from.

**Tests to update:** 
- `TestDiscoverCourtCases.test_converts_matching_source_when_found` — verify returned texts are truncated
- New test: format prompt stays under 20k chars even with large discovery texts

---

### Fix 3: Log and assert model identity on every LLM response

**File:** `enrich_case_overview.py` → `_call_llm_opencode` (after line 1335) and `_call_llm_anthropic` (after line 1429)

**Change A — OpenCode path:** After `payload = _read_sse_json(resp)` at line 1335, extract and log the response model:

```python
payload = _read_sse_json(resp)
response_model = payload.get("model", "")
logger.info(
    "LLM opencode: response model=%s requested=%s match=%s",
    response_model,
    normalized_model,
    response_model == normalized_model or response_model == model,
)
if response_model and response_model != normalized_model and response_model != model:
    logger.warning(
        "LLM model mismatch: requested=%s normalized=%s response=%s",
        model,
        normalized_model,
        response_model,
    )
```

**Change B — Anthropic path:** After `response = client.messages.create(...)` at line 1429:

```python
response_model = getattr(response, "model", "")
logger.info(
    "LLM anthropic: response model=%s requested=%s match=%s",
    response_model,
    model,
    response_model == model,
)
```

**Rationale:** The SSE parser already captures `model` from streaming chunks (line 444-445). We just need to log it and compare. This is a zero-cost observability addition.

**Risk:** None. Read-only logging.

---

## Fix Order (Recommended by Diagnostic)

1. **Fix 2 first** — zero infra changes, reduces format prompt from 371k→~15k for most cases. May resolve the 524 for many cases immediately.
2. **Fix 1** — adds `stream: True`, closes the remaining 524 gap for genuinely large cases.
3. **Fix 3** — logging-only, no behavioral change.

## Verification

1. Run `enrich_case_overview --dry-run --case-id=<known-large-case>` and verify:
   - Format prompt length < 20k chars (was > 350k)
   - No HTTP 524 errors
   - Response model logged and matching
2. Run existing test suite: `cd /paperspace/tmp/code/JawafdehiAPI-JAWA-1606 && python -m pytest tests/commands/test_enrich_case_overview.py -v`
3. Run a 3-case integration test with `--dry-run --limit 3`

## Files Touched

- `cases/management/commands/enrich_case_overview.py` — all 3 fixes
- `tests/commands/test_enrich_case_overview.py` — truncation test for discovery texts

## Constitutional Notes

- Tier 3: safe repository writes to own feature branch `jawa-1606-case-overview`
- Worktree: `/paperspace/tmp/code/JawafdehiAPI-JAWA-1606`
- All fixes are within the existing command file, no new dependencies
