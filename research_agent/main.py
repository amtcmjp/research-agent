"""CLI entry: run the deep research agent over a customers JSON file.

Production entry point. One deep agent is built per batch and
reused across every customer. For each record the agent crawls
the public web for adverse / "bad" information, then a structured
scorer post-processes the agent's final report into a validated
{score, summary, sources[]} JSON printed to stdout. No files are
written to disk.

Usage:
    uv run python -m research_agent.main [CUSTOMERS_JSON]

Env (see research_agent.settings):
    BEDROCK_MODEL_ID             — required
    AWS_REGION                   — optional (default ap-northeast-1)
    TAVILY_API_KEY               — required
    BEDROCK_READ_TIMEOUT         — optional (default 300)
    RESEARCH_LOG_LEVEL           — optional (default INFO)
    RESEARCH_RECURSION_LIMIT     — optional (default 150)
    RESEARCH_HIGH_RISK_THRESHOLD — optional (default 40)
    RESEARCH_CUSTOMERS_PATH      — optional (default customers_coremobile.json)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from pydantic import ValidationError

from research_agent.agent import _build_model, build_agent
from research_agent.models import BadDataReport, CustomerInput, CustomerReport
from research_agent.prompts import BAD_DATA_CASE_PROMPT
from research_agent.settings import ResearchSettings

load_dotenv()

logger = logging.getLogger("research_agent.main")


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_bad_data_prompt(candidate: CustomerInput) -> str:
    """Render the per-case user message from the customer record."""
    return BAD_DATA_CASE_PROMPT.format(
        case_id=candidate.case_id,
        name=candidate.name,
        company_name=candidate.company_name,
        company_address=candidate.company_address,
        sns_url=candidate.sns_url or "(none)",
    )


SCORING_INSTRUCTIONS_TEMPLATE = """
You are the final structured-output step for an OSINT research
agent. You receive a candidate's identifiers plus a research
report (markdown with inline citations and a Sources section).

## Output language
Write `summary` and every `description` field in **Japanese**
(日本語). The `title` field may stay in the source's original
language. `url` is the raw URL.

## Task
Produce a BadDataReport with:
- score: integer 0-100 (overall adverse-info risk)
- summary: two or three neutral sentences in Japanese,
  under 700 characters
- sources: ONLY HIGH-RISK sources whose per-source score is
  >= {high_risk_threshold}. Each entry MUST include:
    * title       — clearest title of the source (original lang OK)
    * url         — direct URL (deduplicate across sources)
    * description — one or two neutral sentences IN JAPANESE
                    stating the adverse finding at that source
    * score       — per-source adverse-risk score 0-100

## What drives the score (SEVERITY, not just confirmation)
Score reflects HARM to anyone transacting with the applicant.
"Confirmed" alone does not justify a high score — a confirmed
clerical mistake is low; an unconfirmed criminal allegation
can still be medium. Calibrate by harm severity + intent.

## Severity calibration (anchor your per-source scores here)

Score 80-100 (HIGH RISK — final gate at or above threshold):
- Confirmed fraud, scam, theft, embezzlement by the person or
  company.
- Criminal indictment, conviction, or arrest of the person.
- Formal regulator sanction with material customer impact
  (業務停止命令, licence revocation, large monetary penalty).
- Court judgment finding fraud, breach of fiduciary duty, or
  willful misconduct.
- Confirmed bankruptcy / 民事再生 that would strand creditors.
- Mass data breach caused by negligence or intent, with
  sensitive customer data exposed to third parties.

Score 60-79 (SERIOUS):
- Multiple serious consumer complaints with regulator or press
  corroboration.
- Open lawsuits alleging fraud / financial harm, actively
  litigated.
- Repeat administrative warnings 業務改善命令 across years.
- Confirmed operational failure that caused material customer
  financial harm (e.g. prolonged service outage with proven
  loss).

Score 40-59 (MATERIAL):
- Single confirmed administrative notice or minor penalty.
- Credible allegations without resolution.
- Material governance / audit issues reported by reputable
  press.

Score 20-39 (MINOR — usually below the HIGH-RISK gate):
- Operational mistakes publicly disclosed and apologized for,
  with NO customer financial loss (e.g. accidental data
  deletion, system outage without data exfiltration).
- Isolated customer complaints with no corroboration.
- Negative reviews without regulator or press follow-up.

Score 0-19 (VERY MINOR):
- Opinion pieces, unverified reviews, forum rumours.
- Pure directory / profile pages with no adverse content.

## Per-source scoring modifiers
- Weight confirmed events and official findings higher than
  anonymous reviews or allegations.
- Reduce by HALF if identity match is weak (name alone is NOT
  identity).
- Exclude pure directory / profile pages (score 0).
- An operational mistake without customer financial harm,
  even when confirmed by reputable press, is MINOR (20-39);
  do not inflate it to HIGH-RISK just because the article is
  credible.

## Consistency rule (overall score vs per-source scores)
The overall `score` MUST be consistent with the surviving
per-source scores:
- If `sources` is empty, overall `score` must be 0-19.
- Otherwise overall `score` must be within +/-10 of the
  maximum per-source `score` that survived the threshold.
  Example: if the top surviving source scores 82, the
  overall score must be 72-92, not 42.

If severity calibration pushes every candidate source below
{high_risk_threshold}, the correct outcome is sources = [],
overall score in 0-19, and a short Japanese summary stating
that no high-risk adverse evidence was found (even if some
MINOR or MATERIAL items were observed).

## Discipline
- Only cite URLs that actually appear in the research report.
- Deduplicate URLs; keep the clearest title.
- Drop every source with score < {high_risk_threshold}; do
  NOT include MINOR or MATERIAL items in the final sources
  list even if they are confirmed.
- Do not label any person a criminal. Do not invent findings.
"""


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------


def _extract_final_report(result: dict[str, Any]) -> str:
    """Return the orchestrator's final chat message as plain text.

    The agent writes nothing to the virtual filesystem — its final
    chat message IS the deliverable. Walk the messages list from
    the end and return the first assistant message with real text.
    """
    messages = result.get("messages") or []
    for message in reversed(messages):
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if content is None:
            continue

        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
            continue

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(block))
            text = "\n".join(parts).strip()
            if text:
                return text

    return ""


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def _score_report(
    candidate: CustomerInput,
    final_report_md: str,
    settings: ResearchSettings,
) -> BadDataReport:
    scorer = _build_model().with_structured_output(BadDataReport)
    instructions = SCORING_INSTRUCTIONS_TEMPLATE.format(
        high_risk_threshold=settings.high_risk_threshold
    )

    prompt = (
        f"{instructions}\n\n"
        f"Case ID: {candidate.case_id}\n"
        f"Person: {candidate.name}\n"
        f"Company: {candidate.company_name}\n"
        f"Address: {candidate.company_address}\n\n"
        f"Research report:\n"
        f"{final_report_md or '(no final_report.md was produced)'}\n\n"
        "Produce the structured BadDataReport now. Keep only "
        f"sources with per-source score >= {settings.high_risk_threshold}."
    )

    try:
        return cast(BadDataReport, scorer.invoke(prompt))
    except ValidationError as exc:
        logger.warning("scorer validation failed, retrying once: %s", exc)
        retry_prompt = (
            f"{prompt}\n\n"
            "YOUR PREVIOUS RESPONSE FAILED VALIDATION:\n"
            f"{exc}\n\n"
            "Fix every reported field. Keep summary under 700 chars."
        )
        try:
            return cast(BadDataReport, scorer.invoke(retry_prompt))
        except Exception as retry_exc:
            logger.exception("scorer retry failed")
            return BadDataReport(
                score=0,
                summary=(
                    f"Scoring failed after retry: "
                    f"{type(retry_exc).__name__}: {retry_exc}"
                )[:700],
                sources=[],
            )
    except Exception as exc:
        logger.exception("scorer failed")
        return BadDataReport(
            score=0,
            summary=f"Scoring failed: {type(exc).__name__}: {exc}"[:700],
            sources=[],
        )


# ---------------------------------------------------------------------------
# Per-customer pipeline
# ---------------------------------------------------------------------------


def run_customer(
    candidate: CustomerInput,
    agent: Any,
    settings: ResearchSettings,
) -> CustomerReport:
    """Run one case against a reusable deep-agent instance."""
    logger.info(
        "case %s start: %s @ %s",
        candidate.case_id,
        candidate.name,
        candidate.company_name,
    )

    prompt = _build_bad_data_prompt(candidate)
    config = {
        "configurable": {"thread_id": candidate.case_id},
        "recursion_limit": settings.recursion_limit,
    }

    started = time.monotonic()
    agent_failed = False
    try:
        result = cast(
            dict[str, Any],
            agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config,
            ),
        )
    except Exception:
        logger.exception("agent invoke failed for %s", candidate.case_id)
        result = {"messages": []}
        agent_failed = True

    final_report = _extract_final_report(result)
    bad_data = _score_report(candidate, final_report, settings)

    # Belt-and-suspenders: drop sources below the configured threshold.
    kept_sources = [
        s for s in bad_data.sources if s.score >= settings.high_risk_threshold
    ]

    # Consistency clamp: overall score must align with the surviving
    # per-source scores. If nothing clears the threshold the overall
    # score belongs in 0-19; otherwise it must sit within +/-10 of
    # the maximum surviving source score.
    if not kept_sources:
        reconciled_score = min(bad_data.score, 19)
    else:
        max_src = max(s.score for s in kept_sources)
        lower, upper = max_src - 10, max_src + 10
        reconciled_score = min(100, max(0, min(upper, max(lower, bad_data.score))))

    if reconciled_score != bad_data.score:
        logger.info(
            "case %s overall score reconciled %d → %d (max_src=%d, kept=%d)",
            candidate.case_id,
            bad_data.score,
            reconciled_score,
            max((s.score for s in kept_sources), default=0),
            len(kept_sources),
        )

    bad_data = bad_data.model_copy(
        update={"sources": kept_sources, "score": reconciled_score}
    )

    duration_ms = int((time.monotonic() - started) * 1000)
    customer_report = CustomerReport(
        case_id=candidate.case_id,
        name=candidate.name,
        company_name=candidate.company_name,
        company_address=candidate.company_address,
        report=bad_data,
        duration_ms=duration_ms,
        agent_failed=agent_failed,
    )

    logger.info(
        "case %s done: score=%d/100 sources=%d duration=%dms failed=%s",
        candidate.case_id,
        bad_data.score,
        len(bad_data.sources),
        duration_ms,
        agent_failed,
    )
    return customer_report


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


def _load_candidates(path: Path) -> list[CustomerInput]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Customers JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise TypeError(f"Expected a list of customer records in {path}")

    candidates: list[CustomerInput] = []
    for idx, record in enumerate(data, start=1):
        if not isinstance(record, dict):
            logger.warning("record %d is not an object; skipping", idx)
            continue
        try:
            candidates.append(CustomerInput.from_record(record))
        except ValidationError as exc:
            logger.warning("record %d failed validation; skipping: %s", idx, exc)
    return candidates


def run_batch(
    customers_path: Path,
    settings: ResearchSettings | None = None,
) -> list[CustomerReport]:
    """Validate every customer record then run the batch against one agent."""
    settings = settings or ResearchSettings()
    candidates = _load_candidates(customers_path)
    if not candidates:
        logger.warning("no valid candidates in %s", customers_path)
        return []

    agent = build_agent()  # build once; reuse across every case
    results: list[CustomerReport] = []
    for idx, candidate in enumerate(candidates, start=1):
        logger.info("processing %d/%d: %s", idx, len(candidates), candidate.case_id)
        try:
            results.append(run_customer(candidate, agent, settings))
        except Exception:
            logger.exception("case %s failed; continuing with next", candidate.case_id)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def main() -> int:
    settings = ResearchSettings()
    _setup_logging(settings.log_level)

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.customers_path
    print(f"Loading customers from: {path}")

    try:
        results = run_batch(path, settings)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        logger.error("batch aborted: %s", exc)
        return 2

    print(f"\n=== Batch complete: {len(results)} cases ===")
    for r in results:
        status = "FAILED" if r.agent_failed else "ok"
        print(
            f"- {r.case_id} | {r.name} | score={r.report.score}/100 | "
            f"sources={len(r.report.sources)} | "
            f"{r.duration_ms}ms | {status}"
        )

    for r in results:
        print(f"\n--- {r.case_id} ---")
        print(json.dumps(r.model_dump(), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
