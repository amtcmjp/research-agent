"""Deep-agent OSINT pipeline.

For each case we build:

  Main orchestrator (deepagents.create_deep_agent)
    ├─ built-in tools:   write_todos, write_file, read_file, edit_file, task, …
    ├─ research tools:   tavily_search, apify_social_search,
    │                    serpapi_image_search, compare_face_to_reference
    └─ subagents:
         identity_verifier  (serpapi + rekognition + tavily)
         web_researcher     (tavily)
         social_researcher  (apify)

The orchestrator plans with write_todos, delegates to the three
subagents via the built-in `task` tool, aggregates findings into
`/report/findings.md` using the built-in filesystem, and stops.

A separate structured scorer then reads /report/findings.md and
emits a Pydantic OsintReport (0-100 risk score + findings).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, cast

from botocore.config import Config as BotoConfig
from deepagents import SubAgent, create_deep_agent
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from pydantic import ValidationError

from osint_agent.config import AUDIT_DIR
from osint_agent.instructions import (
    IDENTITY_VERIFIER_INSTRUCTIONS,
    ORCHESTRATOR_INSTRUCTIONS,
    SCORING_INSTRUCTIONS,
    SOCIAL_RESEARCHER_INSTRUCTIONS,
    WEB_RESEARCHER_INSTRUCTIONS,
)
from osint_agent.models import (
    CandidateInput,
    EvidenceEntry,
    FaceVerification,
    OsintCaseRecord,
    OsintReport,
    Recommendation,
)
from osint_agent.tools import Runtime, build_tools, make_runtime
from shared.rekognition import detect_face as rekognition_detect

load_dotenv()

logger = logging.getLogger("osint_agent.deep_pipeline")

RECURSION_LIMIT = int(os.getenv("OSINT_DEEP_RECURSION_LIMIT", "150"))
FINDINGS_PATH = "/report/findings.md"


# ---------------------------------------------------------------------------
# Pre-flight face detection (deterministic, before the agent runs)
# ---------------------------------------------------------------------------


def _preflight_face_verification(id_front_url: str) -> FaceVerification:
    if not id_front_url.strip():
        return FaceVerification(status="skipped", error="No id_front_url supplied.")

    detection = rekognition_detect(id_front_url)
    if detection["error"]:
        return FaceVerification(status="error", error=detection["error"])

    if not detection["detected"]:
        return FaceVerification(
            status="no_face",
            id_face_detected=False,
            id_face_confidence=detection["confidence"],
            id_face_quality=detection["quality_label"],
            sunglasses=detection["sunglasses"],
        )

    return FaceVerification(
        status="face_only",
        id_face_detected=True,
        id_face_confidence=detection["confidence"],
        id_face_quality=detection["quality_label"],
        sunglasses=detection["sunglasses"],
    )


def _merge_face_comparisons(
    base: FaceVerification,
    runtime: Runtime,
) -> FaceVerification:
    """Fold in any CompareFaces results the agent collected."""
    comparisons = list(base.comparisons) + list(runtime["comparisons"])
    if not comparisons:
        return base

    any_match = any(c.matched for c in comparisons)
    if any_match:
        status = "verified"
    elif base.status in {"no_face", "error", "skipped"}:
        status = base.status
    else:
        status = "mismatch"

    return base.model_copy(update={"status": status, "comparisons": comparisons})


# ---------------------------------------------------------------------------
# Model + agent build
# ---------------------------------------------------------------------------


def _build_model() -> ChatBedrockConverse:
    boto_config = BotoConfig(
        read_timeout=int(os.getenv("BEDROCK_READ_TIMEOUT", "300")),
        connect_timeout=int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "15")),
        retries={"max_attempts": 5, "mode": "adaptive"},
    )
    return ChatBedrockConverse(
        model=os.environ["BEDROCK_MODEL_ID"],
        region_name=os.getenv("AWS_REGION", "ap-northeast-1"),
        temperature=0,
        config=boto_config,
    )


def _build_subagents(tools: dict[str, Any]) -> list[SubAgent]:
    return [
        {
            "name": "identity_verifier",
            "description": (
                "Verify the candidate's identity. Detects faces on the "
                "ID photo, searches Google Images for the person, and "
                "compares candidate photos against the ID via AWS "
                "Rekognition. Confirms whether sites hosting matched "
                "images are actually about this candidate. Use when "
                "id_front_url is present."
            ),
            "system_prompt": IDENTITY_VERIFIER_INSTRUCTIONS,
            "tools": [
                tools["serpapi_image_search"],
                tools["compare_face_to_reference"],
                tools["tavily_search"],
            ],
        },
        {
            "name": "web_researcher",
            "description": (
                "Public-web OSINT via Tavily: news, legal records, "
                "regulators, insolvency, reviews, scam reports. "
                "Delegate one investigation topic per call."
            ),
            "system_prompt": WEB_RESEARCHER_INSTRUCTIONS,
            "tools": [tools["tavily_search"]],
        },
        {
            "name": "social_researcher",
            "description": (
                "Social-media OSINT via Apify: Instagram, Twitter, "
                "Facebook, LinkedIn, TikTok. Finds impersonation / "
                "scam / adverse-post signals. Delegate one topic per "
                "call."
            ),
            "system_prompt": SOCIAL_RESEARCHER_INSTRUCTIONS,
            "tools": [tools["apify_social_search"]],
        },
    ]


def _build_deep_agent(candidate: CandidateInput, runtime: Runtime) -> Any:
    model = _build_model()
    tools = build_tools(runtime)

    agent = create_deep_agent(
        model=model,
        tools=list(tools.values()),
        system_prompt=ORCHESTRATOR_INSTRUCTIONS,
        subagents=_build_subagents(tools),
    )
    return agent


def _initial_message(candidate: CandidateInput) -> str:
    return (
        f"Case ID: {candidate.case_id}\n"
        f"Name: {candidate.name}\n"
        f"Company: {candidate.company_name}\n"
        f"Company address: {candidate.company_address}\n"
        f"ID photo URL: {candidate.id_front_url or '(none)'}\n"
        f"SNS URL: {candidate.sns_url or '(none)'}\n\n"
        "Investigate this candidate for adverse public information "
        "and reused / inauthentic ID photos. Plan with write_todos, "
        "delegate to identity_verifier / web_researcher / "
        "social_researcher, append their findings to "
        "/evidence/notes.md as they arrive, and write a final "
        f"aggregated report to {FINDINGS_PATH} before stopping."
    )


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------


def _extract_file(result: dict[str, Any], path: str) -> str:
    """Pull a file written by the agent out of state.

    deepagents' default filesystem keeps written files in state under
    the `files` key as a mapping of path → content (plain string or
    structured dict depending on backend version).
    """
    files = result.get("files") or {}
    entry = files.get(path)
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        content = entry.get("content") or entry.get("text") or ""
        return str(content)
    return str(entry)


def _collect_findings_markdown(result: dict[str, Any]) -> str:
    """Prefer /report/findings.md; fall back to concatenating any files."""
    primary = _extract_file(result, FINDINGS_PATH)
    if primary.strip():
        return primary

    files = result.get("files") or {}
    if not files:
        return ""
    blocks: list[str] = []
    for path, entry in files.items():
        content = entry if isinstance(entry, str) else (
            entry.get("content") if isinstance(entry, dict) else str(entry)
        )
        if content:
            blocks.append(f"# {path}\n{content}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Scoring (structured output post-process)
# ---------------------------------------------------------------------------


def _score_report(
    candidate: CandidateInput,
    face_verification: FaceVerification,
    findings_markdown: str,
) -> OsintReport:
    model = _build_model()
    scorer = model.with_structured_output(OsintReport)

    face_block = _format_face_verification(face_verification)

    prompt = (
        f"{SCORING_INSTRUCTIONS}\n\n"
        f"Case ID: {candidate.case_id}\n"
        f"Target name: {candidate.name}\n"
        f"Target company: {candidate.company_name}\n"
        f"Target address: {candidate.company_address}\n\n"
        f"Face verification:\n{face_block}\n\n"
        f"Research findings:\n"
        f"{findings_markdown or '(no findings file was produced)'}\n\n"
        "Produce the structured OsintReport now."
    )

    try:
        return cast(OsintReport, scorer.invoke(prompt))
    except ValidationError as exc:
        logger.warning("scorer validation failed, retrying once: %s", exc)
        retry_prompt = (
            f"{prompt}\n\n"
            "YOUR PREVIOUS RESPONSE FAILED VALIDATION:\n"
            f"{exc}\n\n"
            "Fix every reported field. Keep short_description under "
            "700 chars and coverage_notes under 500 chars."
        )
        try:
            return cast(OsintReport, scorer.invoke(retry_prompt))
        except Exception as retry_exc:  # noqa: BLE001 - persist failure
            logger.exception("scorer retry failed")
            return OsintReport(
                risk_score=0,
                short_description="Scoring failed; see coverage_notes.",
                findings=[],
                coverage_notes=(
                    f"Scoring error after retry: "
                    f"{type(retry_exc).__name__}: {retry_exc}"
                ),
            )
    except Exception as exc:  # noqa: BLE001 - persist failure
        logger.exception("scorer failed")
        return OsintReport(
            risk_score=0,
            short_description="Scoring failed; see coverage_notes.",
            findings=[],
            coverage_notes=f"Scoring error: {type(exc).__name__}: {exc}",
        )


def _format_face_verification(fv: FaceVerification) -> str:
    lines = [
        f"status: {fv.status}",
        f"id_face_detected: {fv.id_face_detected}",
        f"id_face_quality: {fv.id_face_quality}",
        f"id_face_confidence: {fv.id_face_confidence:.1f}",
    ]
    if fv.comparisons:
        lines.append("comparisons:")
        for c in fv.comparisons:
            lines.append(
                f"  - matched={c.matched} "
                f"similarity={c.similarity:.1f} "
                f"ref={c.reference_url}"
            )
    if fv.error:
        lines.append(f"error: {fv.error}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recommendation + persistence
# ---------------------------------------------------------------------------


def _recommend(score: int) -> Recommendation:
    from osint_agent.config import recommend_from_score

    return recommend_from_score(score)


def _persist(record: OsintCaseRecord, audit_dir: Path = AUDIT_DIR) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / f"{record.case_id}.json"
    path.write_text(
        json.dumps(record.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_case(candidate: CandidateInput) -> OsintCaseRecord:
    """Run the full deep-agent OSINT pipeline for one candidate."""
    logger.info(
        "case %s start: %s @ %s",
        candidate.case_id,
        candidate.name,
        candidate.company_name,
    )

    face_verification = _preflight_face_verification(candidate.id_front_url)
    logger.info(
        "case %s preflight face: status=%s confidence=%.1f quality=%s",
        candidate.case_id,
        face_verification.status,
        face_verification.id_face_confidence,
        face_verification.id_face_quality,
    )

    runtime = make_runtime(id_front_url=candidate.id_front_url)
    agent = _build_deep_agent(candidate, runtime)

    config = {
        "configurable": {"thread_id": candidate.case_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    try:
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": _initial_message(candidate)}
                ]
            },
            config=config,
        )
    except Exception:  # noqa: BLE001 - persist a failure record
        logger.exception("deep agent invoke failed")
        result = {"files": {}, "messages": []}

    final_face = _merge_face_comparisons(face_verification, runtime)
    findings_md = _collect_findings_markdown(result)

    report = _score_report(candidate, final_face, findings_md)
    recommendation = _recommend(report.risk_score)

    evidence = [
        EvidenceEntry(
            source=entry["source"],  # type: ignore[arg-type]
            query=entry["query"],
            result=entry["result"],
            searched_at=entry["searched_at"],
        )
        for entry in runtime["evidence"]
    ]

    case_record = OsintCaseRecord(
        case_id=candidate.case_id,
        candidate=candidate,
        report=report,
        system_recommendation=recommendation,
        face_verification=final_face,
        evidence=evidence,
        query_history=[entry["query"] for entry in runtime["evidence"]],
        search_count=runtime["total_searches"],
    )

    path = _persist(case_record)
    logger.info(
        "case %s finalized: score=%d/100 recommendation=%s path=%s",
        candidate.case_id,
        report.risk_score,
        recommendation,
        path,
    )

    return case_record
