"""Batch-run the OSINT agent over a customers JSON file."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from osint_agent.config import DEFAULT_CUSTOMERS_PATH
from osint_agent.deep_pipeline import run_case
from osint_agent.models import CandidateInput, OsintCaseRecord

logger = logging.getLogger("osint_agent.batch")


def _compose_address(company: dict) -> str:
    parts = [
        company.get("zip") or "",
        company.get("prefecture") or "",
        company.get("address") or "",
    ]
    return " ".join(p for p in parts if p).strip()


def customer_to_candidate(record: dict) -> CandidateInput:
    rep = record.get("representative") or {}
    company = record.get("company") or {}

    customer_id = record.get("customer_id") or f"idx_{record.get('id', 'unknown')}"

    return CandidateInput(
        case_id=f"cust_{customer_id}",
        name=rep.get("name") or "Unknown",
        company_name=company.get("name") or "Unknown",
        company_address=_compose_address(company) or "Unknown",
        id_front_url=rep.get("id_front") or "",
        id_back_url=rep.get("id_back") or "",
        sns_url=rep.get("sns_url") or "",
        submitted_by="batch_import",
    )


def run_batch(path: Path) -> list[OsintCaseRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected a list of customer records in {path}")

    records: list[OsintCaseRecord] = []
    for idx, item in enumerate(data, start=1):
        candidate = customer_to_candidate(item)
        logger.info(
            "processing %d/%d: case_id=%s name=%s",
            idx,
            len(data),
            candidate.case_id,
            candidate.name,
        )
        try:
            record = run_case(candidate)
        except Exception:
            logger.exception("case %s failed", candidate.case_id)
            continue
        records.append(record)
    return records


def _demo() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CUSTOMERS_PATH

    print(f"Reading customers from: {path}")
    records = run_batch(path)

    print(f"\n=== Batch complete: {len(records)} cases ===")
    for r in records:
        fv_status = r.face_verification.status if r.face_verification else "none"
        print(
            f"- {r.case_id} | {r.candidate.name} | "
            f"score={r.report.risk_score}/100 | "
            f"rec={r.system_recommendation} | face={fv_status}"
        )


if __name__ == "__main__":
    _demo()
