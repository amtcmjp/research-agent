from __future__ import annotations

import logging
import sys
from pathlib import Path

from osint_agent.batch import customer_to_candidate, run_batch
from osint_agent.config import AUDIT_DIR, DEFAULT_CUSTOMERS_PATH
from osint_agent.models import OsintCaseRecord

logger = logging.getLogger("osint_agent.main")


def _print_record(record: OsintCaseRecord) -> None:
    print(f"\n=== OSINT case {record.case_id} ===")
    print(f"Name: {record.candidate.name}")
    print(f"Company: {record.candidate.company_name}")
    print(f"Address: {record.candidate.company_address}")
    print(f"Searches: {record.search_count}")

    fv = record.face_verification
    if fv is not None:
        print(
            f"\nFace verification: {fv.status} "
            f"(id_face_detected={fv.id_face_detected}, "
            f"quality={fv.id_face_quality}, "
            f"confidence={fv.id_face_confidence:.1f})"
        )
        for c in fv.comparisons:
            print(
                f"  compared vs {c.reference_url}: "
                f"matched={c.matched} similarity={c.similarity:.1f}"
            )

    report = record.report
    print(f"\nRisk score: {report.risk_score}/100")
    print(f"Summary: {report.short_description}")
    print(f"System recommendation: {record.system_recommendation}")

    if report.findings:
        print("\nFindings:")
        for f in report.findings:
            parts = [f"[{f.category}/{f.classification}]", f.summary]
            if f.source_url:
                parts.append(f"({f.source_url})")
            if f.source_date:
                parts.append(f"[{f.source_date}]")
            parts.append(f"identity={f.identity_confidence}")
            print("- " + " ".join(parts))
    else:
        print("\nFindings: (none)")

    if report.coverage_notes:
        print(f"\nCoverage notes: {report.coverage_notes}")

    print(f"\nAudit record: {AUDIT_DIR / f'{record.case_id}.json'}")


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CUSTOMERS_PATH
    print(f"Loading customers from: {path}")

    records = run_batch(path)
    for r in records:
        _print_record(r)

    print("\n=== Batch summary ===")
    for r in records:
        fv_status = r.face_verification.status if r.face_verification else "none"
        print(
            f"- {r.case_id} | {r.candidate.name} | "
            f"score={r.report.risk_score}/100 | "
            f"rec={r.system_recommendation} | face={fv_status}"
        )


# Public helpers for callers importing this module directly.
__all__ = ["customer_to_candidate", "main"]


if __name__ == "__main__":
    main()
