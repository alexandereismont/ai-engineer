"""
pii_scanner.py — Presidio-based PII scanner for pension fund data files.

Scans:
    - All CSV files under ../data/raw/
    - All .txt files under ../data/sample_documents/

Reports PII findings to stdout and writes a structured pii_report.json.
Exits with code 1 if unexpected PII is found in columns not in ALLOWED_PII_COLUMNS.

Usage
-----
    python pii_scanner.py                      # default paths
    python pii_scanner.py --raw-dir /tmp/raw   # override data directory
    python pii_scanner.py --report /tmp/report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Columns whose PII content is intentional (Phase 06 exercises)
ALLOWED_PII_COLUMNS: set[str] = {
    "counterparty_email",
    "counterparty_iban",
    "analyst_email",
}

# PII entity types to scan for
TARGET_ENTITIES: list[str] = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "PERSON",
    "LOCATION",
    "DATE_TIME",   # can indicate DOB in pension records
    "NRP",         # nationality/religion/political views
]

# Maximum number of example values to store per column finding
MAX_EXAMPLES: int = 3

# Minimum Presidio confidence score to flag as PII
MIN_CONFIDENCE: float = 0.6

# Maximum number of rows to sample per CSV column (for performance)
MAX_SAMPLE_ROWS: int = 500

# Root paths (relative to this file's location)
_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent

DEFAULT_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
DEFAULT_DOCS_DIR = _PROJECT_ROOT / "data" / "sample_documents"
DEFAULT_REPORT_PATH = _PROJECT_ROOT / "pii_report.json"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PiiFinding:
    """A single PII finding within a file."""

    file_path: str
    file_type: str              # "csv" or "txt"
    column_name: str | None     # None for txt files
    entity_type: str
    row_count_with_pii: int
    example_values: list[str]   # anonymised / redacted examples
    allowed: bool               # True if column is in ALLOWED_PII_COLUMNS
    confidence_min: float
    confidence_max: float


@dataclass
class ScanReport:
    """Top-level scan report."""

    scan_date: str
    raw_dir: str
    docs_dir: str
    files_scanned: int
    total_findings: int
    unexpected_pii_found: bool
    findings: list[PiiFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "findings"},
            "findings": [asdict(f) for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Presidio setup
# ---------------------------------------------------------------------------


def build_analyzer() -> AnalyzerEngine:
    """Build and return a configured Presidio AnalyzerEngine."""
    return AnalyzerEngine()


def build_anonymizer() -> AnonymizerEngine:
    """Build and return a configured Presidio AnonymizerEngine."""
    return AnonymizerEngine()


def redact_text(
    text: str,
    results: list[RecognizerResult],
    anonymizer: AnonymizerEngine,
) -> str:
    """Replace detected PII spans with <ENTITY_TYPE> placeholders.

    Args:
        text:       Original text string.
        results:    Presidio RecognizerResult list (span + entity_type).
        anonymizer: Presidio AnonymizerEngine instance.

    Returns:
        Text with PII replaced by labels, e.g. '<EMAIL_ADDRESS>'.
    """
    if not results:
        return text
    try:
        operators = {
            entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
            for entity in {r.entity_type for r in results}
        }
        anonymized = anonymizer.anonymize(
            text=text, analyzer_results=results, operators=operators
        )
        return anonymized.text
    except Exception:  # noqa: BLE001
        return "<redacted>"


# ---------------------------------------------------------------------------
# CSV scanning
# ---------------------------------------------------------------------------


def scan_csv_file(
    path: Path,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
) -> list[PiiFinding]:
    """Scan all string columns in a CSV file for PII.

    Args:
        path:       Path to the CSV file.
        analyzer:   Presidio AnalyzerEngine instance.
        anonymizer: Presidio AnonymizerEngine instance.

    Returns:
        List of PiiFindings for columns where PII was detected.
    """
    findings: list[PiiFinding] = []

    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return []
            fieldnames: list[str] = list(reader.fieldnames)

            # Read up to MAX_SAMPLE_ROWS rows
            rows: list[dict[str, str]] = []
            for i, row in enumerate(reader):
                if i >= MAX_SAMPLE_ROWS:
                    break
                rows.append(row)

    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR reading {path}: {exc}", file=sys.stderr)
        return []

    if not rows:
        return []

    for column in fieldnames:
        # Collect non-empty cell values
        cell_values = [str(row.get(column, "") or "").strip() for row in rows]
        cell_values = [v for v in cell_values if v]
        if not cell_values:
            continue

        # Track per-cell hits
        pii_cells: list[tuple[str, list[RecognizerResult]]] = []

        for value in cell_values:
            results = analyzer.analyze(
                text=value,
                entities=TARGET_ENTITIES,
                language="en",
            )
            high_conf = [r for r in results if r.score >= MIN_CONFIDENCE]
            if high_conf:
                pii_cells.append((value, high_conf))

        if not pii_cells:
            continue

        # Group by entity type
        entity_types_found: set[str] = {
            r.entity_type for _, results in pii_cells for r in results
        }

        for entity_type in entity_types_found:
            # Examples for this entity type
            examples: list[str] = []
            type_confidences: list[float] = []
            row_hit_count = 0

            for value, results in pii_cells:
                type_results = [r for r in results if r.entity_type == entity_type]
                if type_results:
                    row_hit_count += 1
                    type_confidences.extend(r.score for r in type_results)
                    if len(examples) < MAX_EXAMPLES:
                        examples.append(redact_text(value, type_results, anonymizer))

            allowed = column.lower() in {c.lower() for c in ALLOWED_PII_COLUMNS}
            findings.append(
                PiiFinding(
                    file_path=str(path),
                    file_type="csv",
                    column_name=column,
                    entity_type=entity_type,
                    row_count_with_pii=row_hit_count,
                    example_values=examples,
                    allowed=allowed,
                    confidence_min=min(type_confidences),
                    confidence_max=max(type_confidences),
                )
            )

    return findings


# ---------------------------------------------------------------------------
# TXT scanning
# ---------------------------------------------------------------------------


def scan_txt_file(
    path: Path,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
) -> list[PiiFinding]:
    """Scan a plain-text file for PII entities.

    Args:
        path:       Path to the .txt file.
        analyzer:   Presidio AnalyzerEngine instance.
        anonymizer: Presidio AnonymizerEngine instance.

    Returns:
        List of PiiFindings, one per entity type found.
    """
    findings: list[PiiFinding] = []

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR reading {path}: {exc}", file=sys.stderr)
        return []

    all_results = analyzer.analyze(text=text, entities=TARGET_ENTITIES, language="en")
    all_results = [r for r in all_results if r.score >= MIN_CONFIDENCE]

    if not all_results:
        return []

    entity_types: set[str] = {r.entity_type for r in all_results}

    for entity_type in entity_types:
        type_results = [r for r in all_results if r.entity_type == entity_type]
        examples: list[str] = []
        for r in type_results[:MAX_EXAMPLES]:
            # Extract a short snippet around the detected span
            snippet = text[max(0, r.start - 15): r.end + 15]
            examples.append(redact_text(snippet, [r], anonymizer))

        confidences = [r.score for r in type_results]
        findings.append(
            PiiFinding(
                file_path=str(path),
                file_type="txt",
                column_name=None,
                entity_type=entity_type,
                row_count_with_pii=len(type_results),
                example_values=examples,
                allowed=False,  # txt files are never in ALLOWED_PII_COLUMNS
                confidence_min=min(confidences),
                confidence_max=max(confidences),
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------


def run_scan(
    raw_dir: Path,
    docs_dir: Path,
    report_path: Path,
) -> ScanReport:
    """Execute the full PII scan across raw/ and sample_documents/.

    Args:
        raw_dir:     Directory containing CSV files to scan.
        docs_dir:    Directory containing .txt files to scan.
        report_path: Output path for the JSON report.

    Returns:
        Completed ScanReport instance.
    """
    analyzer = build_analyzer()
    anonymizer = build_anonymizer()

    all_findings: list[PiiFinding] = []
    files_scanned = 0

    # Scan CSV files
    csv_files = sorted(raw_dir.glob("**/*.csv")) if raw_dir.exists() else []
    print(f"\nScanning {len(csv_files)} CSV file(s) in {raw_dir} ...")
    for csv_path in csv_files:
        print(f"  {csv_path.name} ...", end=" ", flush=True)
        file_findings = scan_csv_file(csv_path, analyzer, anonymizer)
        all_findings.extend(file_findings)
        files_scanned += 1
        print(f"{len(file_findings)} finding(s)")

    # Scan TXT files
    txt_files = sorted(docs_dir.glob("**/*.txt")) if docs_dir.exists() else []
    print(f"\nScanning {len(txt_files)} TXT file(s) in {docs_dir} ...")
    for txt_path in txt_files:
        print(f"  {txt_path.name} ...", end=" ", flush=True)
        file_findings = scan_txt_file(txt_path, analyzer, anonymizer)
        all_findings.extend(file_findings)
        files_scanned += 1
        print(f"{len(file_findings)} finding(s)")

    # Determine if any UNEXPECTED PII was found
    unexpected_found = any(not f.allowed for f in all_findings)

    report = ScanReport(
        scan_date=datetime.now(timezone.utc).isoformat(),
        raw_dir=str(raw_dir),
        docs_dir=str(docs_dir),
        files_scanned=files_scanned,
        total_findings=len(all_findings),
        unexpected_pii_found=unexpected_found,
        findings=all_findings,
    )

    # Write JSON report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"\nReport written to: {report_path}")

    return report


def print_summary(report: ScanReport) -> None:
    """Print a human-readable summary of the scan report to stdout."""
    print("\n" + "=" * 60)
    print("PII SCAN SUMMARY")
    print("=" * 60)
    print(f"Scan date:          {report.scan_date}")
    print(f"Files scanned:      {report.files_scanned}")
    print(f"Total findings:     {report.total_findings}")
    print(
        f"Unexpected PII:     "
        f"{'YES — ACTION REQUIRED' if report.unexpected_pii_found else 'None'}"
    )

    if report.findings:
        print("\nFindings by file:")
        grouped: dict[str, list[PiiFinding]] = {}
        for f in report.findings:
            grouped.setdefault(f.file_path, []).append(f)

        for file_path, findings in grouped.items():
            print(f"\n  {Path(file_path).name}")
            for finding in findings:
                col_str = f" [column: {finding.column_name}]" if finding.column_name else ""
                status = "ALLOWED" if finding.allowed else "UNEXPECTED"
                print(
                    f"    [{status}] {finding.entity_type}{col_str} "
                    f"— {finding.row_count_with_pii} occurrence(s) "
                    f"(confidence {finding.confidence_min:.2f}–{finding.confidence_max:.2f})"
                )
                for ex in finding.example_values:
                    print(f"      example: {ex}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan pension data files for PII using Microsoft Presidio."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory containing CSV files to scan (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help=f"Directory containing TXT files to scan (default: {DEFAULT_DOCS_DIR})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Output path for JSON report (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_CONFIDENCE,
        help=f"Minimum Presidio confidence score 0.0–1.0 (default: {MIN_CONFIDENCE})",
    )
    args = parser.parse_args()

    # Override module-level constant if CLI flag was passed
    global MIN_CONFIDENCE  # noqa: PLW0603
    MIN_CONFIDENCE = args.min_confidence

    report = run_scan(
        raw_dir=args.raw_dir,
        docs_dir=args.docs_dir,
        report_path=args.report,
    )
    print_summary(report)

    if report.unexpected_pii_found:
        unexpected = [f for f in report.findings if not f.allowed]
        print(
            f"\nFAIL: {len(unexpected)} unexpected PII finding(s) detected.\n"
            "Review pii_report.json and either anonymise the data or add the column\n"
            f"to ALLOWED_PII_COLUMNS if the PII is intentional.\n"
            f"Current ALLOWED_PII_COLUMNS: {sorted(ALLOWED_PII_COLUMNS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nPASS: No unexpected PII detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
