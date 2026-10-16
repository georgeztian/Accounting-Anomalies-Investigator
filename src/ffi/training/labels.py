"""Starter labels from SEC full-text search: 8-K Item 4.02 (non-reliance on previously issued financials).

Weak labels: the restated fiscal year is approximated as the fiscal year ended before the 8-K was filed.
AAER-based labels (e.g. the USC Marshall AAER dataset) are stronger and can be supplied as the same CSV.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ffi.edgar.client import EdgarClient

EFTS = ("https://efts.sec.gov/LATEST/search-index?q=%22non-reliance%22&forms=8-K&dateRange=custom"
        "&startdt={y}-01-01&enddt={y}-12-31&from={start}")


def build_labels(client: EdgarClient, start_year: int, end_year: int, out: Path, max_per_year: int = 400) -> int:
    rows: dict[tuple[int, int], dict] = {}
    for y in range(start_year, end_year + 1):
        start = 0
        while start < max_per_year:
            data = json.loads(client.get_bytes(EFTS.format(y=y, start=start), max_age_s=90 * 86400))
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break
            for h in hits:
                src = h.get("_source", {})
                if "4.02" not in (src.get("items") or []) or not src.get("ciks"):
                    continue
                cik = int(src["ciks"][0])
                fd = src.get("file_date", f"{y}-12-31")
                fy = int(fd[:4]) - 1
                rows.setdefault((cik, fy), {"cik": cik, "fiscal_year": fy, "label": 1, "source": "8-K 4.02", "filed": fd,
                                            "name": (src.get("display_names") or [""])[0]})
            start += len(hits)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["cik", "fiscal_year", "label", "source", "filed", "name"])
        w.writeheader()
        w.writerows(rows.values())
    return len(rows)
