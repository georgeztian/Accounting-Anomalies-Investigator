"""Mine saved runs for signs that a test, not the company, is wrong.

The AI investigation and the independent auditor often say so in plain words ("the detector most likely compared
figures from different periods", "the extractor filed a line item under total operating expenses"), but the report
still lists the finding. This script reads every saved run (reports/*/run.pkl.gz, no downloads, no AI calls) and
lists, per test:

- results set aside as test errors, and investigation conclusions and innocent explanations that blame the test, the
  data or the parsing;
- auditor corrections that point at a gap in what the pipeline gives the AI (missing evidence, missing definitions);
- investigations that did not finish, with the objections that kept them from finishing;
- test results that are implausible on their face (NaN, out-of-range values, a flag with no evidence).

    .venv/Scripts/python tools/review/mine_runs.py [reports] [--out digest.md]

Each line is a lead to verify against the filing, not a confirmed bug.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

# wording that blames the test or the data rather than the company
BLAME_RE = re.compile(
    r"(?i)\b(?:the )?(?:detector|test|check|extractor|parser|tool)\b[^.]{0,80}?\b(?:most likely|likely|probably|may have|may not|appears to|seems to|"
    r"mis(?:read|label|classif|match)\w*|paired|compared figures|confus\w*|picked|labels?|labell?ed|filed|assigned|wrong|error)|artifact|different periods|"
    r"quarter\w*[^.]{0,40}\b(?:full[- ]year|annual)\b[^.]{0,30}\b(?:compar|pair|mix)|not comparable|mixed denominators?|"
    r"(?:line item|segment|subsidiary|product line)s?[^.]{0,40}\b(?:as|under|for) (?:the )?(?:total|company-wide|whole company)|"
    r"pars(?:ing|er) (?:error|problem|mismatch)|tagging (?:change|error)|scale error|units? (?:mismatch|error)|sign convention")
# auditor corrections that point at a pipeline gap rather than at the writer
GAP_RE = re.compile(r"(?i)evidence(?: pack)? (?:has|contains|gives|shows) no|no evidence (?:item|in the pack)|not in the (?:evidence )?pack|"
                    r"does not (?:state|define|show) (?:how|which|what)|no definition|cannot be checked|truncated|cut off|"
                    r"only available for")


def _finite(x) -> bool:
    return isinstance(x, int | float) and math.isfinite(x)


def _bounds_problem(s) -> str | None:
    from ffi.detectors.plausibility import MAX_DAYS, MAX_YEARS

    v, unit = s.raw_value, (s.unit or "").lower()
    if v is None:
        return None
    if not _finite(v):
        return "result is NaN or infinite"
    if not _finite(s.severity) or not 0 <= s.severity <= 1:
        return f"severity {s.severity} outside 0-1"
    if unit == "days" and not 0 <= v <= MAX_DAYS:
        return f"{v:.0f} days is outside any plausible range"
    if unit == "years" and not 0 <= v <= MAX_YEARS:
        return f"{v:.1f} years is outside any plausible range"
    if s.flagged and not s.evidence and s.family not in ("D",):
        return "flagged with no evidence attached"
    return None


def mine(root: Path) -> str:
    from ffi.report import snapshot

    blame: dict[str, list[str]] = defaultdict(list)
    gaps: dict[str, list[str]] = defaultdict(list)
    unfinished: list[str] = []
    implausible: list[str] = []
    runs = sorted(root.glob(f"*/{snapshot.SNAPSHOT}"))
    for path in runs:
        try:
            out = snapshot.load(path.parent)
        except Exception as e:  # noqa: BLE001 - an old or broken snapshot must not stop the digest
            print(f"skip {path.parent.name}: {e}", file=sys.stderr)
            continue
        run = path.parent.name
        span = getattr(out, "kind", "") == "span"
        sc = out.span if span else out.score
        aside = list(getattr(sc, "set_aside", []))  # results the investigation showed to be test errors: the strongest leads
        findings = list(sc.findings) + aside
        results = [r for y in out.years for r in y.results] if span else out.results
        by_id = {f.id: f for f in findings}
        for f in findings:
            test = f.detectors[0] if f.detectors else f.id
            if f in aside:
                blame[test].append(f"{run} · {f.title}: set aside as a test error: {f.set_aside[:300]}")
            texts = [f.narrative or ""] + list(f.alternatives) + list((getattr(f, "client", None) or {}).values())
            for t in texts:
                for sentence in re.split(r"(?<=[.!?])\s+", t):
                    if BLAME_RE.search(sentence):
                        blame[test].append(f"{run} · {f.title}: {sentence.strip()[:300]}")
        for c in out.agent_cases:
            if not isinstance(c, dict):
                continue
            f = by_id.get(c.get("finding_id", "").split("@")[0]) or by_id.get(c.get("finding_id", ""))
            test = (f.detectors[0] if f and f.detectors else c.get("finding_id", "?"))
            for a in c.get("audits", []):
                for v in a.get("verdicts", []):
                    if v["verdict"] != "supported" and GAP_RE.search(v.get("correction", "")):
                        gaps[test].append(f"{run}: {v['correction'][:260]}")
            if c.get("status") == "degraded":
                last = next((a for a in reversed(c.get("audits", [])) if a.get("verdicts")), None)
                objections = [v["correction"][:160] for v in (last or {}).get("verdicts", []) if v["verdict"] in ("unsupported", "overstated")]
                problems = (c.get("verification") or {}).get("problems") or []
                unfinished.append(f"{run} · {test}: " + "; ".join((problems + objections)[:4]))
        for r in results:
            for s in r.signals:
                why = _bounds_problem(s)
                if why:
                    implausible.append(f"{run} · {s.detector_id} ({s.period}): {why}. {s.rationale[:160]}")

    lines = [f"# Leads from {len(runs)} saved runs", "",
             "Each item is a lead to verify against the filing, not a confirmed bug.", ""]
    lines += ["## 1. The investigation or the auditor blames the test, the data or the parsing", ""]
    for test, items in sorted(blame.items(), key=lambda kv: -len(kv[1])):
        lines += [f"### {test} ({len(items)})"] + [f"- {x}" for x in items[:8]] + [""]
    lines += ["## 2. Auditor corrections pointing at a gap in what the pipeline gives the AI", ""]
    for test, items in sorted(gaps.items(), key=lambda kv: -len(kv[1])):
        lines += [f"### {test} ({len(items)})"] + [f"- {x}" for x in items[:6]] + [""]
    lines += ["## 3. Investigations that did not finish (last objections)", ""] + [f"- {x}" for x in unfinished] + [""]
    lines += ["## 4. Implausible test results", ""] + ([f"- {x}" for x in implausible] or ["- none"]) + [""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default="reports")
    ap.add_argument("--out", help="write the digest to this file instead of printing it")
    a = ap.parse_args()
    digest = mine(Path(a.root))
    if a.out:
        Path(a.out).write_text(digest, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(digest)


if __name__ == "__main__":
    main()
