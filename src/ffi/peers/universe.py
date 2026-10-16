"""Industry universes from SEC's company-by-SIC listing (BLUEPRINT §7)."""

from __future__ import annotations

from ffi.edgar import endpoints
from ffi.edgar.client import EdgarClient


def sic_universe(client: EdgarClient, sic: int, digits: int) -> list[int]:
    """All registrants whose SIC shares the first `digits` digits with `sic`."""
    if digits >= 4:
        return endpoints.companies_by_sic(client, sic)
    prefix = str(sic).zfill(4)[:digits]
    codes = [c for c in endpoints.sic_codes(client) if str(c).zfill(4).startswith(prefix)]
    if sic not in codes:
        codes.append(sic)
    ciks: set[int] = set()
    for code in codes:
        ciks.update(endpoints.companies_by_sic(client, code))
    return sorted(ciks)


def expanding_universe(
    client: EdgarClient, sic: int, min_active: int, active: set[int]
) -> tuple[list[int], int]:
    """Widen 4 -> 3 -> 2 SIC digits until at least `min_active` registrants have current XBRL data.

    `active` is the set of CIKs present in the cross-section frames, so shells and dormant
    registrants don't count toward the minimum.
    """
    last: list[int] = []
    for digits in (4, 3, 2):
        listed = sic_universe(client, sic, digits)
        last = [c for c in listed if c in active]
        if len(last) >= min_active:
            return last, digits
    return last, 2
