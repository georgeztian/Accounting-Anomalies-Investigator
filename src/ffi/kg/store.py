"""Embedded knowledge graph: DuckDB tables + NetworkX views, bitemporal.

Every node and edge carries `recorded_at` (when it became public / was extracted). All reads go
through `as_of`, so a run as of a past date cannot see later filings.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import networkx as nx
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS node (
    id VARCHAR PRIMARY KEY,
    type VARCHAR NOT NULL,
    label VARCHAR,
    props JSON,
    valid_from DATE,
    valid_to DATE,
    recorded_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS edge (
    src VARCHAR NOT NULL,
    dst VARCHAR NOT NULL,
    type VARCHAR NOT NULL,
    props JSON,
    valid_from DATE,
    valid_to DATE,
    recorded_at TIMESTAMP,
    source_accession VARCHAR,
    confidence DOUBLE,
    extractor VARCHAR,
    verified BOOLEAN,
    PRIMARY KEY (src, dst, type)
);
"""
NODE_COLS = ["id", "type", "label", "props", "valid_from", "valid_to", "recorded_at"]
EDGE_COLS = ["src", "dst", "type", "props", "valid_from", "valid_to", "recorded_at", "source_accession", "confidence",
             "extractor", "verified"]


def _ts(v: Any) -> Any:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


class DuckNXGraphStore:
    """Default backend. One DuckDB file per run; NetworkX for traversal."""

    def __init__(self, path: Path | str = ":memory:", as_of: date | None = None) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.con = duckdb.connect(self.path)
        self.con.execute(SCHEMA)
        self.as_of = as_of
        self._cache: dict[tuple, nx.MultiDiGraph] = {}
        self._lock = threading.RLock()  # DuckDB connections are not safe for concurrent use

    # ------------------------------------------------------------- writes (bulk, via DataFrames)
    def reset(self) -> None:
        self.con.execute("DELETE FROM edge; DELETE FROM node;")
        self._cache.clear()

    def upsert_nodes(self, nodes: Iterable[dict[str, Any]]) -> None:
        with self._lock:
            self._upsert_nodes(nodes)

    def upsert_edges(self, edges: Iterable[dict[str, Any]]) -> None:
        with self._lock:
            self._upsert_edges(edges)

    def _upsert_nodes(self, nodes: Iterable[dict[str, Any]]) -> None:
        rows = [(n["id"], n["type"], n.get("label", n["id"]), json.dumps(n.get("props", {}), default=str),
                 _ts(n.get("valid_from")), _ts(n.get("valid_to")), _ts(n.get("recorded_at"))) for n in nodes]
        if not rows:
            return
        df = pd.DataFrame(rows, columns=NODE_COLS).drop_duplicates("id", keep="last")
        for c in ("valid_from", "valid_to", "recorded_at"):
            df[c] = pd.to_datetime(df[c])
        self.con.register("_nodes_in", df)
        self.con.execute("INSERT OR REPLACE INTO node SELECT id, type, label, props::JSON, valid_from::DATE, "
                         "valid_to::DATE, recorded_at FROM _nodes_in")
        self.con.unregister("_nodes_in")
        self._cache.clear()

    def _upsert_edges(self, edges: Iterable[dict[str, Any]]) -> None:
        rows = [(e["src"], e["dst"], e["type"], json.dumps(e.get("props", {}), default=str),
                 _ts(e.get("valid_from")), _ts(e.get("valid_to")), _ts(e.get("recorded_at")),
                 e.get("source_accession"), float(e.get("confidence", 1.0)), e.get("extractor", "rule"),
                 bool(e.get("verified", True))) for e in edges]
        if not rows:
            return
        df = pd.DataFrame(rows, columns=EDGE_COLS).drop_duplicates(["src", "dst", "type"], keep="last")
        for c in ("valid_from", "valid_to", "recorded_at"):
            df[c] = pd.to_datetime(df[c])
        df["source_accession"] = df["source_accession"].astype("object")
        self.con.register("_edges_in", df)
        self.con.execute("INSERT OR REPLACE INTO edge SELECT src, dst, type, props::JSON, valid_from::DATE, valid_to::DATE, "
                         "recorded_at, source_accession, confidence, extractor, verified FROM _edges_in")
        self.con.unregister("_edges_in")
        self._cache.clear()

    # ------------------------------------------------------------- reads (always as_of)
    def _asof_clause(self) -> str:
        if self.as_of is None:
            return "TRUE"
        cutoff = datetime.combine(self.as_of, datetime.max.time()).isoformat(sep=" ")
        return f"(recorded_at IS NULL OR recorded_at <= TIMESTAMP '{cutoff}')"

    def _frame(self, q: str, params: list[Any]) -> pd.DataFrame:
        with self._lock:
            df = self.con.execute(q, params).df()
        if not df.empty:
            df["props"] = df["props"].map(lambda s: json.loads(s) if isinstance(s, str) else (s or {}))
        return df

    def nodes(self, type_: str | None = None) -> pd.DataFrame:
        q, params = f"SELECT * FROM node WHERE {self._asof_clause()}", []
        if type_:
            q += " AND type = ?"
            params.append(type_)
        return self._frame(q, params)

    def edges(self, type_: str | None = None, verified_only: bool = True) -> pd.DataFrame:
        q, params = f"SELECT * FROM edge WHERE {self._asof_clause()}", []
        if type_:
            q += " AND type = ?"
            params.append(type_)
        if verified_only:
            q += " AND verified"
        return self._frame(q, params)

    def node(self, node_id: str) -> dict[str, Any] | None:
        df = self._frame(f"SELECT * FROM node WHERE id = ? AND {self._asof_clause()}", [node_id])
        return None if df.empty else df.iloc[0].to_dict()

    def graph(self, edge_types: Iterable[str] | None = None, verified_only: bool = True) -> nx.MultiDiGraph:
        with self._lock:
            return self._graph(edge_types, verified_only)

    def _graph(self, edge_types: Iterable[str] | None = None, verified_only: bool = True) -> nx.MultiDiGraph:
        key = (tuple(sorted(edge_types)) if edge_types else None, verified_only)
        if key in self._cache:
            return self._cache[key]
        g = nx.MultiDiGraph()
        for r in self.nodes().itertuples(index=False):
            g.add_node(r.id, type=r.type, label=r.label, **{k: v for k, v in (r.props or {}).items() if k not in ("type", "label")})
        edges = self.edges(verified_only=verified_only)
        if edge_types and not edges.empty:
            edges = edges[edges["type"].isin(list(edge_types))]
        for r in edges.itertuples(index=False):
            props = {k: v for k, v in (r.props or {}).items() if k not in ("type", "confidence")}
            g.add_edge(r.src, r.dst, key=r.type, type=r.type, confidence=r.confidence, **props)
        self._cache[key] = g
        return g

    def neighborhood(self, node_id: str, edge_types: Iterable[str] | None = None, hops: int = 1) -> nx.MultiDiGraph:
        g = self.graph(edge_types)
        if node_id not in g:
            return nx.MultiDiGraph()
        keep = nx.single_source_shortest_path_length(g.to_undirected(as_view=True), node_id, cutoff=hops).keys()
        return g.subgraph(keep).copy()

    def counts(self) -> dict[str, int]:
        with self._lock:
            return self._counts()

    def _counts(self) -> dict[str, int]:
        n = self.con.execute(f"SELECT type, count(*) FROM node WHERE {self._asof_clause()} GROUP BY type ORDER BY type").fetchall()
        e = self.con.execute(f"SELECT type, count(*) FROM edge WHERE {self._asof_clause()} GROUP BY type ORDER BY type").fetchall()
        return {**{f"node:{t}": c for t, c in n}, **{f"edge:{t}": c for t, c in e}}

    def close(self) -> None:
        self.con.close()
