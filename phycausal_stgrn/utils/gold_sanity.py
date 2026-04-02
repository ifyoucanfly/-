from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TOKEN_UNIPROT = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9](?:-\d+)?$|^[A-NR-Z][0-9][A-Z0-9]{3}[0-9](?:-\d+)?$")
TOKEN_PDB = re.compile(r"^[0-9][A-Z0-9]{3}(?:_[A-Z0-9])?$")
TOKEN_COMPLEX = re.compile(r"^COMPLEX:", re.IGNORECASE)
TOKEN_BAD = re.compile(r"(^COMPLEX:)|(_)" + "|(^[OPQ][0-9][A-Z0-9]{3}[0-9]$)|(^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$)")


@dataclass
class GoldSanitizeResult:
    rows: List[Tuple[str, str]]
    summary: Dict[str, object]


def _norm(g: str) -> str:
    return str(g).strip().upper()


def _resolve_columns(fieldnames: Sequence[str] | None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not fieldnames:
        return None, None, None
    lower = {str(c).strip().lower(): str(c) for c in fieldnames}
    pairs = [
        ("src_gene", "dst_gene"),
        ("src", "dst"),
        ("tf", "target"),
        ("source", "target"),
        ("regulator", "target"),
    ]
    src_key = dst_key = None
    for a, b in pairs:
        if a in lower and b in lower:
            src_key, dst_key = lower[a], lower[b]
            break
    if src_key is None:
        src_key = fieldnames[0]
    if dst_key is None:
        dst_key = fieldnames[1] if len(fieldnames) > 1 else fieldnames[0]
    label_key = None
    for k in ("label", "y", "sign"):
        if k in lower:
            label_key = lower[k]
            break
    return src_key, dst_key, label_key


def _read_gene_pairs(path: str, limit: int = 50000) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        src_key, dst_key, _ = _resolve_columns(reader.fieldnames)
        if not src_key or not dst_key:
            return rows
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append((_norm(row.get(src_key, '')), _norm(row.get(dst_key, ''))))
    return rows


def _find_local_id_map(path: str) -> Optional[str]:
    base_dir = Path(path).resolve().parent
    candidates: List[Path] = []
    for pat in ("*idmap*.csv", "*mapping*.csv", "*map*.csv", "*map*.tsv"):
        candidates.extend(sorted(base_dir.glob(pat)))
    for cand in candidates:
        if cand.is_file() and cand.stat().st_size > 0:
            return str(cand)
    return None


def _load_local_id_map(mapping_path: Optional[str]) -> Dict[str, str]:
    if not mapping_path or not os.path.exists(mapping_path):
        return {}
    sep = "\t" if str(mapping_path).lower().endswith(".tsv") else ","
    with open(mapping_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        if not reader.fieldnames:
            return {}
        lower = {str(c).strip().lower(): str(c) for c in reader.fieldnames}
        src_keys = [k for k in ("uniprot", "uniprot_id", "pdb", "protein_id", "source", "token", "id") if k in lower]
        dst_keys = [k for k in ("gene_symbol", "symbol", "gene", "target", "mapped_symbol") if k in lower]
        if not src_keys or not dst_keys:
            return {}
        src_key = lower[src_keys[0]]
        dst_key = lower[dst_keys[0]]
        out: Dict[str, str] = {}
        for row in reader:
            s = _norm(row.get(src_key, ""))
            t = _norm(row.get(dst_key, ""))
            if s and t:
                out[s] = t
        return out


def _expand_complex(token: str) -> List[str]:
    tok = _norm(token)
    if TOKEN_COMPLEX.match(tok):
        tok = tok.split(":", 1)[1]
    pieces = [_norm(p) for p in tok.split("_") if _norm(p)]
    return pieces or ([tok] if tok else [])


def _map_single_token(token: str, *, present: set[str], id_map: Dict[str, str]) -> Optional[str]:
    tok = _norm(token)
    if not tok:
        return None
    if tok in present:
        return tok
    if tok in id_map:
        mapped = _norm(id_map[tok])
        return mapped if (not present or mapped in present) else None
    # Already a gene-like token with letters and digits but not protein/PDB codes.
    if not TOKEN_UNIPROT.match(tok) and not TOKEN_PDB.match(tok) and tok.replace("-", "").isalnum() and any(c.isalpha() for c in tok):
        return tok if (not present or tok in present) else None
    return None


def _resolve_token_candidates(token: str, *, present: set[str], id_map: Dict[str, str]) -> List[str]:
    tok = _norm(token)
    if not tok:
        return []
    direct = _map_single_token(tok, present=present, id_map=id_map)
    if direct is not None:
        return [direct]
    if TOKEN_COMPLEX.match(tok) or "_" in tok:
        parts = _expand_complex(tok)
        mapped = []
        for p in parts:
            mp = _map_single_token(p, present=present, id_map=id_map)
            if mp:
                mapped.append(mp)
        # Deduplicate while preserving order.
        seen = set()
        uniq = []
        for g in mapped:
            if g not in seen:
                uniq.append(g)
                seen.add(g)
        return uniq
    return []


def sanitize_gold_pairs(
    path: str,
    *,
    present_genes: Optional[Sequence[str]] = None,
    mapping_path: Optional[str] = None,
    label_threshold: float = 0.5,
    max_rows: int = 200000,
) -> GoldSanitizeResult:
    present_iter = [] if present_genes is None else list(present_genes)
    present = set(_norm(g) for g in present_iter if _norm(g))
    auto_mapping_path = mapping_path or _find_local_id_map(path)
    id_map = _load_local_id_map(auto_mapping_path)

    raw_rows = 0
    kept_rows: List[Tuple[str, str]] = []
    dropped_rows = 0
    dropped_token_counter: Counter[str] = Counter()
    mapped_token_counter: Counter[str] = Counter()
    exploded_complex_rows = 0
    duplicate_rows = 0

    if not path or not os.path.exists(path):
        summary = {
            "ok": False,
            "reason": "empty_or_missing",
            "mapping_path": auto_mapping_path,
            "num_rows": 0,
            "sanitized_rows": 0,
            "dropped_rows": 0,
            "duplicate_rows": 0,
            "exploded_complex_rows": 0,
            "bad_token_examples": [],
            "mapped_token_examples": [],
        }
        return GoldSanitizeResult(rows=[], summary=summary)

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        src_key, dst_key, label_key = _resolve_columns(reader.fieldnames)
        if not src_key or not dst_key:
            return GoldSanitizeResult(rows=[], summary={"ok": False, "reason": "missing_columns", "mapping_path": auto_mapping_path})
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            raw_rows += 1
            if label_key is not None:
                try:
                    y = float(row.get(label_key, "1"))
                except Exception:
                    y = 1.0
                if y <= label_threshold:
                    continue
            src_raw = row.get(src_key, "")
            dst_raw = row.get(dst_key, "")
            src_cands = _resolve_token_candidates(src_raw, present=present, id_map=id_map)
            dst_cands = _resolve_token_candidates(dst_raw, present=present, id_map=id_map)
            if not src_cands:
                dropped_token_counter[_norm(src_raw)] += 1
            if not dst_cands:
                dropped_token_counter[_norm(dst_raw)] += 1
            if not src_cands or not dst_cands:
                dropped_rows += 1
                continue
            if len(src_cands) > 1 or len(dst_cands) > 1:
                exploded_complex_rows += 1
            for s in src_cands:
                for t in dst_cands:
                    if not s or not t:
                        continue
                    kept_rows.append((s, t))
                    if _norm(src_raw) != s:
                        mapped_token_counter[_norm(src_raw)] += 1
                    if _norm(dst_raw) != t:
                        mapped_token_counter[_norm(dst_raw)] += 1

    dedup_rows: List[Tuple[str, str]] = []
    seen = set()
    for s, t in kept_rows:
        key = (s, t)
        if key in seen:
            duplicate_rows += 1
            continue
        dedup_rows.append(key)
        seen.add(key)

    uniq_genes = sorted({g for ab in dedup_rows for g in ab if g})
    overlap = [g for g in uniq_genes if (not present or g in present)]
    evaluable = 0
    if present:
        evaluable = sum(1 for s, t in dedup_rows if s in present and t in present)
    ok = bool(dedup_rows)
    summary = {
        "ok": ok,
        "reason": "ok" if ok else "all_rows_dropped_after_sanitization",
        "mapping_path": auto_mapping_path,
        "num_rows": raw_rows,
        "sanitized_rows": int(len(dedup_rows)),
        "num_unique_genes": int(len(uniq_genes)),
        "overlap_gene_count": int(len(overlap)),
        "evaluable_edge_count": int(evaluable),
        "dropped_rows": int(dropped_rows),
        "duplicate_rows": int(duplicate_rows),
        "exploded_complex_rows": int(exploded_complex_rows),
        "bad_token_examples": [k for k, _ in dropped_token_counter.most_common(12)],
        "mapped_token_examples": [k for k, _ in mapped_token_counter.most_common(12)],
        "drop_reasons": {
            "protein_or_complex_tokens_without_mapping": int(sum(dropped_token_counter.values())),
        },
    }
    return GoldSanitizeResult(rows=dedup_rows, summary=summary)


def sanitize_gold_csv(
    path: str,
    *,
    output_path: Optional[str] = None,
    present_genes: Optional[Sequence[str]] = None,
    mapping_path: Optional[str] = None,
    summary_path: Optional[str] = None,
    label_threshold: float = 0.5,
) -> Dict[str, object]:
    res = sanitize_gold_pairs(
        path,
        present_genes=present_genes,
        mapping_path=mapping_path,
        label_threshold=label_threshold,
    )
    out_path = output_path or str(Path(path).with_suffix("")) + ".sanitized.csv"
    os.makedirs(str(Path(out_path).parent), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_gene", "dst_gene", "label"])
        for s, t in res.rows:
            w.writerow([s, t, 1])
    summary = dict(res.summary)
    summary["sanitized_path"] = out_path
    if summary_path:
        os.makedirs(str(Path(summary_path).parent), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def drop_bad_tokens(
    path: str,
    *,
    present_genes: Optional[Sequence[str]] = None,
    mapping_path: Optional[str] = None,
    output_path: Optional[str] = None,
    summary_path: Optional[str] = None,
    label_threshold: float = 0.5,
) -> Tuple[str, Dict[str, object]]:
    summary = sanitize_gold_csv(
        path,
        output_path=output_path,
        present_genes=present_genes,
        mapping_path=mapping_path,
        summary_path=summary_path,
        label_threshold=label_threshold,
    )
    return str(summary.get("sanitized_path", output_path or path)), summary


def inspect_gold_file(path: str, present_genes: Sequence[str] | None = None, min_overlap_edges: int = 20) -> Dict[str, object]:
    res = sanitize_gold_pairs(path, present_genes=present_genes)
    rows = res.rows
    toks = [g for ab in rows for g in ab if g]
    uniq = sorted(set(toks))
    present_iter = [] if present_genes is None else list(present_genes)
    present = set(_norm(g) for g in present_iter if _norm(g))
    overlap = [g for g in uniq if g in present] if present else uniq
    evaluable = 0
    if present:
        evaluable = sum(1 for s, t in rows if s in present and t in present)
    ok = bool(rows) and (not present or evaluable >= int(min_overlap_edges))
    reason = []
    if not rows:
        reason.append('empty_or_missing')
    if present and evaluable < int(min_overlap_edges):
        reason.append(f'evaluable_edges={evaluable}')
    out = dict(res.summary)
    out.update({
        'ok': ok,
        'num_rows': int(res.summary.get('num_rows', len(rows))),
        'num_unique_genes': len(uniq),
        'bad_token_examples': list(res.summary.get('bad_token_examples', []))[:12],
        'mapped_token_examples': list(res.summary.get('mapped_token_examples', []))[:12],
        'overlap_gene_count': len(overlap),
        'evaluable_edge_count': int(evaluable),
        'reason': ';'.join(reason) if reason else 'ok',
    })
    return out
