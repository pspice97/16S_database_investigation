# tax_tree.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import csv
import re
import io

# Terminal rank is "species"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def _clean_taxon(raw: str) -> Optional[str]:
    """
    Normalize values like:
      '2759 Eukaryota(4751)' -> 'Eukaryota'
      'NA Pleosporales fam. incertae sedis' -> 'Pleosporales fam. incertae sedis'
      'NA' or '' -> None
    Also strips trailing commas and trailing parenthetical codes.
    """
    if raw is None:
        return None
    s = raw.strip().rstrip(",")
    if not s or s == "NA":
        return None
    # Drop leading numeric or 'NA'
    m = re.match(r"^\s*(?:NA|\d+)\s+(.*)$", s)
    s = m.group(1).strip() if m else s
    # Drop trailing '(...)'
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    return s or None


@dataclass
class Node:
    name: str
    rank: Optional[str]  # None for artificial root
    children: Dict[str, "Node"] = field(default_factory=dict)
    _leaves: set = field(default_factory=set)  # specI_cluster IDs aggregated below this node

    def child(self, name: str, rank: str) -> "Node":
        if name not in self.children:
            self.children[name] = Node(name=name, rank=rank)
        return self.children[name]

    def add_leaf(self, spec_id: str) -> None:
        self._leaves.add(spec_id)

    def leaves(self) -> List[str]:
        return sorted(self._leaves)


class TaxonomyTree:
    """
    Minimal taxonomy tree supporting:
      - insert_path(...)
      - extract_under(from_rank, taxon, to_rank)
      - species_to_ids()
    """
    def __init__(self):
        self.spec_meta: Dict[str, Dict[str, str]] = {}  # specI_cluster -> non-empty ranks
        self.root = Node(name="root", rank=None)
        self.index: Dict[Tuple[str, str], Node] = {}    # (rank, name) -> Node

    def insert_path(self, specI_cluster: str, ranks: Dict[str, Optional[str]]) -> None:
        """
        Insert a single record: traverse ranks in order and create nodes as needed.
        Aggregate the leaf 'specI_cluster' to every ancestor along the path.
        """
        node = self.root
        node.add_leaf(specI_cluster)
        for rank in RANKS:
            name = ranks.get(rank)
            if not name:
                continue
            node = node.child(name, rank)
            node.add_leaf(specI_cluster)
            self.index[(rank, name)] = node
        if specI_cluster not in self.spec_meta:
            self.spec_meta[specI_cluster] = {k: v for k, v in ranks.items() if v}

    def extract_under(
        self,
        from_rank: str,
        taxon: str,
        to_rank: str,
        group: bool = False,
        include_missing: bool = False,
    ):
        """
        Return entries at `to_rank` under the subtree rooted at (from_rank, taxon).

        If group=True, returns {to_rank_name: [specI_cluster, ...]}.
        Else returns [{"specI_cluster": ..., to_rank: ...}, ...] (deduped & sorted).

        Raises:
          KeyError if (from_rank, taxon) not found.
          ValueError if rank order invalid.
        """
        if from_rank not in RANKS or to_rank not in RANKS:
            raise ValueError(f"Ranks must be in {RANKS}")
        if RANKS.index(to_rank) < RANKS.index(from_rank):
            raise ValueError(f"to_rank '{to_rank}' is above from_rank '{from_rank}'")

        node = self.index.get((from_rank, taxon))
        if node is None:
            raise KeyError(f"No node for rank='{from_rank}' and taxon='{taxon}'")

        if group:
            out: Dict[Optional[str], List[str]] = {}
            for sid in node.leaves():
                meta = self.spec_meta.get(sid, {})
                val = meta.get(to_rank)  # may be None
                if val is None and not include_missing:
                    continue
                out.setdefault(val, []).append(sid)
            for k in list(out.keys()):
                out[k] = sorted(set(out[k]))
            return out

        rows = []
        for sid in node.leaves():
            meta = self.spec_meta.get(sid, {})
            val = meta.get(to_rank)
            if val is None and not include_missing:
                continue
            rows.append({"specI_cluster": sid, to_rank: val})

        seen = set()
        dedup = []
        for r in rows:
            key = (r["specI_cluster"], r[to_rank])
            if key not in seen:
                seen.add(key)
                dedup.append(r)
        dedup.sort(key=lambda r: ("" if r[to_rank] is None else r[to_rank], r["specI_cluster"]))
        return dedup

    def species_to_ids(self, include_missing: bool = False, sort_lists: bool = True) -> Dict[Optional[str], List[str]]:
        """
        Build {species_name: [ids,...]} from the terminal 'species' rank.
        If include_missing=True, include a None key for entries without species.
        """
        buckets: Dict[Optional[str], set] = {}
        for sid, meta in self.spec_meta.items():
            sp = meta.get("species")
            if sp is None and not include_missing:
                continue
            buckets.setdefault(sp, set()).add(sid)

        out: Dict[Optional[str], List[str]] = {}
        for sp, ids in buckets.items():
            lst = list(ids)
            if sort_lists:
                lst.sort()
            out[sp] = lst
        return out


# ---- Builder for `k__...;p__...;...;s__...` two-column TSV ----

_KPCOFGS_TO_RANK = {
    "k": "kingdom",
    "p": "phylum",
    "c": "class",
    "o": "order",
    "f": "family",
    "g": "genus",
    "s": "species",
}

def _parse_path(tax_str: str) -> Dict[str, Optional[str]]:
    """
    Parse a semicolon-separated taxonomy like:
      'k__Bacteria;p__Proteobacteria;...;s__Azospirillum brasilense'
    into {rank -> name}, skipping missing levels.
    """
    ranks: Dict[str, Optional[str]] = {}
    if not tax_str:
        return ranks
    for part in tax_str.split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([kpcfogs])__\s*(.+)$", part)
        if not m:
            continue
        letter, value = m.group(1), m.group(2).strip()
        mapped = _KPCOFGS_TO_RANK.get(letter)
        if mapped:
            ranks[mapped] = _clean_taxon(value)
    return ranks


def build_tree(tsv) -> TaxonomyTree:
    """
    Build a TaxonomyTree from a TWO-COLUMN TSV:
        <leaf_id>\t<k__...;p__...;c__...;o__...;f__...;g__...;s__...>
    Lines starting with '#' are ignored.

    Accepts a file path, file-like, or full TSV string.
    """
    # Open file path / file-like / full string
    if isinstance(tsv, str) and ("\n" in tsv):
        f = io.StringIO(tsv)
        _needs_close = True
    elif isinstance(tsv, str):
        f = open(tsv, "r", encoding="utf-8")
        _needs_close = True
    else:
        f = tsv
        _needs_close = False

    tree = TaxonomyTree()
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        if not row:
            continue
        first = (row[0] or "").strip()
        if first.startswith("#"):
            continue
        if len(row) < 2:
            continue
        spec_id = first
        tax_path = (row[1] or "").strip()
        if not spec_id or not tax_path:
            continue
        ranks = _parse_path(tax_path)
        tree.insert_path(spec_id, ranks)

    if _needs_close:
        f.close()
    return tree

