#!/usr/bin/env python3
r"""
Pairwise global-/local-align identities for selected taxa.

Given:
  - a nucleotide FASTA
  - a taxonomy TSV compatible with `tax_tree.build_tree`
  - optionally: a CSV mapping sequence IDs to strings (labels)

This script:
  - selects sequences under one or more taxa at a given rank (default: genus)
  - computes all-vs-all pairwise alignments with per-pair choice of:
      • Needleman–Wunsch (global, free end gaps)
      • Smith–Waterman (local)
    based on:
      - if an ID→string CSV is provided:
          * if both IDs have labels AND labels match AND
            len(shorter) ≥ 0.9 * len(longer) → Needleman–Wunsch
          * else → Smith–Waterman
      - if no CSV is provided:
          * if len(shorter) ≥ 0.9 * len(longer) → Needleman–Wunsch
          * else → Smith–Waterman
  - writes:
      • identities (square matrix TSV)
      • identities (long/tidy TSV)
      • optional plaintext alignment visualizations (match: '|', mismatch: '.', gaps: ' ')

Examples (semicolon-separated; '_' inside species):
  # quoted (simplest)
  python script.py data/genes.fna data/taxonomy.tsv \
    --taxa "Bacillus;Parabacteroides" --rank genus

  python script.py data/genes.fna data/taxonomy.tsv \
    --taxa "Bacillus_subtilis;Faecalibacterium_prausnitzii" --rank species

  # taxa file (one per line; same rule)
  python script.py data/genes.fna data/taxonomy.tsv --taxa-file taxa.txt --rank species

  # with ID→string CSV
  python script.py data/genes.fna data/taxonomy.tsv \
    --taxa "Bacteroides" \
    --id-strings-csv id_to_label.csv

Requirements:
  - biopython
  - a local module `tax_tree` providing:
      build_tree(path)
      .extract_under(from_rank=<rank>, taxon=<name>, to_rank="species")
      .species_to_ids() -> dict[species] -> list[sequence_id]
"""

from __future__ import annotations
import argparse
import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed

from Bio.Align import PairwiseAligner
from tax_tree import build_tree


# ---------------------- Small utility ----------------------

def _sanitize_label(s: str) -> str:
    """Make a label safe for filenames (avoid spaces/slashes)."""
    return s.replace(" ", "_").replace("/", "_")


# ---------------------- Alignment config ----------------------

def _make_aligner(mode: str) -> PairwiseAligner:
    """
    Create and configure a pairwise aligner.

    mode: "global" (Needleman–Wunsch with free end gaps)
          or "local" (Smith–Waterman)
    """
    aligner = PairwiseAligner()
    aligner.mode = mode
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.match_score = 1
    aligner.mismatch_score = -1

    if mode == "global":
        # Free end gaps (semi-global behavior)
        aligner.target_end_gap_score = 0
        aligner.query_end_gap_score = 0

    # For local mode, end-gap scores are irrelevant by definition.
    return aligner


# ---------------------- I/O helpers ----------------------

def parse_fasta_file(path: Path) -> Dict[str, str]:
    """
    Minimal FASTA reader returning {header -> SEQUENCE (uppercased)}.
    Handles multi-line sequences; ignores empty lines; preserves headers as-is (minus '>').
    """
    seqs: Dict[str, List[str]] = {}
    header: str | None = None

    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
                seqs.setdefault(header, [])
            else:
                if header is None:
                    raise ValueError(f"Sequence encountered before first header in {path}")
                seqs[header].append(line.upper())

    return {h: "".join(chunks) for h, chunks in seqs.items()}


def parse_id_strings_csv(path: Path) -> Dict[str, str]:
    """
    Parse an optional ID→string CSV.

    Expects at least two columns:
      col 0: sequence ID (matching FASTA header IDs)
      col 1: string label

    Returns: {sequence_id: label}
    """
    mapping: Dict[str, str] = {}
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            sid = row[0].strip()
            label = row[1].strip()
            if sid:
                mapping[sid] = label
    return mapping


def write_identity_tables(identity_matrix: List[List[float]],
                          ids: List[str],
                          label: str,
                          outdir: Path) -> Tuple[Path, Path]:
    """
    Writes:
      - Square matrix TSV: {label}_identities.tsv (rows/cols are IDs)
      - Long/‘tidy’ TSV:   {label}_identities_long.tsv (query_id, target_id, identity)
    """
    outdir.mkdir(parents=True, exist_ok=True)
    norm_label = _sanitize_label(label)

    square_path = outdir / f"{norm_label}_identities.tsv"
    long_path   = outdir / f"{norm_label}_identities_long.tsv"

    with square_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id"] + ids)
        for i, row in enumerate(identity_matrix):
            w.writerow([ids[i]] + [f"{v:.6f}" for v in row])

    with long_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query_id", "target_id", "identity"])
        for i, qid in enumerate(ids):
            for j, tid in enumerate(ids):
                w.writerow([qid, tid, f"{identity_matrix[i][j]:.6f}"])

    return square_path, long_path


# ---------------------- Alignment core ----------------------

@dataclass(frozen=True)
class PairTask:
    i: int
    j: int
    id_i: str
    id_j: str
    seq_i: str
    seq_j: str
    need_alignment_strings: bool
    mode: str  # "global" (Needleman–Wunsch) or "local" (Smith–Waterman)


def _align_and_identity_worker(task: PairTask) -> Tuple[int, int, float, str | None, str | None]:
    """
    Worker function for a pair (i, j).
    Returns: (i, j, identity_percent, aligned_query?, aligned_target?)
    """
    aligner = _make_aligner(task.mode)
    aln = aligner.align(task.seq_i, task.seq_j)[0]
    a_blocks, b_blocks = aln.aligned  # coordinate blocks for query and target

    s1 = s2 = None
    matches = 0
    aligned_len = 0

    if task.need_alignment_strings:
        aligned1: List[str] = []
        aligned2: List[str] = []
        qi = ti = 0
        for (qs, qe), (ts, te) in zip(a_blocks, b_blocks):
            if qi < qs:
                aligned1.append(task.seq_i[qi:qs])
                aligned2.append("-" * (qs - qi))
            if ti < ts:
                aligned1.append("-" * (ts - ti))
                aligned2.append(task.seq_j[ti:ts])
            aligned1.append(task.seq_i[qs:qe])
            aligned2.append(task.seq_j[ts:te])
            qi, ti = qe, te

        if qi < len(task.seq_i):
            aligned1.append(task.seq_i[qi:])
            aligned2.append("-" * (len(task.seq_i) - qi))
        if ti < len(task.seq_j):
            aligned1.append("-" * (len(task.seq_j) - ti))
            aligned2.append(task.seq_j[ti:])

        s1 = "".join(aligned1)
        s2 = "".join(aligned2)

        for x, y in zip(s1, s2):
            if x != "-" and y != "-":
                aligned_len += 1
                if x == y:
                    matches += 1
    else:
        # Count matches directly from aligned blocks without reconstructing full strings
        for (qs, qe), (ts, te) in zip(a_blocks, b_blocks):
            block_len = min(qe - qs, te - ts)
            aligned_len += block_len
            s_i = task.seq_i[qs:qs + block_len]
            s_j = task.seq_j[ts:ts + block_len]
            matches += sum(1 for x, y in zip(s_i, s_j) if x == y)

    identity = (matches / aligned_len * 100.0) if aligned_len else 0.0
    return task.i, task.j, identity, s1, s2


def _write_alignment_block(fh,
                           qid: str,
                           tid: str,
                           s1: str,
                           s2: str,
                           identity: float,
                           width: int = 80) -> None:
    """Pretty-prints one alignment block with match/mismatch visualization."""
    def chunks(s: str, w: int):
        for k in range(0, len(s), w):
            yield s[k:k + w]

    mid_chars: List[str] = []
    for a, b in zip(s1, s2):
        if a == "-" or b == "-":
            mid_chars.append(" ")
        elif a == b:
            mid_chars.append("|")
        else:
            mid_chars.append(".")

    mid = "".join(mid_chars)
    fh.write(f"## {qid} vs {tid}  |  identity: {identity:.2f}%\n")
    for a, m, b in zip(chunks(s1, width), chunks(mid, width), chunks(s2, width)):
        fh.write(a + "\n")
        fh.write(m + "\n")
        fh.write(b + "\n\n")


# ---------------------- Selection utilities ----------------------

def dedup_by_id(records: Iterable[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
    """
    records: iterable of tuples (group, species, sid, seq)
    returns: list with unique sid order-preserved
    """
    seen: "OrderedDict[str, Tuple[str, str, str, str]]" = OrderedDict()
    for rec in records:
        sid = rec[2]
        if sid not in seen:
            seen[sid] = rec
    return list(seen.values())


def _decide_alignment_mode(
    id_i: str,
    id_j: str,
    seq_i: str,
    seq_j: str,
    id_to_label: Dict[str, str] | None,
    length_ratio_threshold: float = 0.9,
) -> str:
    """
    Decide per-pair alignment mode ("global" for Needleman–Wunsch, "local" for Smith–Waterman)
    based on the rules:

    If id_to_label is provided:
      - if both IDs have labels AND labels match AND
        len(shorter) >= threshold * len(longer) → "global"
      - else → "local"

    If id_to_label is not provided:
      - if len(shorter) >= threshold * len(longer) → "global"
      - else → "local"
    """
    len_i = len(seq_i)
    len_j = len(seq_j)
    shorter = min(len_i, len_j)
    longer = max(len_i, len_j)
    len_ok = shorter >= length_ratio_threshold * longer

    if id_to_label is not None:
        li = id_to_label.get(id_i)
        lj = id_to_label.get(id_j)
        if li is not None and lj is not None and li == lj and len_ok:
            return "global"
        else:
            return "local"
    else:
        return "global" if len_ok else "local"


# ---------------------- Workflow ----------------------

def run_multi_taxon(tree,
                    fasta_dict: Dict[str, str],
                    species_dict: Dict[str, List[str]],
                    taxa: Sequence[str],
                    rank: str,
                    outdir: Path,
                    save_alignments: bool,
                    workers: int,
                    id_to_label: Dict[str, str] | None = None,
                    length_ratio_threshold: float = 0.9) -> None:
    """
    Multi-taxon workflow across a rank (default: genus).
    Outputs: identity tables; optional alignments.

    Alignment mode per pair is determined by _decide_alignment_mode().
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Build human-readable label and a file-safe version
    raw_label = f"{rank}_" + "_".join(taxa)
    label = _sanitize_label(raw_label)

    align_path = outdir / f"{label}_alignments.txt"
    present: List[Tuple[str, str, str, str]] = []  # (group, species, sid, seq)

    for taxon in taxa:
        for entry in tree.extract_under(from_rank=rank, taxon=taxon, to_rank="species"):
            spec = entry["species"]
            for sid in species_dict.get(spec, []):
                seq = fasta_dict.get(sid)
                if seq is not None:
                    present.append((taxon, spec, sid, seq))

    if not present:
        raise SystemExit("No sequences collected for the provided taxa.")

    present = dedup_by_id(present)
    ids = [p[2] for p in present]
    seqs = [p[3] for p in present]
    n = len(ids)

    # Only compute upper triangle (including diagonal) and mirror results.
    tasks: List[PairTask] = []
    for i in range(n):
        for j in range(i, n):  # j >= i → upper triangle
            mode = _decide_alignment_mode(
                ids[i], ids[j], seqs[i], seqs[j], id_to_label, length_ratio_threshold
            )
            need_strings = save_alignments  # all tasks satisfy i <= j
            tasks.append(
                PairTask(
                    i=i,
                    j=j,
                    id_i=ids[i],
                    id_j=ids[j],
                    seq_i=seqs[i],
                    seq_j=seqs[j],
                    need_alignment_strings=need_strings,
                    mode=mode,
                )
            )

    identity_matrix = [[0.0] * n for _ in range(n)]
    align_results: List[Tuple[int, int, float, str, str]] = []

    with ProcessPoolExecutor(max_workers=workers or None) as ex:
        futures = {ex.submit(_align_and_identity_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            i, j, ident, s1, s2 = fut.result()
            identity_matrix[i][j] = ident
            identity_matrix[j][i] = ident  # mirror to lower triangle
            if save_alignments and s1 is not None and s2 is not None:
                align_results.append((i, j, ident, s1, s2))

    # Use the human-readable label here; write_identity_tables will sanitize consistently
    write_identity_tables(identity_matrix, ids, raw_label, outdir=outdir)

    if save_alignments:
        with align_path.open("w") as fh:
            for i, j, ident, s1, s2 in sorted(align_results, key=lambda x: (x[0], x[1])):
                _write_alignment_block(fh, ids[i], ids[j], s1, s2, ident)


# ---------------------- CLI (semicolon + underscore parsing) ----------------------

def _read_taxa_file(path: Path) -> List[str]:
    names: List[str] = []
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            names.append(s)
    return names


def _split_semicolon_or_comma(s: str) -> List[str]:
    # Split by ';' primarily; fall back to ',' for backwards compatibility
    parts = []
    for chunk in s.split(";"):
        if not chunk.strip():
            continue
        sub = [x for x in (y.strip() for y in chunk.split(",")) if x]
        parts.extend(sub)
    return parts


def _normalize_taxa_string(raw: str) -> List[str]:
    """
    Parse the --taxa string where entities are separated by ';' (or ','),
    and species use '_' between genus and species. '_' (and '+') become spaces.
    """
    if not raw:
        return []
    names = _split_semicolon_or_comma(raw)
    names = [n.replace("_", " ").replace("+", " ").strip() for n in names]
    return [n for n in names if n]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pairwise_identities",
        description="Compute pairwise align identities for selected taxa (global/local per pair)."
    )
    p.add_argument("fasta", type=Path, help="Input nucleotide FASTA (.fna/.fa).")
    p.add_argument("taxonomy", type=Path, help="Taxonomy TSV (kpcofgs).")

    # Single string; user separates entities with ';'
    p.add_argument(
        "--taxa",
        default="Bacteroides",
        help=(
            "List of taxa separated by ';'. Use '_' inside multi-word names "
            "(e.g., 'Bacillus_subtilis;Faecalibacterium_prausnitzii'). "
            "Quote the whole value in shells: --taxa \"...;...\""
        ),
    )
    p.add_argument(
        "--taxa-file",
        type=Path,
        default=None,
        help="Optional file with one taxon per line (comments with #, blank lines ignored).",
    )

    p.add_argument("--rank", default="genus",
                   help="Taxonomic rank for taxa names (default: genus).")
    p.add_argument("--outdir", type=Path, default=Path("./results"),
                   help="Output directory (default: ./results).")
    p.add_argument("--no-alignments", action="store_true",
                   help="Do not write plaintext alignment visualizations.")
    p.add_argument("--workers", type=int, default=0,
                   help="Number of worker processes (default 0 = use CPU count).")

    p.add_argument(
        "--id-strings-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with two columns: sequence ID and a string label. "
            "If provided, per-pair alignment mode (Needleman–Wunsch vs Smith–Waterman) "
            "is chosen using ID labels and length ratios as described in the script docstring."
        ),
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    taxa: List[str] = []
    if args.taxa_file:
        taxa.extend(_read_taxa_file(args.taxa_file))
    taxa.extend(_normalize_taxa_string(args.taxa))

    fasta_dict = parse_fasta_file(args.fasta)
    tree = build_tree(str(args.taxonomy))
    species_index = tree.species_to_ids()

    id_to_label = parse_id_strings_csv(args.id_strings_csv) if args.id_strings_csv else None

    run_multi_taxon(
        tree=tree,
        fasta_dict=fasta_dict,
        species_dict=species_index,
        taxa=taxa,
        rank=args.rank,
        outdir=args.outdir,
        save_alignments=not args.no_alignments,
        workers=args.workers,
        id_to_label=id_to_label,
        length_ratio_threshold=0.9,
    )


if __name__ == "__main__":
    main()

