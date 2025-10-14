#!/usr/bin/env python3
"""
Heatmap + stacked & overlaid histograms of pairwise identities.
- Heatmap groups rows/cols by SPECIES (using s__ from the taxonomy file).
- Histograms can categorize by GENUS or SPECIES via --level.

Color mapping for categories is consistent between the stacked and overlaid histograms.
"""

from __future__ import annotations
import argparse
import matplotlib
from pathlib import Path
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import math
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import matplotlib.colors as mcolors

# ---------- defaults ----------
THRESHOLD_DEFAULT = 40.0
PX_PER_CELL_DEFAULT = 1.2
OVERLAY_ALPHA_DEFAULT = 0.45
EDGE_COLOR_DEFAULT = "#222"
EDGE_LINEWIDTH_DEFAULT = 0.7
INTER_COLOR_DEFAULT = "#7f7f7f"  # neutral gray for "inter"


# ---------- helpers ----------
def ensure_outdir(path: Path) -> None:
    if path and path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

def choose_dpi(n: int) -> int:
    return int(max(150, min(500, n / 10)))


# ---------- data loaders ----------
def load_identity_square(path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path, sep="\t", header=0, index_col=0)
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Matrix not square: {df.shape}")
    ids = df.index.astype(str).tolist()
    mat = df.to_numpy(dtype=np.float32, copy=False)
    return mat, ids

def _parse_rank_from_lineage(lineage: str, rank_prefix: str) -> str:
    for part in lineage.split(";"):
        p = part.strip()
        if p.startswith(rank_prefix):
            val = p[len(rank_prefix):].strip()
            return val if val else "Unclassified"
    return "Unclassified"

def load_id_to_species_and_genus(tax_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    id2sp, id2gen = {}, {}
    with tax_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) < 2:
                parts = ln.split()
                if len(parts) < 2:
                    continue
            seq_id, lineage = parts[0].strip(), parts[1].strip()
            id2sp[seq_id]  = _parse_rank_from_lineage(lineage, "s__")
            id2gen[seq_id] = _parse_rank_from_lineage(lineage, "g__")
    return id2sp, id2gen


# ---------- heatmap ----------
def build_group_order_by_species(ids: list[str], id2sp: dict[str, str]):
    species_order: dict[str, list[str]] = {}
    for sid in ids:
        sp = id2sp.get(sid, "Unclassified")
        species_order.setdefault(sp, []).append(sid)
    ordered_species = sorted(species_order.keys())
    ordered_ids: list[str] = []
    species_slices: list[tuple[str, int, int]] = []
    start = 0
    for sp in ordered_species:
        members = species_order[sp]
        ordered_ids.extend(members)
        end = start + len(members)
        species_slices.append((sp, start, end))
        start = end
    oldpos = {sid: i for i, sid in enumerate(ids)}
    newpos = [oldpos[sid] for sid in ordered_ids]
    return ordered_ids, species_slices, newpos

def plot_grouped_heatmap(
    square_tsv: Path,
    taxonomy_tsv: Path,
    out_png: Path,
    threshold: float,
    px_per_cell: float,
) -> None:
    mat, ids = load_identity_square(square_tsv)
    n = len(ids)
    id2sp, _ = load_id_to_species_and_genus(taxonomy_tsv)

    _, species_slices, newpos = build_group_order_by_species(ids, id2sp)
    mat = mat[np.ix_(newpos, newpos)]

    tick_positions, species_labels = [], []
    for sp, s, e in species_slices:
        tick_positions.append((s + e) / 2.0 - 0.5)
        species_labels.append(sp)

    dpi = choose_dpi(n)
    width_px = max(1500, int(n * px_per_cell))
    figsize = (width_px / dpi, width_px / dpi)

    norm = mcolors.Normalize(vmin=threshold, vmax=100.0, clip=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(
        mat, interpolation="nearest", cmap="viridis",
        norm=norm, aspect="equal", origin="upper"
    )

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(species_labels, fontsize=8)

    ax.set_xticks([])
    ax.tick_params(axis="x", which="both", length=0)

    tx = blended_transform_factory(ax.transData, ax.transAxes)
    S = len(species_labels)
    fs = max(6, min(10, int(2000 / max(1, S))))

    for x, lab in zip(tick_positions, species_labels):
        ax.text(
            x, 0.0, lab, rotation=45, rotation_mode='anchor',
            ha="right", va="top", fontsize=fs, transform=tx, clip_on=False
        )

    for _, s, e in species_slices:
        ax.axvline(s - 0.5, linewidth=0.5, color="white", alpha=0.6)
        ax.axvline(e - 0.5, linewidth=0.5, color="white", alpha=0.6)
        ax.axhline(s - 0.5, linewidth=0.5, color="white", alpha=0.6)
        ax.axhline(e - 0.5, linewidth=0.5, color="white", alpha=0.6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Pairwise identity (%)", fontsize=9)

    plt.tight_layout()
    ensure_outdir(out_png)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ---------- histogram prep ----------
def _read_long_tsv(long_tsv: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            long_tsv, sep=r"\s+|\t", engine="python",
            usecols=["query_id", "target_id", "identity"]
        )
        df = df.rename(columns={"query_id": "q", "target_id": "t", "identity": "id"})
    except Exception:
        df = pd.read_csv(long_tsv, sep=r"\s+|\t", engine="python", header=None, usecols=[0, 1, 2])
        df.columns = ["q", "t", "id"]
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("float32")
    df = df[np.isfinite(df["id"])]
    df = df[(df["id"] >= 0.0) & (df["id"] <= 100.0)]
    return df

def _parse_names_arg(names: str | None, names_file: Path | None) -> list[str]:
    items: list[str] = []
    if names:
        items.extend([g.strip() for g in names.split(",") if g.strip()])
    if names_file:
        for line in names_file.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                items.append(s)
    # de-dup (preserve order)
    seen = set()
    uniq: list[str] = []
    for g in items:
        if g not in seen:
            uniq.append(g); seen.add(g)
    return uniq

def compute_hist_data(
    long_tsv: Path,
    taxonomy_tsv: Path,
    level: str,         # "genus" or "species"
    names: list[str],
    drop_self: bool,
):
    if len(names) < 1:
        raise ValueError("Provide at least one category via --names/--names-file.")
    id2sp, id2gen = load_id_to_species_and_genus(taxonomy_tsv)
    mapper = id2gen if level == "genus" else id2sp

    df = _read_long_tsv(long_tsv)
    if drop_self:
        df = df[df["q"] != df["t"]]

    qcat = df["q"].map(mapper).fillna("Unclassified")
    tcat = df["t"].map(mapper).fillna("Unclassified")
    names_set = set(names)

    # keep pairs where at least one endpoint is in the requested set
    mask = qcat.isin(names_set) | tcat.isin(names_set)
    df = df[mask]; qcat = qcat[mask]; tcat = tcat[mask]
    if df.empty:
        return None

    # categories: intra:<name> for each requested name, plus "inter"
    categories = [f"intra:{g}" for g in names] + ["inter"]
    cat_to_vals = {cat: [] for cat in categories}

    for (c1, c2, val) in zip(qcat.values, tcat.values, df["id"].values):
        v = float(val)
        if (c1 in names_set) and (c2 in names_set):
            if c1 == c2:
                cat_to_vals[f"intra:{c1}"].append(v)
            else:
                cat_to_vals["inter"].append(v)
        else:
            cat_to_vals["inter"].append(v)

    all_vals = np.concatenate([np.asarray(vs, dtype=np.float32) for vs in cat_to_vals.values() if len(vs) > 0]) \
               if any(len(vs) > 0 for vs in cat_to_vals.values()) else np.array([], dtype=np.float32)
    if all_vals.size == 0:
        return None

    vmin = float(np.nanmin(all_vals)); vmax = float(np.nanmax(all_vals))
    if vmax == vmin:
        vmin -= 0.5; vmax += 0.5
    pad = 0.02 * (vmax - vmin)
    vmin -= pad; vmax += pad
    span = vmax - vmin
    n_bins = int(np.clip(round(span / 0.5), 20, 200))  # ~0.5% per bin

    bins = np.linspace(vmin, vmax, n_bins + 1)
    widths = np.diff(bins)
    centers = bins[:-1] + widths / 2.0

    labels, rows = [], []
    for cat in categories:
        labels.append(cat)
        vals = np.array(cat_to_vals[cat], dtype=np.float32)
        if vals.size == 0:
            rows.append(np.zeros(n_bins, dtype=np.float32))
        else:
            h, _ = np.histogram(vals, bins=bins)
            rows.append(h.astype(np.float32))

    counts = np.vstack(rows)  # (C, B)
    return {"labels": labels, "counts": counts, "centers": centers, "widths": widths, "vmin": vmin, "vmax": vmax}


# ---------- CONSISTENT PALETTE ----------
def make_category_palette(labels: list[str],
                          inter_color: str = INTER_COLOR_DEFAULT) -> dict[str, str]:
    """
    Build a stable color map:
      - intra:<name> categories get colors from tab10 in order
      - "inter" gets a neutral gray
    """
    cmap = plt.get_cmap("tab10")
    palette: dict[str, str] = {}
    intra_labels = [lab for lab in labels if lab.startswith("intra:")]
    for i, lab in enumerate(intra_labels):
        palette[lab] = mcolors.to_hex(cmap(i % cmap.N))
    if "inter" in labels:
        palette["inter"] = inter_color
    return palette


# ---------- histogram plotting (uses shared palette) ----------
def plot_stacked_histogram_from_data(
    hist_data, out_png: Path, edge_color: str, edge_lw: float, palette: dict[str, str]
):
    labels = hist_data["labels"]; counts = hist_data["counts"]
    centers = hist_data["centers"]; widths = hist_data["widths"]
    vmin = hist_data["vmin"]; vmax = hist_data["vmax"]

    plt.figure(figsize=(9, 5), dpi=150)
    bottom = np.zeros(counts.shape[1], dtype=np.float32)
    for i, lab in enumerate(labels):
        plt.bar(
            centers, counts[i], bottom=bottom, width=widths, align="center",
            label=lab, color=palette.get(lab, None),
            edgecolor=edge_color, linewidth=edge_lw
        )
        bottom += counts[i]

    plt.xlabel("Pairwise identity (%)")
    plt.ylabel("Count")
    plt.title("Pairwise identity distribution by origin (stacked)")
    plt.xlim(vmin, vmax)
    plt.legend(title="Category", fontsize=8)
    plt.tight_layout()
    ensure_outdir(out_png)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_overlaid_histogram_from_data(
    hist_data, out_png: Path, overlay_alpha: float, edge_color: str, edge_lw: float, palette: dict[str, str]
):
    labels = hist_data["labels"]; counts = hist_data["counts"]
    centers = hist_data["centers"]; widths = hist_data["widths"]
    vmin = hist_data["vmin"]; vmax = hist_data["vmax"]

    plt.figure(figsize=(9, 5), dpi=150)
    # draw "inter" first so intra bars are visible
    order = list(range(len(labels)))
    if "inter" in labels:
        inter_idx = labels.index("inter")
        order.remove(inter_idx); order = [inter_idx] + order

    for i in order:
        lab = labels[i]
        plt.bar(
            centers, counts[i], width=widths, align="center",
            label=lab, alpha=overlay_alpha,
            color=palette.get(lab, None),
            edgecolor=edge_color, linewidth=edge_lw
        )

    plt.xlabel("Pairwise identity (%)")
    plt.ylabel("Count")
    plt.title("Pairwise identity distribution by origin (overlaid)")
    plt.xlim(vmin, vmax)
    plt.legend(title="Category", fontsize=8)
    plt.tight_layout()
    ensure_outdir(out_png)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# ---------- PCoA utilities ----------
def identities_to_distance(mat: np.ndarray, mode: str = "one_minus_fraction") -> np.ndarray:
    """
    Convert identity percentages to distances.
    mode:
      - 'one_minus_fraction': D = 1 - (identity/100)
      - 'hundred_minus_percent': D = 100 - identity
    Returns a square distance matrix (float32).
    """
    if mode == "one_minus_fraction":
        D = 1.0 - (mat.astype(np.float32) / 100.0)
    elif mode == "hundred_minus_percent":
        D = 100.0 - mat.astype(np.float32)
    else:
        raise ValueError(f"Unknown distance mode: {mode}")
    # force zeros on diagonal; clip tiny negatives that may arise numerically
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0
    return D


def pcoa_from_distance(D: np.ndarray, n_components: int = 2):
    """
    Classical MDS / PCoA on a square distance matrix D.
    Returns (coords, eigenvalues, explained_variance), where:
      - coords: (n, k) principal coordinates
      - eigenvalues: (n,) sorted descending
      - explained_variance: (k,) fraction of variance per axis
    """
    if D.shape[0] != D.shape[1]:
        raise ValueError(f"Distance matrix must be square, got {D.shape}")
    n = D.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 points for PCoA.")

    # Double-centering: B = -0.5 * J * D^2 * J
    J = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    D2 = (D.astype(np.float64)) ** 2
    B = -0.5 * (J @ D2 @ J)

    # Eigen-decomposition
    evals, evecs = np.linalg.eigh(B)  # returns ascending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Keep positive eigenvalues only
    pos = evals > 0
    if not np.any(pos):
        raise ValueError("No positive eigenvalues in PCoA; check your distance matrix.")
    evals_pos = evals[pos]
    evecs_pos = evecs[:, pos]

    k = min(n_components, evecs_pos.shape[1])
    lam = evals_pos[:k]
    vecs = evecs_pos[:, :k]
    coords = vecs * np.sqrt(lam)

    explained = lam / np.sum(evals_pos)
    return coords.astype(np.float32), evals.astype(np.float32), explained.astype(np.float32)


def _make_simple_palette(labels: list[str]) -> dict[str, str]:
    """
    Stable palette for arbitrary category labels.
    Uses tab20 then repeats if needed.
    """
    cmap = plt.get_cmap("tab20")
    palette = {}
    for i, lab in enumerate(labels):
        palette[lab] = mcolors.to_hex(cmap(i % cmap.N))
    return palette


# ---------- PCoA utilities ----------
def identities_to_distance(mat: np.ndarray, mode: str = "one_minus_fraction") -> np.ndarray:
    """
    Convert identity percentages to distances.
    mode:
      - 'one_minus_fraction': D = 1 - (identity/100)
      - 'hundred_minus_percent': D = 100 - identity
    Returns a square distance matrix (float32).
    """
    if mode == "one_minus_fraction":
        D = 1.0 - (mat.astype(np.float32) / 100.0)
    elif mode == "hundred_minus_percent":
        D = 100.0 - mat.astype(np.float32)
    else:
        raise ValueError(f"Unknown distance mode: {mode}")
    # force zeros on diagonal; clip tiny negatives that may arise numerically
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0
    return D


def pcoa_from_distance(D: np.ndarray, n_components: int = 2):
    """
    Classical MDS / PCoA on a square distance matrix D.
    Returns (coords, eigenvalues, explained_variance), where:
      - coords: (n, k) principal coordinates
      - eigenvalues: (n,) sorted descending
      - explained_variance: (k,) fraction of variance per axis
    """
    if D.shape[0] != D.shape[1]:
        raise ValueError(f"Distance matrix must be square, got {D.shape}")
    n = D.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 points for PCoA.")

    # Double-centering: B = -0.5 * J * D^2 * J
    J = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    D2 = (D.astype(np.float64)) ** 2
    B = -0.5 * (J @ D2 @ J)

    # Eigen-decomposition
    evals, evecs = np.linalg.eigh(B)  # returns ascending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Keep positive eigenvalues only
    pos = evals > 0
    if not np.any(pos):
        raise ValueError("No positive eigenvalues in PCoA; check your distance matrix.")
    evals_pos = evals[pos]
    evecs_pos = evecs[:, pos]

    k = min(n_components, evecs_pos.shape[1])
    lam = evals_pos[:k]
    vecs = evecs_pos[:, :k]
    coords = vecs * np.sqrt(lam)

    explained = lam / np.sum(evals_pos)
    return coords.astype(np.float32), evals.astype(np.float32), explained.astype(np.float32)


def _make_simple_palette(labels: list[str]) -> dict[str, str]:
    """
    Stable, colorblind-safe palette (Okabe–Ito) for up to 8 groups.
    Falls back to tab20 if more are needed.
    """
    okabe_ito = [
        "#0072B2",  # blue
        "#D55E00",  # vermilion
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#F0E442",  # yellow
        "#56B4E9",  # sky blue
        "#000000",  # black
        "#E69F00",  # orange
    ]
    if len(labels) <= len(okabe_ito):
        colors = okabe_ito[:len(labels)]
    else:
        cmap = plt.get_cmap("tab20")
        colors = [mcolors.to_hex(cmap(i % cmap.N)) for i in range(len(labels))]
    return {lab: colors[i] for i, lab in enumerate(labels)}



def plot_pcoa_from_square(
    square_tsv: Path,
    taxonomy_tsv: Path,
    out_png: Path,
    color_level: str = "species",           # 'species' or 'genus'
    distance_mode: str = "one_minus_fraction",
    names: list[str] | None = None,         # optional filter (normalized later as-is)
    out_coords_csv: Path | None = None,
    label_points: bool = False,
) -> None:
    """
    Run PCoA from the square identity matrix and plot colored by genus/species.
    If 'names' is provided, keep only IDs whose category is in 'names'.
    """
    # Load identities and IDs in their current order
    mat, ids = load_identity_square(square_tsv)
    id2sp, id2gen = load_id_to_species_and_genus(taxonomy_tsv)

    # Choose mapper
    if color_level == "genus":
        mapper = id2gen
    elif color_level == "species":
        mapper = id2sp
    else:
        raise ValueError("--pcoa-color-level must be 'genus' or 'species'.")

    # Categories for each id
    cats = [mapper.get(sid, "Unclassified") for sid in ids]

    # Optional filtering by names (exact match to mapper output)
    if names:
        names_set = set(names)
        keep = [i for i, c in enumerate(cats) if c in names_set]
        if len(keep) == 0:
            # Save an empty plot with a message for consistency with your style
            plt.figure(figsize=(7, 6), dpi=150)
            plt.title("No sequences for requested categories in PCoA")
            plt.tight_layout()
            ensure_outdir(out_png)
            plt.savefig(out_png, bbox_inches="tight")
            plt.close()
            return
        ids = [ids[i] for i in keep]
        cats = [cats[i] for i in keep]
        mat = mat[np.ix_(keep, keep)]

    # Convert to distances and run PCoA
    D = identities_to_distance(mat, mode=distance_mode)
    coords, evals, explained = pcoa_from_distance(D, n_components=2)

    # Build palette for the present categories
    present = sorted(set(cats))
    palette = _make_simple_palette(present)

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=150)
    for cat in present:
        idx = [i for i, c in enumerate(cats) if c == cat]
        xy = coords[idx, :]
        ax.scatter(xy[:, 0], xy[:, 1], s=20, alpha=0.9, label=cat, color=palette[cat])

    if label_points and len(ids) <= 200:
        # avoid clutter for large n
        for (x, y), lab in zip(coords, ids):
            ax.text(x, y, lab, fontsize=6, va="center", ha="left")

    ax.set_xlabel(f"PCoA1 ({explained[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PCoA2 ({explained[1]*100:.1f}% var.)")
    ax.set_title(f"PCoA of pairwise distances (colored by {color_level})")
    ax.axhline(0, lw=0.5, color="#999", alpha=0.5)
    ax.axvline(0, lw=0.5, color="#999", alpha=0.5)

    # Reasonable aspect and legend outside
    ax.set_aspect("auto")
    leg = ax.legend(title=color_level.capitalize(), fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    ensure_outdir(out_png)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Optional CSV of coordinates
    if out_coords_csv:
        ensure_outdir(out_coords_csv)
        df = pd.DataFrame({
            "id": ids,
            "category": cats,
            "PCoA1": coords[:, 0],
            "PCoA2": coords[:, 1],
        })
        # add explained variance as metadata-like extra rows (prefixed)
        df.to_csv(out_coords_csv, index=False)


def plot_pcoa_from_square(
    square_tsv: Path,
    taxonomy_tsv: Path,
    out_png: Path,
    color_level: str = "species",           # 'species' or 'genus'
    distance_mode: str = "one_minus_fraction",
    names: list[str] | None = None,         # optional filter (normalized later as-is)
    out_coords_csv: Path | None = None,
    label_points: bool = False,
    label_max: int = 200,
    label_size: float = 6.0,
) -> None:
    """
    Run PCoA from the square identity matrix and plot colored by genus/species.
    If 'names' is provided, keep only IDs whose category is in 'names'.
    """
    # Load identities and IDs in their current order
    mat, ids = load_identity_square(square_tsv)
    id2sp, id2gen = load_id_to_species_and_genus(taxonomy_tsv)

    # Choose mapper
    if color_level == "genus":
        mapper = id2gen
    elif color_level == "species":
        mapper = id2sp
    else:
        raise ValueError("--pcoa-color-level must be 'genus' or 'species'.")

    # Categories for each id
    cats = [mapper.get(sid, "Unclassified") for sid in ids]

    # Optional filtering by names (exact match to mapper output)
    if names:
        names_set = set(names)
        keep = [i for i, c in enumerate(cats) if c in names_set]
        if len(keep) == 0:
            # Save an empty plot with a message for consistency with your style
            plt.figure(figsize=(7, 6), dpi=150)
            plt.title("No sequences for requested categories in PCoA")
            plt.tight_layout()
            ensure_outdir(out_png)
            plt.savefig(out_png, bbox_inches="tight")
            plt.close()
            return
        ids = [ids[i] for i in keep]
        cats = [cats[i] for i in keep]
        mat = mat[np.ix_(keep, keep)]

    # Convert to distances and run PCoA
    D = identities_to_distance(mat, mode=distance_mode)
    coords, evals, explained = pcoa_from_distance(D, n_components=2)

    # Build palette for the present categories
    present = sorted(set(cats))
    palette = _make_simple_palette(present)

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=150)
    for cat in present:
        idx = [i for i, c in enumerate(cats) if c == cat]
        xy = coords[idx, :]
        ax.scatter(xy[:, 0], xy[:, 1], s=20, alpha=0.9, label=cat, color=palette[cat])

    if label_points and len(ids) <= 200:
        # avoid clutter for large n
        for (x, y), lab in zip(coords, ids):
            ax.text(x, y, lab, fontsize=6, va="center", ha="left")

    ax.set_xlabel(f"PCoA1 ({explained[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PCoA2 ({explained[1]*100:.1f}% var.)")
    ax.set_title(f"PCoA of pairwise distances (colored by {color_level})")
    ax.axhline(0, lw=0.5, color="#999", alpha=0.5)
    ax.axvline(0, lw=0.5, color="#999", alpha=0.5)

    # Optional labels (sequence IDs)
    if label_points:
        if len(ids) > label_max:
            # avoid wall-of-text; tell the user how many were suppressed
            ax.text(0.99, 0.01,
                    f"Labels suppressed ({len(ids)}>{label_max}). "
                    f"Use --pcoa-label-max to override.",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        else:
            # simple text labels with a faint outline for readability
            try:
                from matplotlib.patheffects import Stroke, Normal
                pe = [Stroke(linewidth=2.0, foreground="white", alpha=0.9), Normal()]
            except Exception:
                pe = None
            for (x, y), lab in zip(coords, ids):
                ax.text(x, y, lab, fontsize=label_size, va="center", ha="left",
                        path_effects=pe)

    # Reasonable aspect and legend outside
    ax.set_aspect("auto")
    leg = ax.legend(title=color_level.capitalize(), fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    ensure_outdir(out_png)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Optional CSV of coordinates
    if out_coords_csv:
        ensure_outdir(out_coords_csv)
        df = pd.DataFrame({
            "id": ids,
            "category": cats,
            "PCoA1": coords[:, 0],
            "PCoA2": coords[:, 1],
        })
        # add explained variance as metadata-like extra rows (prefixed)
        df.to_csv(out_coords_csv, index=False)


# ---------- argparse ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="create_plots",
        description="Heatmap (species-grouped) + stacked & overlaid histograms (genus/species categories) of pairwise identities."
    )
    # Inputs
    p.add_argument("--square", type=Path, required=True, help="Square identity matrix TSV (ids as header/rows).")
    p.add_argument("--taxonomy", type=Path, required=True, help="Two-column taxonomy TSV (<id>\\t<k__...;...;s__...>).")
    p.add_argument("--long", type=Path, required=True, help="Long identities TSV (query_id, target_id, identity).")

    # Category level + names (new API)
    p.add_argument("--level", choices=["genus", "species"], default="genus",
                   help="Category level for histograms (default: genus).")
    p.add_argument("--names", help="Comma-separated categories at the chosen level (e.g. 'Bacteroides,Bacillus' or 'Bacillus subtilis,Escherichia coli').")
    p.add_argument("--names-file", type=Path, help="File with one category per line (comments with # ignored).")

    # Legacy flags (kept as hidden aliases)
    p.add_argument("--genera", help=argparse.SUPPRESS)
    p.add_argument("--genera-file", type=Path, help=argparse.SUPPRESS)

    # Outputs
    p.add_argument("--out-heatmap", type=Path, required=True, help="Output PNG for heatmap.")
    p.add_argument("--out-hist-stacked", type=Path, required=True, help="Output PNG for stacked histogram.")
    p.add_argument("--out-hist-overlay", type=Path, required=True, help="Output PNG for overlaid histogram.")

    # Tuning
    p.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT)
    p.add_argument("--px-per-cell", type=float, default=PX_PER_CELL_DEFAULT)
    p.add_argument("--overlay-alpha", type=float, default=OVERLAY_ALPHA_DEFAULT)
    p.add_argument("--edge-color", default=EDGE_COLOR_DEFAULT)
    p.add_argument("--edge-linewidth", type=float, default=EDGE_LINEWIDTH_DEFAULT)
    p.add_argument("--inter-color", default=INTER_COLOR_DEFAULT, help="Color for 'inter' category.")
    p.add_argument("--drop-self", action="store_true", help="Drop self pairs (q==t) before histograms.")


    # PCoA outputs & options
    p.add_argument("--out-pcoa", type=Path, help="Output PNG for PCoA scatter.")
    p.add_argument("--out-pcoa-csv", type=Path, help="Optional CSV to save PCoA coordinates.")
    p.add_argument("--pcoa-color-level", choices=["genus", "species"], default="species",
                   help="Taxonomic level to color PCoA points (default: species).")
    p.add_argument("--pcoa-distance", choices=["one_minus_fraction", "hundred_minus_percent"],
                   default="one_minus_fraction",
                   help="How to convert identities to distances for PCoA (default: 1 - identity/100).")
    p.add_argument("--pcoa-label-points", action="store_true", help="Label points with sequence IDs on the PCoA.")

    # in parse_args()
    p.add_argument("--pcoa-label-max", type=int, default=200,
               help="Maximum number of points to label to avoid clutter (default: 200).")
    p.add_argument("--pcoa-label-size", type=float, default=6.0,
               help="Font size for PCoA point labels (default: 6).")

    return p.parse_args()


# ---------- main ----------
def main() -> None:
    args = parse_args()

    # Heatmap
    plot_grouped_heatmap(
        square_tsv=args.square,
        taxonomy_tsv=args.taxonomy,
        out_png=args.out_heatmap,
        threshold=args.threshold,
        px_per_cell=args.px_per_cell,
    )

    # Category names (new flags or legacy aliases)
    names = _parse_names_arg(args.names or args.genera, args.names_file or args.genera_file)
    if not names:
        raise SystemExit("No categories provided. Use --names/--names-file (or legacy --genera/--genera-file).")

    # Histogram data (once)
    hist_data = compute_hist_data(
        long_tsv=args.long,
        taxonomy_tsv=args.taxonomy,
        level=args.level,
        names=names,
        drop_self=args.drop_self,
    )
    if hist_data is None:
        for out_png, title in [
            (args.out_hist_stacked,  "No pairs found for the requested categories (stacked)"),
            (args.out_hist_overlay, "No pairs found for the requested categories (overlaid)"),
        ]:
            plt.figure(figsize=(9, 5), dpi=150)
            plt.title(title)
            plt.tight_layout()
            ensure_outdir(out_png)
            plt.savefig(out_png, bbox_inches="tight")
            plt.close()
        return

    # Single palette reused for both plots (ensures consistent colors)
    palette = make_category_palette(hist_data["labels"], inter_color=args.inter_color)

    # Stacked + Overlaid
    plot_stacked_histogram_from_data(
        hist_data, args.out_hist_stacked,
        edge_color=args.edge_color, edge_lw=args.edge_linewidth, palette=palette
    )
    plot_overlaid_histogram_from_data(
        hist_data, args.out_hist_overlay,
        overlay_alpha=args.overlay_alpha,
        edge_color=args.edge_color, edge_lw=args.edge_linewidth, palette=palette
    )

        # ----- PCoA (optional) -----
        # ----- PCoA (optional) -----
    if args.out_pcoa:
        # Reuse parsed names (may be empty)
        names_for_pcoa = names if names else None
        try:
            plot_pcoa_from_square(
                square_tsv=args.square,
                taxonomy_tsv=args.taxonomy,
                out_png=args.out_pcoa,
                color_level=args.pcoa_color_level,
                distance_mode=args.pcoa_distance,
                names=names if names else None,
                out_coords_csv=getattr(args, "out_pcoa_csv", None),
                label_points=args.pcoa_label_points,
                label_max=args.pcoa_label_max,
                label_size=args.pcoa_label_size,
    )
        except Exception as e:
            # Produce a diagnostic image instead of failing hard (keeps your UX consistent)
            plt.figure(figsize=(7, 6), dpi=150)
            plt.title(f"PCoA failed: {e}")
            plt.tight_layout()
            ensure_outdir(args.out_pcoa)
            plt.savefig(args.out_pcoa, bbox_inches="tight")
            plt.close()



if __name__ == "__main__":
    main()

