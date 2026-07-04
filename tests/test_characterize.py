"""Unit tests for the characterize subpackage (pure-python, no GPU/BLAST).

Covers the error-prone parsing/partitioning glue: the round-robin sharder
(no overlap, full coverage), FASTA read + degap, the TMalign stdout parser
against a saved sample, single-chain PDB extraction, the BLAST tabular
parser + best-hit selection, and the fold-call / summary-merge logic.

    .venv/bin/python -m pytest tests/test_characterize.py -q
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from SBM.characterize import blast, fold, natural_cache, summary, tmscore


# ── Sharding ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n,ns", [(1, 1), (10, 3), (97, 8), (28140, 64), (5, 8)])
def test_shard_partition_covers_without_overlap(n: int, ns: int) -> None:
    seen: list[int] = []
    for t in range(ns):
        seen.extend(fold.shard_indices(n, ns, t))
    assert sorted(seen) == list(range(n))  # full coverage
    assert len(set(seen)) == len(seen)  # no overlap


def test_shard_bounds_validated() -> None:
    with pytest.raises(ValueError):
        fold.shard_indices(10, 3, 3)
    with pytest.raises(ValueError):
        fold.shard_indices(10, 0, 0)


def test_shard_records_roundtrips() -> None:
    recs = [(f"id{i}", "ACDE") for i in range(10)]
    got = [r for t in range(4) for r in fold.shard_records(recs, 4, t)]
    assert sorted(got) == sorted(recs)


# ── Resume (done_ids): robust to a change in shard count ─────────────────────


def _write_fold_cache(
    scores_dir: Path, structures_dir: Path, ids: list[str], n_shards: int,
    *, pdbs: set[str] | None = None,
) -> None:
    """Write a fold-score cache exactly as fold_sequences.py leaves it: one
    ``shard_<t>.tsv`` per shard with the round-robin rows, one ``<id>.pdb`` per
    folded id (``pdbs`` defaults to all ids)."""
    scores_dir.mkdir(parents=True, exist_ok=True)
    structures_dir.mkdir(parents=True, exist_ok=True)
    recs = [(rid, "ACDE") for rid in ids]
    for t in range(n_shards):
        lines = ["id\tgroup\tlength\tplddt_mean\tptm"]
        lines += [f"{rid}\tG\t4\t90.00\t0.9000"
                  for rid, _ in fold.shard_records(recs, n_shards, t)]
        (scores_dir / f"shard_{t}.tsv").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    for rid in (set(ids) if pdbs is None else pdbs):
        (structures_dir / f"{rid}.pdb").write_text("ATOM\n", encoding="utf-8")


def test_done_ids_finds_all_regardless_of_shard_layout(tmp_path: Path) -> None:
    ids = [f"seq{i}" for i in range(20)]
    sc, st = tmp_path / "fold_scores", tmp_path / "structures"
    _write_fold_cache(sc, st, ids, n_shards=7)
    assert fold.done_ids(sc, st) == set(ids)


def test_done_ids_resume_survives_shard_count_change(tmp_path: Path) -> None:
    # Regression: cache built with 7 shards, re-run with 3. Every record of every
    # new shard must already be "done" so nothing re-folds. The old per-shard
    # check (read only shard_<t>.tsv) re-folded ~all of them here.
    ids = [f"seq{i}" for i in range(20)]
    sc, st = tmp_path / "fold_scores", tmp_path / "structures"
    _write_fold_cache(sc, st, ids, n_shards=7)
    done = fold.done_ids(sc, st)
    recs = [(rid, "ACDE") for rid in ids]
    for new_shard in range(3):
        todo = [rid for rid, _ in fold.shard_records(recs, 3, new_shard)
                if rid not in done]
        assert todo == []


def test_done_ids_torn_write_is_not_done(tmp_path: Path) -> None:
    # A score row whose PDB is missing (torn/deleted) must be re-folded.
    ids = [f"seq{i}" for i in range(6)]
    sc, st = tmp_path / "fold_scores", tmp_path / "structures"
    _write_fold_cache(sc, st, ids, n_shards=2, pdbs=set(ids) - {"seq3"})
    assert fold.done_ids(sc, st) == set(ids) - {"seq3"}


def test_done_ids_missing_scores_dir(tmp_path: Path) -> None:
    assert fold.done_ids(tmp_path / "nope", tmp_path / "structures") == set()


# ── FASTA + degap ───────────────────────────────────────────────────────────


def test_read_fasta_multiline_and_id(tmp_path: Path) -> None:
    p = tmp_path / "x.fasta"
    p.write_text(">seqA desc here\nACDE\nFGHI\n>seqB\nKLMN\n", encoding="utf-8")
    recs = fold.read_fasta(p)
    assert recs == [("seqA", "ACDEFGHI"), ("seqB", "KLMN")]


def test_degap_and_canonical() -> None:
    assert fold.degap("A-C.d-E") == "ACDE"
    assert fold.is_canonical("ACDEFGHIKLMNPQRSTVWY")
    assert not fold.is_canonical("ACDX")  # X non-canonical
    assert not fold.is_canonical("")  # empty is not canonical


def test_mean_plddt_from_pdb_scale_100() -> None:
    # Two CA atoms with B-factors 90.00 and 70.00 (0–100 scale) -> mean 80.00.
    pdb = (
        "ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00 88.00           N\n"
        "ATOM      2  CA  ALA A   1      1.000   0.000   0.000  1.00 90.00           C\n"
        "ATOM      3  CA  GLY A   2      2.000   0.000   0.000  1.00 70.00           C\n"
    )
    assert math.isclose(fold.mean_plddt_from_pdb(pdb), 80.0, abs_tol=1e-9)
    assert math.isnan(fold.mean_plddt_from_pdb("HEADER only\n"))


def test_mean_plddt_from_pdb_scale_01_rescaled() -> None:
    # transformers 5.x writes pLDDT on a 0–1 scale; must be rescaled to 0–100.
    pdb = (
        "ATOM      2  CA  ALA A   1      1.000   0.000   0.000  1.00  0.90           C\n"
        "ATOM      3  CA  GLY A   2      2.000   0.000   0.000  1.00  0.70           C\n"
    )
    assert math.isclose(fold.mean_plddt_from_pdb(pdb), 80.0, abs_tol=1e-9)


# ── TMalign parsing + chain extraction ──────────────────────────────────────

_TMALIGN_SAMPLE = """
 *********************************************************************
 * TM-align (Version 20190822)                                       *
 *********************************************************************
Name of Chain_1: query.pdb
Name of Chain_2: ref.pdb
Length of Chain_1:   82 residues
Length of Chain_2:   91 residues

Aligned length=   78, RMSD=   3.21, Seq_ID=n_identical/n_aligned= 0.115
TM-score= 0.55123 (if normalized by length of Chain_1, i.e., LN=82, d0=2.94)
TM-score= 0.49876 (if normalized by length of Chain_2, i.e., LN=91, d0=3.20)
(":" denotes residue pairs of d < 5.0 A)
"""


def test_parse_tmalign_stdout() -> None:
    r = tmscore.parse_tmalign_stdout(_TMALIGN_SAMPLE)
    assert math.isclose(r.tm_query, 0.55123, abs_tol=1e-9)
    assert math.isclose(r.tm_ref, 0.49876, abs_tol=1e-9)
    assert math.isclose(r.rmsd, 3.21, abs_tol=1e-9)
    assert r.aligned_len == 78
    assert math.isclose(r.seq_id, 0.115, abs_tol=1e-9)


def test_parse_tmalign_stdout_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        tmscore.parse_tmalign_stdout("no useful lines here")


def test_extract_chain_filters(tmp_path: Path) -> None:
    src = tmp_path / "multi.pdb"
    src.write_text(
        "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00 50.0           C\n"
        "ATOM      2  CA  GLY B   1       1.0   0.0   0.0  1.00 50.0           C\n"  # other chain
        "HETATM    3  O   HOH A   2       2.0   0.0   0.0  1.00  0.0           O\n"  # hetatm
        "ATOM      4  CA  BSER A   3       3.0   0.0   0.0  1.00 50.0           C\n"  # altloc B
        "ATOM      5  CA  SER A   4       4.0   0.0   0.0  1.00 50.0           C\n"
        "ENDMDL\n"
        "ATOM      6  CA  ALA A   5       5.0   0.0   0.0  1.00 50.0           C\n"  # 2nd model
        "END\n",
        encoding="utf-8",
    )
    out = tmp_path / "chainA.pdb"
    n = tmscore.extract_chain(src, "A", out)
    assert n == 2  # residues 1 and 4 (B altloc, hetatm, other chain, 2nd model excluded)
    text = out.read_text(encoding="utf-8")
    assert " B " not in text and "HOH" not in text


# ── BLAST parsing + degap ───────────────────────────────────────────────────


def test_parse_blast_tsv_and_best_hit(tmp_path: Path) -> None:
    tsv = tmp_path / "hits.tsv"
    tsv.write_text(
        "q1\tsp_A\t45.0\t80\t5\t0\t1\t80\t1\t80\t1e-20\t120.5\t82\t91\t95\n"
        "q1\tsp_B\t30.0\t70\t10\t1\t1\t70\t1\t70\t1e-5\t60.0\t82\t85\t80\n"
        "q2\tsp_C\t99.0\t82\t0\t0\t1\t82\t1\t82\t1e-40\t180.0\t82\t82\t100\n",
        encoding="utf-8",
    )
    hits = blast.parse_blast_tsv(tsv)
    assert set(hits) == {"q1", "q2"}
    assert len(hits["q1"]) == 2
    bh = blast.best_hit(hits, "q1")
    assert bh.sseqid == "sp_A" and math.isclose(bh.bitscore, 120.5)
    assert blast.best_hit(hits, "q_missing") is None


def test_degap_fasta(tmp_path: Path) -> None:
    src = tmp_path / "aln.fasta"
    src.write_text(">a\nAC-DE\n.FG.\n>b\n----\n>c\nKLMN\n", encoding="utf-8")
    out = tmp_path / "degap.fasta"
    n = blast.degap_fasta(src, out)
    assert n == 2  # 'b' is all gaps -> dropped
    text = out.read_text(encoding="utf-8")
    assert ">a\nACDEFG\n" in text
    assert ">c\nKLMN\n" in text
    assert "-" not in text and "." not in text


# ── Summary merge ───────────────────────────────────────────────────────────


def test_build_summary_rows_design(tmp_path: Path) -> None:
    designed = tmp_path / "designed.tsv"
    designed.write_text(
        "chain\tstart_type\tE_A_polish\tE_B_polish\tE_tot_polish\n"
        "0\trandom\t-157.0\t-158.0\t-157.5\n",
        encoding="utf-8",
    )
    energies = summary.load_design_energies(designed)
    assert "design_chain0000" in energies
    assert math.isclose(energies["design_chain0000"]["E_A"], -157.0)

    fold_rows = [{"id": "design_chain0000", "group": "design", "length": "82",
                  "plddt_mean": "78.5", "ptm": "0.72"}]
    compare = {"design_chain0000": {"tm_ref_A": "0.62", "rmsd_A": "2.1",
                                    "tm_ref_B": "0.45", "rmsd_B": "3.4"}}

    class _Hit:  # duck-typed BlastHit
        def __init__(self, sseqid, pident, evalue):
            self.sseqid, self.pident, self.evalue = sseqid, pident, evalue

    rows = summary.build_summary_rows(
        fold_rows, compare,
        energies=energies,
        swissprot={"design_chain0000": _Hit("P12345", 41.2, 1e-9)},
        cmfam={"design_chain0000": _Hit("1ECM_A", 55.0, 1e-15)},
        ppicfam={},
        annotations={"P12345": "Chorismate mutase"},
    )
    r = rows[0]
    assert r["fold_call"] == "A"  # tm_A 0.62 > tm_B 0.45, both cross? only A>=0.5
    assert r["swissprot_top_hit"] == "P12345"
    assert r["swissprot_annotation"] == "Chorismate mutase"
    assert r["cmfam_top_hit"] == "1ECM_A"
    assert r["ppicfam_top_hit"] == ""  # no ppic hit
    assert math.isclose(float(r["delta_E"]), 1.0, abs_tol=1e-6)  # -157 - (-158)


def test_write_and_read_tsv_roundtrip(tmp_path: Path) -> None:
    rows = [{"sequence_id": "s1", "group": "design", "tm_A": "0.6"}]
    p = tmp_path / "out.tsv"
    summary.write_tsv(rows, ["sequence_id", "group", "tm_A"], p)
    back = summary.read_tsv(p)
    assert back[0]["sequence_id"] == "s1" and back[0]["tm_A"] == "0.6"


# ── natural TM-align cache ──────────────────────────────────────────────────


def test_ref_pair_key_deterministic_and_sensitive() -> None:
    k = natural_cache.ref_pair_key("aaa", "A", "bbb", "A")
    assert k == natural_cache.ref_pair_key("aaa", "A", "bbb", "A")  # deterministic
    assert len(k) == natural_cache.REFKEY_LEN
    # Any change to a reference sha or chain mints a new key.
    assert k != natural_cache.ref_pair_key("aaZ", "A", "bbb", "A")  # ref A content
    assert k != natural_cache.ref_pair_key("aaa", "B", "bbb", "A")  # ref A chain
    assert k != natural_cache.ref_pair_key("aaa", "A", "bbb", "B")  # ref B chain
    # A/B are distinct roles: swapping them is a different key.
    assert k != natural_cache.ref_pair_key("bbb", "A", "aaa", "A")


def _write_compare_tsv(path: Path, ids: list[str], group: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("id\tgroup\ttm_ref_A\ttm_ref_B\n")
        for rid in ids:
            fh.write(f"{rid}\t{group}\t0.60\t0.40\n")


def test_cache_covers_superset_and_partial(tmp_path: Path) -> None:
    tsv = tmp_path / "tm_vs_refs" / "key.tsv"
    _write_compare_tsv(tsv, ["n1", "n2", "n3"], "CM-natural")
    assert natural_cache.cache_covers(tsv, {"n1", "n2"})  # subset -> hit
    assert natural_cache.cache_covers(tsv, {"n1", "n2", "n3"})  # exact -> hit
    assert not natural_cache.cache_covers(tsv, {"n1", "n4"})  # missing id -> miss
    assert not natural_cache.cache_covers(tmp_path / "absent.tsv", {"n1"})  # no file
    assert not natural_cache.cache_covers(tsv, set())  # nothing required -> not a hit


def test_ids_in_fold_scores_unions_shards(tmp_path: Path) -> None:
    fs = tmp_path / "fold_scores"
    fs.mkdir(parents=True)
    (fs / "shard_0.tsv").write_text("id\tgroup\nn1\tCM\nn2\tCM\n", encoding="utf-8")
    (fs / "shard_1.tsv").write_text("id\tgroup\nn3\tCM\n", encoding="utf-8")
    assert natural_cache.ids_in_fold_scores(tmp_path) == {"n1", "n2", "n3"}


def test_merge_compare_tsvs_dedup_and_sort(tmp_path: Path) -> None:
    design = tmp_path / "design_compare.tsv"
    nat_a = tmp_path / "a.tsv"
    nat_b = tmp_path / "b.tsv"
    _write_compare_tsv(design, ["d2", "d1"], "design")
    _write_compare_tsv(nat_a, ["a1"], "CM-natural")
    _write_compare_tsv(nat_b, ["b1", "a1"], "PPIC-natural")  # a1 duplicates nat_a
    out = tmp_path / "structure_compare.tsv"
    n = natural_cache.merge_compare_tsvs([design, nat_a, nat_b], out)
    assert n == 4  # d1, d2, a1, b1 (a1 deduped)
    rows = summary.read_tsv(out)
    assert [r["id"] for r in rows] == ["a1", "b1", "d1", "d2"]  # sorted by id
    # First source to define an id wins: a1 came from nat_a (CM-natural).
    a1 = next(r for r in rows if r["id"] == "a1")
    assert a1["group"] == "CM-natural"


def test_write_meta_records_provenance(tmp_path: Path) -> None:
    meta_path = natural_cache.cache_meta(tmp_path / "sha8abcd", "refkey123456")
    meta = natural_cache.write_meta(
        meta_path, refkey="refkey123456",
        ref_a="data/1ECM.pdb", ref_a_sha256="aaa", chain_a="A",
        ref_b="data/1JNT.pdb", ref_b_sha256="bbb", chain_b="A",
        tmalign="pipeline/bin/TMalign", n_rows=1253, source_sha8="sha8abcd")
    assert meta_path.exists()
    import json
    on_disk = json.loads(meta_path.read_text(encoding="utf-8"))
    assert on_disk["refkey"] == "refkey123456"
    assert on_disk["n_rows"] == 1253
    assert on_disk["source_sha8"] == "sha8abcd"
    assert on_disk["ref_a_sha256"] == "aaa" and on_disk["ref_b_sha256"] == "bbb"
