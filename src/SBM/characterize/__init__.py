"""Downstream structural + BLAST characterization of designed sequences.

Given a FASTA of sequences (designs and/or natural controls), this
subpackage predicts a 3-D structure per sequence (ESMFold, single-
sequence — the right tool for de novo designs where an MSA is
ill-defined), compares each model to the two reference folds by
TM-align, and BLASTs the designs against SwissProt + per-family
databases. See ``docs/CHARACTERIZE.md`` and the CLIs under
``scripts/characterize/``.

Import-light by design: ``fold`` lazy-imports torch/transformers inside
its runtime functions so the module (and ``tmscore``/``blast``/``summary``)
imports cleanly in the CPU ``.venv`` for the compare/blast/summary/render
stages and in tests, while the GPU fold stage runs under ``bioM3_env``.
"""
