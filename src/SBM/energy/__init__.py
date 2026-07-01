"""Energy of a sequence under one or two fitted Potts models.

Submodules are imported directly (the package ``__init__`` stays light, as
elsewhere in ``SBM``):

- :mod:`SBM.energy.model`    — :class:`PottsModel` + :func:`load_model`.
- :mod:`SBM.energy.potts`    — in-frame Potts energy (spec §2 base case).
- :mod:`SBM.energy.encoding` — residue ↔ integer conversions, gap stripping.
- :mod:`SBM.energy.hmm`      — profile-HMM alignment proposal (forward / Viterbi / FFBS).
- :mod:`SBM.energy.score`    — :func:`score_sequence` / :func:`score_two_models`.

See ``docs/two_model_progress.md`` (goal + status) and ``docs/POTTS_ALIGN.md``
(the production aligner) for the specification this implements.
"""
