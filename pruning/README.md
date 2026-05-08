## Building Pruned Potts Models with sBM

Here, we include scripts to generate pruning masks based on rank-ordered statistics of the input alignment. Each Potts parameter family — couplings $J$ and fields $h$ — has its own set of strategies; the two are independent and can be mixed and matched.

**Couplings ($J$) strategies — produce shape $(L, L, q, q)$ binary masks:**

1. $F_{ij}^{ab}$ (`fij`), the pairwise frequencies of amino acid *a* at position *i* with amino acid *b* at position *j*.
2. $C_{ij}^{ab}$ (`cij`), the pairwise correlations for amino acid *a* at position *i* with amino acid *b* at position *j*, equivalent to $F_{ij}^{ab} - F_i^a F_j^b$.
3. $\tilde{C}_{ij}^{ab}$ (`sca`), the pairwise **conserved** correlations for amino acid *a* at position *i* with amino acid *b* at position *j*. This is defined as $\phi_i^a \phi_j^b C_{ij}^{ab}$. See [Rivoire, Reynolds and Ranganathan 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004817) for additional details.

**Fields ($h$) strategies — produce shape $(L, q)$ binary masks:**

4. $F_i^a$ (`fia`), the per-site, per-amino-acid frequencies (parallel to `fij` for couplings).
5. $D_i^a$ (`dia`), the per-site, per-amino-acid **KL divergence** of the observed distribution from a background — i.e., the per-position conservation signal that pySCA computes internally for the SCA matrix (parallel to `sca` for couplings). By default the background is gap-corrected; pass `--Dia-prior uniform` to use a uniform 21-bin prior instead.

### Dependencies

Mask creation based on $\tilde{C}_{ij}^{ab}$ introduces an additional package dependency relative to the main sBM software.

| Package Name | Tested Version | Existing sBM Dependency? |
|--------------|---------|------------------|
|python        |3.11.11  | Yes
|numpy         |2.4.2    | Yes
|scipy         |1.17.1   |Yes
|[pySCA](https://github.com/ranganathanlab/pySCA)|7.0|**No**|

Mask creation based on $F_{ij}^{ab}$ and $C_{ij}^{ab}$ can proceed without installing pySCA. To create pruning masks based on $\tilde{C}_{ij}^{ab}$, follow the installation instructions in the pySCA repository.

### Required Inputs

A multiple sequence alignment for your family of interest in FASTA format or as a numerical alignment in NumPy format (M sequences x L positions, 0=gaps) or in MATLAB format under the variable name "align" (MxL, 1=gaps).

This is passed in with the `--alg` flag. File format is inferred.

### Optional Parameters
| Parameter Flag | Description | Default |
|-|-|-|
|`--theta`| Similiarity threshold to use for sequence reweighting. | `0.7`
|`--lbda`| Pseudocount to use when calculating alignment statistics for SCA. | `0.03`
|`--strategies`| How to rank parameters for pruning. J: `"fij"`, `"cij"`, `"sca"`. h: `"fia"`, `"dia"`. Mix and match: any combination is permitted in a single run. | `"fij" "cij" "sca"`
|`--Dia-prior`| Background distribution for the `dia` strategy. `"gap-corrected"` uses the alignment's estimated gap rate plus standard 20-AA frequencies; `"uniform"` uses `np.ones(21)/21`. | `"gap-corrected"`
|`--ext`| File format to save output matrices in. Options are `.npy` or `.mat`. Applies to J masks only; h masks are always saved as `.npy`. | `.npy`
|`--label`| Any unique identifier or information to include in the name of the output file, for example the protein family name. | `CM`
|`--path`| Parent directory under which a per-run subdir (`<YYYY-MM-DD>_<label>_<idx>/`) is created. All masks + manifest sidecars from one invocation land in that subdir. | `pruning/masks/` (resolved relative to `build_mask.py`)
|`--percent-J`| Proportion of **couplings** to exclude (as a percent). Used by the `fij`, `cij`, and `sca` strategies. Multiple values produce one J mask per percent. | `95`
|`--percent-h`| Proportion of **fields** to exclude (as a percent). Used by the `fia` and `dia` strategies. Multiple values produce one h mask per percent. | `95`

### Usage

The mask generation script can be run from the command line (with all optional parameters specified) as:

```
> python build_mask.py --alg $FULL_CM_ALG \
        --theta 0.7 --lbda 0.03 \
        --strategies "fij" "cij" "sca" "fia" "dia" \
        --ext ".npy" --label "CM" \
        --path "./prune_output" \
        --percent-J 95 98 --percent-h 95 98
```

The script prints `Run dir: <abs path>` to stdout; downstream scripts scrape that line to locate the generated masks (see `CM_example.sh` for the pattern).

### Outputs

Each invocation creates a fresh per-run subdir of `--path`, named `<YYYY-MM-DD>_<label>_<idx>/` (the index auto-increments across sibling dirs). All output files for that invocation — one mask per (strategy, percent) combination, plus a `.manifest.json` sidecar per mask — land inside that subdir. Mask filenames follow:

`<output-path>/<YYYY-MM-DD>_<label>_<idx>/<percent>_<strategy>_<label>_SeqW_<theta>.<ext>`

### Inference with Pruning Mask

With the generated mask(s), a Potts model can be inferred in three ways:
1. If calling `sbm.SBM` directly, include any combination of:
   - `"Pruning": True` and `"Pruning Mask Couplings": "path/to/J_mask.npy"` (J pruning),
   - `"Pruning Fields": True` and `"Pruning Mask Fields": "path/to/h_mask.npy"` (h pruning),
   in your options dictionary.
2. If running SBM through `scripts/train_sbm.py`, pass either or both of the command-line flags `--prune-J "path/to/J_mask.npy"` and `--prune-h "path/to/h_mask.npy"`.
3. If running SBM through `scripts/run_sbm.sh`, the same flags are forwarded: `--prune-J "path/to/J_mask.npy"` and `--prune-h "path/to/h_mask.npy"`.

### Example

The script `CM_example.sh` generates both a J pruning mask (SCA) and an h pruning mask (Dia) for the chorismate mutase family that exclude 98% of the parameters apiece, and then uses both masks to infer a Potts model.

It can be run from the command line as:
`bash CM_example.sh`
