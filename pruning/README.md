## Building Pruned Potts Models with sBM

Here, we include scripts to generate pruning masks based on rank-ordered statistics of the input alignment. Current options are:

1. $F_{ij}^{ab}$, the pairwise frequencies of amino acid *a* at position *i* with amino acid *b* at position *j*.
2. $C_{ij}^{ab}$, the pairwise correlations for amino acid *a* at position *i* with amino acid *b* at position *j*, equivalent to $F_{ij}^{ab} - F_i^a F_j^b$.
3. $\tilde{C}_{ij}^{ab}$, the pairwise **conserved** correlations for amino acid *a* at position *i* with amino acid *b* at position *j*. This is defined as $\phi_i^a \phi_j^b C_{ij}^{ab}$. See [Rivoire, Reynolds and Ranganathan 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004817) for additional details.

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
|`--strategies`| How to rank parameters for pruning. Options are `"fij"` for $F_{ij}^{ab}$ pruning, `"cij"` for $C_{ij}^{ab}$ pruning, and `"sca"` for $\tilde{C}_{ij}^{ab}$ pruning. Multiple options can be included in a single run (as is the default). | `"fij" "cij" "sca"`
|`--ext`| File format to save output matrices in. Options are `.npy` or `.mat`. | `.npy`
|`--label`| Any unique identifier or information to include in the name of the output file, for example the protein family name. | `CM`
|`--path`| Directory in which to save output files | `.` (directory in which the script is run)
|`--percent`| What proportion of parameters to exclude (as a percent). Multiple values can be included in a single run. | `95`

### Usage

The mask generation script can be run from the command line (with all optional parameters specified) as:

```
> python build_mask.py --alg $FULL_CM_ALG \
        --theta 0.7 --lbda 0.03 \
        --strategies "fij" "cij" "sca" \
        --ext ".npy" --label "CM" \
        --path "./prune_output" --percent 95 98
```

### Outputs

The mask generation script will output one file per combined strategy and percent provided in the specified output directory (or current directory, if none specified). The output file will follow the following naming format:

`<output-path>/<percent>_<strategy>_<label>_SeqW_<theta>.<ext>`

### Inference with Pruning Mask

With the generated mask, a Potts model can be inferred in two ways:
1. If calling `sbm.SBM` directly, include `Prune: True` and `Prune Mask Couplings: "path/to/prune_mask"` in your options dictionary.
2. If running SBM through a wrapper Python script, as in the demo `SBM-CM-family.py`, include the pruning mask as a command line flag `--prune "path/to/prune_mask"`.

### Example

The script `CM_example.sh` generates a pruning mask for the chorismate mutase family that excludes 98% of the parameters based on SCA, and then it uses the mask to infer a Potts model.

It can be run from the command line as:
`>zsh CM_example.sh`.
