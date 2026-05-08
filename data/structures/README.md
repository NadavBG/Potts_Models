# data/structures/

Crystal structures used for downstream analysis (e.g. ProteinMPNN
foldability scoring).

## 1ECM.pdb

E. coli chorismate mutase, anchor structure for the CM family.

- Source URL: https://files.rcsb.org/download/1ECM.pdb
- Downloaded: 2026-05-07
- sha256: `43e85dbf325c505b7b82ca41b9d86eecb2450b97f291bd356f2151af4c3d30cc`
- Chain used downstream: `A` (91 residues, numbered 5–95)
- WT MSA residue count: 94 (96 columns minus 2 gaps at columns 0 and 65)

The PDB chain is missing the first 3 N-terminal residues (`TSE`) of the
MSA wildtype. `SBM.utils.mpnn_score.build_msa_to_pdb_map` aligns the
PDB chain sequence to the WT MSA row to produce the MSA-column ↔
PDB-residue map at runtime; do not assume `len(pdb_residues)` equals
the number of non-gap WT columns.

Do not modify these files. Re-download from RCSB if a fresh copy is
needed and update the sha256 above.
