# data/structures/

Reference PDB structures, one per family. Kept as immutable reference
structures.

## 1ECM.pdb

E. coli chorismate mutase, anchor structure for the CM family.

- Source URL: https://files.rcsb.org/download/1ECM.pdb
- Downloaded: 2026-05-07
- sha256: `43e85dbf325c505b7b82ca41b9d86eecb2450b97f291bd356f2151af4c3d30cc`
- Chain used downstream: `A` (91 residues, numbered 5–95)
- WT MSA residue count: 94 (96 columns minus 2 gaps at columns 0 and 65)

The PDB chain is missing the first 3 N-terminal residues (`TSE`) of the
MSA wildtype.

## 1JNT.pdb

E. coli peptidyl-prolyl cis/trans-isomerase parvulin 10 (Par10; gene
*ppiC* / *parA*, UniProt P0A9L5), reference structure for the PPIC
family.

- Source URL: https://files.rcsb.org/download/1JNT.pdb
- Downloaded: 2026-06-09
- sha256: `59fa18ac4b945ff46c1c57df20ecbed418302a7a225ad05470a7dd43110c4704`
- Chain used downstream: `A` (92 residues, numbered 1–92)
- WT MSA residue count: 91 (91 columns, no gaps in the WT row)

This family's experimental structures are solution NMR only — there is
no X-ray structure of E. coli Par10. We use **1JNT**, the deposited
single-model `MINIMIZED AVERAGE` of the canonical 18-conformer ensemble
**1JNS** (`REMARK 210`: "ENSEMBLE (1JNS) WAS REGULARIZED"; 1JNS itself
designates no single representative conformer — `BEST REPRESENTATIVE
CONFORMER IN THIS ENSEMBLE : NULL`). 1JNT and 1JNS have identical
chain-A sequences; 1JNT avoids the multi-MODEL ambiguity of the
ensemble file.

The PDB chain has one extra N-terminal residue (`A`, residue 1)
relative to the MSA wildtype, which starts at PDB residue 2 (`K`).

Do not modify these files. Re-download from RCSB if a fresh copy is
needed and update the sha256 above.
