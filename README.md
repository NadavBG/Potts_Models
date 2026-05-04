# Stochastic Boltzmann Machine (SBM)
Tested with Python 3.11.11

SBM contains the implementation of a boltzmann machine with L-BFGS gradient descent and implicit regularization. The Monte-Carlo is written in C++ with OpenMP, compiled automatically using scikit-build-core and CMake.

## Installation

### Create a virtual environment (recommended)

Here's an example using virtualenv:

```
pip3 install virtualenv
virtualenv -p python3 env_SBM
source env_SBM/bin/activate
```

### System requirements

These requirements depend on your operating system.

**macOS**

macOS’s default compiler AppleClang does not support OpenMP.
Therefore you must install LLVM + libomp + ninja:
```
brew install llvm libomp ninja
```
The build system automatically detects and uses:
- /opt/homebrew/opt/llvm/bin/clang
- /opt/homebrew/opt/llvm/bin/clang++
- OpenMP via libomp
- ninja as CMake generator

**Linux**

The following system dependencies are required:

- GCC / G++ (via build-essential): Required to compile the C++ Monte Carlo modules.

- Python development headers (python3-dev): Required because the C++ modules include Python.h.

- CMake: Used by scikit-build-core to configure and build the extension modules.

- Ninja (ninja-build): Recommended build backend used automatically by scikit-build-core.

### Python dependencies

Install Python requirements:
```
pip3 install -r requirements.txt --no-cache-dir
```

### Build & install SBM (with C++ modules)

```
pip3 install -e .
```

## Dataset format

Before using SBM to infer fields and couplings from a MSA you need to load your fasta file and turn this fasta file into a numpy array of size (Number of sequences x Protein length)

```python
import SBM.utils.utils as ut
MSA = ut.load_fasta('fasta_file')
np.save('data/MSA_array/MSA_fam.npy',MSA)
```

## Training

See demo-SBM-CM-family for an example on the Chorismate Mutase family.

## Example structure (data folder)

```
data/
├── fasta
├── MSA_array
	└── MSA_CM.npy
├── Ind_train
	└── Ind_train_CM.npy
```


## Jupyter notebook on a server

```
pip3 install ipykernel
````
Then select the environment kernel env_SBM inside Jupyter.

## Citation

If you use this code or data, please cite the associated publication.
