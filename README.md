# Aflow Metadynamics Comparison Pipeline

This repository implements a pipeline for comparing molecular dynamics ensembles generated from:

- long, unbiased reference molecular dynamics
- AlphaFlow / AlphaFold–derived ensemble sketches
- biased metadynamics runs using distance-based collective variables (CVs) derived from those ensemble sketches

All experiments are run on proteins from the ATLAS dataset.

The goal is to project all ensembles into a **shared structural space** and compute **PCA-based free energy surfaces (FES)** so that biased, unbiased, and generative ensembles can be compared in a controlled and reproducible way.  
Future extensions will add additional metrics and collective variable types.

---

## What this pipeline does


### 1. Data sources and collective variable definition

- We first extract ensembles from AlphaFlow
  (https://github.com/bjing2016/alphaflow, using the alphaflow-MD base),
  producing a multi-model PDB containing ~250 structural samples.
- From these models, we define collective variables intended to capture the dominant modes of structural variation.
  By default, we use distance-based CVs constructed from the top M most variable inter-residue distances.
- These CVs are then encoded into a PLUMED input file and used to bias OpenMM simulations via metadynamics
  (https://www.plumed.org/doc-v2.9/user-doc/html/belfast-6.html).

Distance-based CVs are chosen because they are simple, robust to minor topology differences, and map cleanly onto the downstream PCA space.

### 2. OpenMM (+ PLUMED) simulation output
Each OpenMM run produces:
- a trajectory (`.xtc`)
- a final structure (`*_final.pdb`)
- optional PLUMED `COLVAR` files for metadynamics

Runs are stored in structured per-system directories.

Biased trajectories are run for ~200 ns, while unbiased reference trajectories are run for ~5 µs.
The goal is for the biased simulations to recover the same regions of configuration space as the long unbiased runs,
as assessed by overlap and free energy structure in PCA space.
---

### 3. Trajectory postprocessing
Each run directory is postprocessed to generate **canonical, comparison-ready outputs**:

- `fixed.xtc`
  - molecules made whole across periodic boundaries
  - molecules imaged back into the primary unit cell
- `fixed_final.pdb`
  - final structure used as the topology for analysis

After this step, all downstream analysis uses only these files.
---

### 4. Topology harmonization (minimal & robust)
Before any PCA or metric computation, all datasets are harmonized by:

- selecting **Cα atoms only**
- matching residues by **(chain index, residue number, residue name)**
- intersecting the common residue set
- slicing all trajectories to the same atom ordering

This avoids issues from:
- missing hydrogens
- different atom naming conventions
- engine-specific topology differences

No atom remapping, optimization, or graph matching is performed.

---

### 5. Standardization into reference space
All harmonized trajectories are:

- superposed onto the reference MD
- represented in the same coordinate basis
- guaranteed to have identical atom counts and ordering

This ensures fair, apples-to-apples comparison.

---

### 6. Metric computation
Metrics are implemented as **pluggable modules**.

Currently supported:
- `pca_compare`
  - PCA projection into reference space
  - free energy surface estimation
  - PLUMED metadynamics reweighting via `rbias`
  - consistent color scaling across ensembles

Each metric writes results into its own output subdirectory.

---

## Expected run directory structure

Each OpenMM run directory must contain:


run_dir/
├── fixed.xtc
├── fixed_final.pdb
├── COLVAR_biased.dat (or COLVAR_unbiased.dat)
├── params.json
└── ...



The comparison pipeline **only consumes these canonical files**.

---

## Future extensions

- adding a yaml file to import environment
- small code fixes, remove a few hard-coded values that were for my internal workflow.
- Defining locations for data directories.
- residue renumbering support
- distance / dihedral feature PCA
- ensemble overlap metrics (KL, JS, EMD)
- multi-replica aggregation
- trajectory time alignment


## Acknowledgements and prior work

This pipeline builds on ideas and tooling from prior work in ensemble-based collective variable construction and enhanced sampling.

In particular:

AlphaFlow
Ensemble generation and sketching are based on the AlphaFlow framework by Jing et al.
https://github.com/bjing2016/alphaflow

AlphaFlow is released under the MIT License and is used here to generate structural ensembles that serve as the basis for defining collective variables and downstream comparisons.

AlphaFold2-RAVE
The logic for selecting informative distance-based collective variables is inspired by the AlphaFold2-RAVE framework developed by the Tiwary Lab:
https://github.com/tiwarylab/alphafold2rave

Specifically, the idea of ranking and selecting distances based on ensemble variability follows the approach implemented in:
https://github.com/tiwarylab/alphafold2rave/blob/f77266e385b5279eb15bc868bf10d1031a0cce6f/ravefuncs.py#L162-L198

AlphaFold2-RAVE is also released under the MIT License.

This repository does not reuse AlphaFold2-RAVE code directly, but adapts the underlying concept to a deterministic, PCA-aligned comparison pipeline designed for metadynamics validation and ensemble benchmarking.