## NIfTI Super-Resolution Inference

## Input Structure


Example data and weights are in this link.


Each subdirectory under the input root represents one case:

```text
input_root/
└── case_name/
    ├── 3D.nii.gz
    ├── T1_brain.nii.gz
    └── T1_brain_mask.nii.gz
```

Requirements:

- `3D.nii.gz` must have shape `(x, y, z, 31)`.
- Volume 0 must be b0, and volumes 1–30 must be the 30 b1000 directions.
- The first three dimensions of the DWI, T1, and mask images must match.
- The DWI, T1, and mask images must already be registered.

The data in the example is a presentation, showing only a part of the data.

## Running Inference

The default input and checkpoint paths for the current cluster are defined in
`sample_nifti.sh`. Submit the job from this directory with:

```bash
sbatch sample_nifti.sh
```

You can override the input path, checkpoint, output path, and slicing direction
without editing any files:

```bash
INPUT_ROOT=/path/to/input \
CHECKPOINT=/path/to/model.pt \
OUTPUT_ROOT=/path/to/output \
DIRECTION=axial \
sbatch sample_nifti.sh
```

```bash
bash sample_nifti.sh
```

Use the command above only from an interactive session with an allocated GPU.

Results are saved under `output/YYMMDD_HH/` by default. The final NIfTI file is
named:

```text
case_name_2_axial.nii.gz
```

## Python Environment

Activate the Python environment that was previously used to run this model. For
a new environment, start by installing the listed dependencies:

```bash
pip install -r requirements.txt
```

CUDA and PyTorch must be compatible with the installed GPU driver. This
inference entry point uses one process and one GPU; MPI is not required.
