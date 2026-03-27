---
sidebar_position: 2
---

# Installation

Amigo uses C++ under the hood, so it needs a compiler and a few native libraries.
The easiest way to get everything set up is with **Conda** — it handles the tricky parts (CMake, MPI, MKL) automatically.

## Quick Start (Recommended)

### 1. Install Miniconda

If you don't have Conda yet, install [Miniconda](https://docs.anaconda.com/miniconda/install/).

### 2. Platform Prerequisites

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="windows" label="Windows">

Install **Visual Studio Build Tools** (free):

1. Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Run the installer and select **"Desktop development with C++"**
3. Complete the installation

</TabItem>
<TabItem value="mac" label="macOS">

Install the Xcode Command Line Tools (provides the C++ compiler):

```bash
xcode-select --install
```

</TabItem>
<TabItem value="linux" label="Linux">

Install a C++ compiler and make (most distributions include these):

```bash
# Ubuntu/Debian
sudo apt install build-essential

# Fedora
sudo dnf install gcc-c++ make
```

</TabItem>
</Tabs>

### 3. Create the Conda Environment

Clone the repository and create the environment:

```bash
git clone https://github.com/smdogroup/amigo.git
cd amigo
conda env create -f environment.yml
conda activate amigo
```

### 4. Install Amigo

```bash
pip install -e .
```

That's it! You're ready to run the examples.

## Verifying the Installation

Run a quick test to make sure everything works:

```bash
cd examples
python -c "import amigo; print('Amigo installed successfully!')"
```

## Advanced Options

### OpenMP Support

To enable OpenMP parallelization:

```bash
pip install -e . -v \
    -Ccmake.args="-DCMAKE_CXX_COMPILER=mpicxx" \
    -Ccmake.args="-DAMIGO_ENABLE_OPENMP=ON" \
    -Ccmake.args="-DAMIGO_ENABLE_CUDA=OFF"
```

### CUDA Support

For GPU acceleration (requires NVIDIA GPU and CUDA toolkit):

```bash
pip install -e . -v \
    -Ccmake.args="-DCMAKE_CXX_COMPILER=mpicxx" \
    -Ccmake.args="-DAMIGO_ENABLE_OPENMP=OFF" \
    -Ccmake.args="-DAMIGO_ENABLE_CUDA=ON" \
    -Ccmake.args="-DCUDSS_HOME=/path/to/cudss" \
    -Ccmake.args="-DAMIGO_ENABLE_CUDSS=ON" \
    -Ccmake.args="-DCMAKE_CUDA_ARCHITECTURES=native"
```

:::note
You cannot enable both CUDA and OpenMP at the same time.
:::

### OpenMDAO Integration

```bash
pip install amigo[openmdao]
```

## Troubleshooting

### CMake can't find BLAS/LAPACK (Windows)

Make sure the `amigo` conda environment is activated. Conda provides MKL which CMake detects automatically.

### CMake can't find MPI

Conda installs MPI automatically via `mpi4py`. Ensure the conda environment is activated before running `pip install`.

### Compiler not found (Windows)

Ensure Visual Studio Build Tools is installed with the **"Desktop development with C++"** workload. You may need to restart your terminal after installation.

### Compiler not found (macOS)

Run `xcode-select --install` and accept the license agreement. If you already have Xcode installed, this step is not needed.
