#!/bin/bash
set -e

echo "=============================================="
echo "   HydraRNA Environment Auto Installer"
echo "=============================================="

# ----------------------------
# 1. Create and activate conda env (with existence check)
# ----------------------------

echo "[1/10] Checking if conda environment 'HydraRNA' already exists ..."
EXISTS=$(conda env list | awk '{print $1}' | grep -w "^HydraRNA$" || true)
if [ ! -z "$EXISTS" ]; then
    echo "ERROR: Conda environment 'HydraRNA' already exists."
    echo "Please remove it first with:  conda remove -n HydraRNA --all"
    echo "Or choose a different environment name."
    exit 1
fi
echo "Environment not found. Creating HydraRNA environment..."
conda create -y -n HydraRNA python=3.9.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate HydraRNA

# ----------------------------
# 2. Install PyTorch 2.3.1 + cu118
# ----------------------------

echo "[2/10] Installing PyTorch 2.3.1 + cu118 ..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu118

# ----------------------------
# 3. Install CUDA Toolkit 11.8 (with nvcc)
# ----------------------------

echo "[3/10] Installing CUDA Toolkit 11.8 ..."
conda install -y -c "nvidia/label/cuda-11.8.0" cuda

# ----------------------------
# 4. Install gcc-11 & g++-11
# ----------------------------

echo "[4/10] Installing gcc-11 & g++-11 ..."
conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11

# ----------------------------
# 5. Export compiler & CUDA env variables
# ----------------------------

echo "[5/10] Setting compiler and CUDA environment variables ..."
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export CUDAHOSTCXX=$CXX

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ----------------------------
# 6. Install mamba-ssm
# ----------------------------

echo "[6/10] Installing mamba-ssm ..."
pip install --no-build-isolation "mamba-ssm[causal-conv1d]==2.2.2"

# ----------------------------
# 7. Install flash-attn==2.6.1
# ----------------------------

echo "[7/10] Installing flash-attn==2.6.1 ..."
pip install "flash-attn==2.6.1" --no-build-isolation

# ----------------------------
# 8. Install matching fused_dense_lib from tag v2.6.1
# ----------------------------

echo "[8/10] Installing fused_dense_lib from flash-attn v2.6.1 ..."
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
git checkout v2.6.1
cd csrc/fused_dense_lib
pip install .
cd ../../..

# ----------------------------
# 9. Install remaining Python dependencies
# ----------------------------

echo "[9/10] Installing additional dependencies ..."
pip install pip==24.0
pip install pandas tqdm tensorboardX pysam transformers==4.44.0 cython

# ----------------------------
# 10. Install HydraRNA fairseq (with cython-generated cpp)
# ----------------------------

echo "[10/10] Installing HydraRNA + fairseq ..."
git clone https://github.com/GuipengLi/HydraRNA.git
cd HydraRNA/fairseq

# Generate missing Cython cpp files
cython fairseq/data/data_utils_fast.pyx -3
cython fairseq/data/token_block_utils_fast.pyx -3

pip install --no-build-isolation -e .

echo "=============================================="
echo " HydraRNA installation complete!"
echo " Run:  conda activate HydraRNA"
echo " Then test example in HydraRNA/examples/"
echo "=============================================="
