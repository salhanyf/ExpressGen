#!/encs/bin/tcsh

date
echo "======================================================================"
# Load modules
module load anaconda3/2023.03/default
module load cuda/11.8/default
module load python/3.11.6/default

# Set CUDA environment variables
setenv CUDA_HOME /encs/pkg/cuda-11.8/root
setenv PATH ${CUDA_HOME}/bin:${PATH}
setenv LD_LIBRARY_PATH ${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "Setting some initial vairables..."
set ENV_NAME = ExpressGen_env
set ENV_DIR = /speed-scratch/$USER/comp6841/envs
set ENV_PATH = $ENV_DIR/$ENV_NAME

set TMP_DIR = $ENV_DIR/tmp
set PKGS_DIR = $ENV_DIR/pkgs
set CACHE_DIR = $ENV_DIR/cache
set HF_DIR = $ENV_DIR/huggingface

mkdir -p $ENV_DIR $TMP_DIR $PKGS_DIR $CACHE_DIR $HF_DIR

setenv TMP $TMP_DIR
setenv TMPDIR $TMP_DIR
setenv CONDA_PKGS_DIRS $PKGS_DIR
setenv PIP_CACHE_DIR $CACHE_DIR
setenv HF_HOME $HF_DIR
setenv HF_HUB_CACHE $CACHE_DIR

# Check if the environment already exists
if ( -d "$ENV_PATH" ) then
    echo "Environment $ENV_NAME already exists at $ENV_PATH. Activating it..."
    echo "======================================================================"
    conda activate "$ENV_PATH"
    if ($status != 0) then
        echo "Error: Failed to activate Conda environment."
        exit 1
	endif
else
	echo "Creating Conda environment $ENV_NAME at $ENV_PATH from environment.yml..."
    echo "======================================================================"
    conda create -p "$ENV_PATH" -c conda-forge python=3.11.6 -y

	echo "Activating Conda environment $ENV_NAME..."
	echo "======================================================================"
    conda activate "$ENV_PATH"
    if ($status != 0) then
        echo "Error: Failed to activate Conda environment."
        exit 1
	endif

	echo "Installing essential packages..."
	echo "======================================================================"
	pip install --upgrade pip
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install matplotlib numpy tqdm pandas pillow ipywidgets ipykernel
    pip install transformers datasets accelerate
    conda install -c conda-forge jupyterlab -y

    python -m ipykernel install --user --name=ExpressGen_env --display-name "Python (ExpressGen)"
endif

echo "Conda environemnt summary..."
echo "======================================================================"
conda info --envs
conda list
pip list

echo "Saving environment to environment.yml..."
echo "======================================================================"
conda env export --from-history > environment-clean.yml
conda env export > environment.yml

# Deactivate Conda env
echo "Deactivating Conda environment..."
conda deactivate
echo "================================ DONE ================================"
date
exit
