#!/encs/bin/tcsh

# Set environment variables
set ENV_NAME = ExpressGen_env
set ENV_DIR = /speed-scratch/$USER/comp6841/envs
set ENV_PATH = $ENV_DIR/$ENV_NAME

set TMP_DIR = $ENV_DIR/tmp
set PKGS_DIR = $ENV_DIR/pkgs
set CACHE_DIR = $ENV_DIR/cache
set HF_DIR = $ENV_DIR/huggingface

setenv TMP $TMP_DIR
setenv TMPDIR $TMP_DIR
setenv CONDA_PKGS_DIRS $PKGS_DIR
setenv PIP_CACHE_DIR $CACHE_DIR
setenv HF_HOME $HF_DIR
setenv HF_HUB_CACHE $CACHE_DIR

# Load Anaconda module if not already loaded
module load anaconda3/2023.03/default

# Activate the Conda environment
echo "Activating Conda environment $ENV_NAME..."
echo "======================================================================"
conda activate "$ENV_PATH"
if ($status != 0) then
    "Error: Failed to activate the Conda environment $ENV_NAME."
    exit 1
endif

# Update Conda packages
echo "Updating Conda packages from environment.yml..."
echo "======================================================================"
conda env update -p "$ENV_PATH" --file environment.yml --prune

echo "Conda environemnt summary..."
echo "======================================================================"
conda list
pip list

# Deactivate Conda env
echo "Deactivating Conda environment..."
conda deactivate
echo "================================ DONE ================================"