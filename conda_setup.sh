#!/encs/bin/tcsh

date
echo "======================================================================"
# Load Anaconda module
module load anaconda3/2023.03/default

echo "Setting some initial vairables..."
set ENV_NAME = ExpressGen_env
set ENV_DIR = /speed-scratch/$USER/envs
set ENV_PATH = $ENV_DIR/$ENV_NAME

set TMP_DIR = $ENV_DIR/tmp
set PKGS_DIR = $ENV_DIR/pkgs
set PIP_DIR = $ENV_DIR/cache

mkdir -p $ENV_DIR $TMP_DIR $PKGS_DIR $CACHE_DIR

setenv TMP $TMP_DIR
setenv TMPDIR $TMP_DIR
setenv CONDA_PKGS_DIRS $PKGS_DIR
setenv PIP_CACHE_DIR $PIP_DIR

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
    conda env create -p "$ENV_PATH" -f environment.yml

	echo "Activating Conda environment $ENV_NAME..."
	echo "======================================================================"
    conda activate "$ENV_PATH"
    if ($status != 0) then
        echo "Error: Failed to activate Conda environment."
        exit 1
	endif

	echo "Installing essential packages..."
	echo "======================================================================"
    conda install -c conda-forge jupyterlab -y
	pip install --upgrade pip
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
endif

echo "Conda environemnt summary..."
echo "======================================================================"
conda info --envs
conda list
pip list

# Deactivate Conda env
echo "Deactivating Conda environment..."
conda deactivate
echo "================================ DONE ================================"
date
exit