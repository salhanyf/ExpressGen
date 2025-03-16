#!/encs/bin/tcsh

# Set environment variables
set ENV_NAME = ExpressGen_env
set ENV_DIR = /speed-scratch/$USER/envs
set ENV_PATH = $ENV_DIR/$ENV_NAME
set TMP_DIR = $ENV_DIR/tmp
set PKGS_DIR = $ENV_DIR/pkgs
set PIP_DIR = $ENV_DIR/cache

setenv TMP $TMP_DIR
setenv TMPDIR $TMP_DIR
setenv CONDA_PKGS_DIRS $PKGS_DIR
setenv PIP_CACHE_DIR $PIP_DIR

# Navigate to the Jupyter directory
cd .. || exit

# Load Anaconda module if not already loaded
module load anaconda3/2023.03/default

# Activate the Conda environment
conda activate "$ENV_PATH"
if ($status != 0) then
    "Error: Failed to activate the Conda environment $ENV_NAME."
    exit 1
endif

# Get the compute node name and user information
set node = `hostname -s`
set user = `whoami`

# Get an available port for Jupyter
set port = `python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'`

# Print SSH tunneling instructions
echo "======================================================================"
echo "To connect to the compute node ${node} on speed.encs.concordia.ca running your Jupyter notebook server, \n\
use the following SSH command in a new terminal on your workstation/laptop:\n\n\
ssh -L ${port}:${node}:${port} ${user}@speed.encs.concordia.ca\n\n\
Then, copy the URL provided below by the Jupyter server (starting with http://127.0.0.1/...) and paste it into your browser.\n\n\
IMPORTANT: Before disconnecting, make sure to close open notebooks and shut down Jupyter to avoid manual job cancellation."
echo "======================================================================"

# Start the Jupyter server in the background
echo "Starting Jupyter server in background with requested resources"
jupyter lab --no-browser --notebook-dir=$PWD --ip="0.0.0.0" --port=${port} --port-retries=50