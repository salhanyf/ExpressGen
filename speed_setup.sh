#!/encs/bin/tcsh

#SBATCH -J ExpressGen
#SBATCH -A comp6841w25
#SBATCH --mail-type=ALL

# Set output directory/files to current
#SBATCH --chdir=./
#SBATCH -o output-%A.log
#SBATCH -e error-%A.log

#SBATCH --gres=gpu:nvidia_a100_7g.80gb:1
#SBATCH --mem=100G

# Request CPU slots (processes and threads: n * c)
#SBATCH -n 1
#SBATCH -c 16

date
echo "======================================================================"
./jupyter_run.sh
echo "================================ DONE ================================"
date
# EOF
