#!/bin/bash -l
#SBATCH --job-name=stack_images
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=3G
#SBATCH --array=0-39%10

# Keep --ntasks=1 as stack_images.py doesn't make use of MPI.

# Increase --mem as necessary up to the maximum memory of a single node.

# The --array numbers specify the range of indices in the FIELDS array that will be passed
# to stack_images.py, one per job. The number after % specifies the maximum number of
# simultaneous jobs, useful for throttling if required. 

FIELDS=(
	"0012-06A"
	"0012+00A"
	"0037-06A"
	"0037+00A"
	"0102-06A"
	"0102+00A"
	"0126-06A"
	"0126+00A"
	"0151-06A"
	"0151+00A"
	"0216-06A"
	"0216+00A"
	"0241-06A"
	"0241+00A"
	"0306-06A"
	"0306+00A"
	"0331-06A"
	"0331+00A"
	"0355-06A"
	"0355+00A"
	"2004-06A"
	"2004+00A"
	"2028-06A"
	"2028+00A"
	"2053-06A"
	"2053+00A"
	"2118-06A"
	"2118+00A"
	"2143-06A"
	"2143+00A"
	"2208-06A"
	"2208+00A"
	"2233-06A"
	"2233+00A"
	"2257-06A"
	"2257+00A"
	"2322-06A"
	"2322+00A"
	"2347-06A"
	"2347+00A"
)

source activate vast
module use /sharedapps/LS/cgca/modulefiles
module load swarp

field=${FIELDS[${SLURM_ARRAY_TASK_ID}]}
srun --job-name="${SLURM_ARRAY_TASK_ID}_${field}" python ~/vast/stack_images.py -o temp/ -s 500 --clean "${field}" -vv -i "EPOCH*/COMBINED/STOKESI_IMAGES/"