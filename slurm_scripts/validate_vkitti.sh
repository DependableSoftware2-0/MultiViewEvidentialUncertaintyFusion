#!/bin/bash
#SBATCH --partition gpu      # partition (queue)
#SBATCH --nodes 1                # number of nodes
#SBATCH --mem 20G               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --tasks=32
#SBATCH --time 1-04:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output vkitti%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error vkitti%j.err  # filename for STDERR
while getopts c: flag
do
    case "${flag}" in
        c) comment=${OPTARG};;
    esac
done
echo "Comment : $comment";
module load cuda
module load gcc/9.1.0

source /scratch/dnair2m/miniconda3/bin/activate
conda activate tpytorch

cd $PBS_O_WORKDIR
# execute sequential program
which python
cd /home/dnair2m/MultiViewEvidentialUncertaintyFusion
#python3 train.py
#python3 fusion_train_validate.py --dataset vkitti --architecture efficientnet --slurm_id 483325
python3 fusion_train_validate.py --dataset vkitti --architecture resnet18 --slurm_id 485512
pwd
date
