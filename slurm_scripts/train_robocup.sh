#!/bin/bash
#SBATCH --partition gpu      # partition (queue)
#SBATCH --nodes 1                # number of nodes
#SBATCH --mem 40G               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --tasks=64
#SBATCH --time 0-04:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output robocup%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error robocup%j.err  # filename for STDERR
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
#python3 train.py --dataset robocup --architecture resnet18
python3 train.py --dataset robocup --architecture efficientnet --max_epochs 130
#python3 train.py --dataset robocup --architecture mobilenet --max_epochs 130
#python3 train.py --dataset robocup --architecture resnet18 --max_epochs 130
pwd
date
