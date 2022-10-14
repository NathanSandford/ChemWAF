#!/bin/bash
# Job name:
#SBATCH --job-name=EriII_noSFHPrior
#
# Account:
#SBATCH --account=co_dweisz
#
# Partition:
#SBATCH --partition=savio2
#
# QoS:
#SBATCH --qos=dweisz_savio2_normal
#
# Nodes
#SBATCH --nodes=1
#
# Tasks per node
#SBATCH --ntasks-per-node=24
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
#SBATCH --output=logs/EriII_noSFHPrior.txt
#
## Command(s) to run:
echo "Loading modules"
source activate /clusterfs/dweisz/nathan_sandford/.conda/envs/ChemEv

python /clusterfs/dweisz/nathan_sandford/github_repos/ChemWAF/scripts/EriII_noSFHPrior.py