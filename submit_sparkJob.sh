#!/bin/bash

#SBATCH --job-name=similarity-calc
#SBATCH --output=similarity-calc-%j.out
#SBATCH --error=similarity-calc-%j.err
#SBATCH --nodes=2            # Utilize 4 nodes
#SBATCH --ntasks-per-node=8  # Run 4 tasks per node
#SBATCH --cpus-per-task=4    # Each task uses 4 CPUs
#SBATCH --mem=32Gb           # Memory per node
#SBATCH --time=08:00:00      # Time limit hrs:min:sec
#SBATCH --partition=short    # Partition to submit to




module load python/3.8.1

source ~/my_pyspark_env/bin/activate


export SPARK_HOME=/home/rivera.and/dependencies/spark/spark-3.5.1-bin-hadoop3
export PATH=$SPARK_HOME/bin:$PATH


srun python -u main.py
