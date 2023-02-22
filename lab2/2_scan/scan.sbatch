#!/bin/bash
#SBATCH -J scan            # job name
#SBATCH -o scan.o%j        # output and error file name (%j expands to jobID)
#SBATCH -N 1               # total number of node
#SBATCH -n 1               # total number of mpi tasks requested
#SBATCH -p gtx             # queue (partition) -- normal, development, etc.
#SBATCH -t 00:30:00        # run time (hh:mm:ss) - 1.5 hours
module load gcc
module load cuda
make
./checker.pl scan             
./checker.pl find_repeats 
