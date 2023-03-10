#!/bin/bash
#SBATCH -J render           # job name
#SBATCH -o render.o%j       # expands to jobID
#SBATCH -N 1                # total number of node
#SBATCH -n 1                # total number of mpi tasks requested
#SBATCH -p gtx              # queue (partition) — normal, development, etc.
#SBATCH -t 00:30:00         # run time (hh:mm:ss) - 1.5 hours
module load gcc
module load cuda
./render -r cuda -b 0:4 -f rgbtest_cuda -s 768 rgb
./checker.pl 
