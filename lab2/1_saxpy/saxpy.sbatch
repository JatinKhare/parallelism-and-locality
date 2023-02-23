#!/bin/bash
#SBATCH -J saxpy           # job name
#SBATCH -o saxpy.o%j       # output and error file name (%j expands to jobID)
#SBATCH -N 1               # total number of node
#SBATCH -n 1               # total number of mpi tasks requested
#SBATCH -p gtx             # queue (partition) -- normal, development, etc.
#SBATCH -t 00:30:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-type=end    # email me when the job finishes
module load gcc
module load cuda
export CUDA_PATH=$TACC_CUDA_DIR 
make
#nsys profile -–stats=true  ./cudaSaxpy -n 500000000 -p 1 -i 4
#nsys profile -t cuda,nvtx --stats=true --trace=cuda,osrt,cublas,mpi --force-overwrite true -o profile_report ./cudaSaxpy -n 500000000 -p 1 -i 4
nsys profile --stats=true ./cudaSaxpy -n 500000000 -p 1 -i 4
nsys profile --stats=true ./cudaSaxpyUvm -n 500000000 -p 1 -i 4
nsys profile --stats=true ./cudaSaxpyStream -n 500000000 -p 512 -i 4
#./cudaSaxpy -n 500000000 -p 1 -i 4
#./cudaSaxpyUvm -n 500000000 -p 1 -i 4
#./cudaSaxpyStream -n 500000000 -p 512 -i 4
