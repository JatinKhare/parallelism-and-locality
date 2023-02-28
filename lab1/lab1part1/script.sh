#!/bin/bash
#gcc -g chatgpt.c -o matmul
#gcc matmul.c -o matmul -O3
gcc -g matmul_opti.c -o matmul_opti -O3
#./matmul
perf stat -e cycles:u,instructions:u,cache-references:u,cache-misses:u,L1-dcache-loads:u --repeat 20 ./matmul_opti
perf stat -e cycles:u,instructions:u,cache-references:u,cache-misses:u,L1-dcache-loads:u --repeat 20 ./matmul

#,L1-dcache-load-misses:u,LLC-loads:u,LLC-load-misses:u ./matmul 
