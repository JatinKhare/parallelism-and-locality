#!/bin/bash
#gcc -g chatgpt.c -o matmul
gcc matmul.c -o matmul -O3
#./matmul
perf stat -e cycles:u,instructions:u,cache-references:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,LLC-load:u,LLC-load-misses:u ./matmul 
