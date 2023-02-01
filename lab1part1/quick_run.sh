#!/bin/bash
gcc -mavx -g $1.c -o $1 -O3
gcc -g $2.c -o $2 -O3
#perf stat -e cycles:u,LLC-load-misses:u,LLC-loads,cache-references:u,cache-misses:u --repeat 5 ./$1
#perf stat -e cycles:u,LLC-load-misses:u,LLC-loads,cache-references:u,cache-misses:u --repeat 5 ./matmul
perf stat -e ref-cycles:u,L1-dcache-loads:u,L1-dcache-load-misses:u,l2_rqsts.references:u,l2_rqsts.miss:u --repeat 20 ./$1
#perf stat -e ref-cycles:u,L1-dcache-loads:u,L1-dcache-load-misses:u,l2_rqsts.references:u,l2_rqsts.miss:u --repeat 5 ./$2

# ref-cycles:u
# L1-dcache-loads
# L1-dcache-load-misses
# l2_rqsts.references
# l2_rqsts.miss

# cycles:u
# LLC-load-misses
# LLC-loads
# cache-references:u
# cache-misses:u
