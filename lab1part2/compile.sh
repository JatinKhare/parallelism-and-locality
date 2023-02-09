#!/bin/bash
gcc $1.c -o $1 -O3
perf stat -e ref-cycles:u,L1-dcache-loads:u,L1-dcache-load-misses:u,l2_rqsts.references:u,l2_rqsts.miss:u --repeat 20 ./$1
