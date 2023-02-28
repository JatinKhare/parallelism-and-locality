#!/bin/bash
g++ -std=c++17  $1.cpp -o $1
#rm our_hit_rate_$1.csv
#rm our_access_$1.csv
#rm L2_access_$1.csv
#rm L2_hits_$1.csv
./$1

