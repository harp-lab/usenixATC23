#!/bin/bash
#COBALT -n 1
#COBALT -t 60
#COBALT -q single-gpu
#COBALT -A dist_relational_alg
#COBALT -O single-gpu-job
#COBALT -e single-gpu-job.error
. /etc/profile
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/USENIXATC_2023/
make run