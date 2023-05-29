#!/bin/bash
#COBALT -n 1
#COBALT -t 30
#COBALT -A dist_relational_alg
module load conda/2022-07-01
conda activate /home/arsho/gpujoinenv
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/usenixATC23/code/cudf_related/
/home/arsho/gpujoinenv/bin/python transitive_closure.py