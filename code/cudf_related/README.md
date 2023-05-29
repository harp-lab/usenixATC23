### Run Transitive Closure Computation using cuDF in Conda environment
- Create and activate new `conda` environment:
```
conda create --name gpu_env
conda activate gpu_env
```
- Install packages:
```
conda install -c rapidsai -c nvidia -c numba -c conda-forge \
    cudf=22.06 python=3.9 cudatoolkit=11.2
```
- Run the program:
```
python transitive_closure.py
```

### Run Transitive Closure Computation using cuDF in ThetaGPU
- Login to theta gpu node:
```shell
ssh USERNAME@theta.alcf.anl.gov
ssh thetagpusn1
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/usenixATC23/code/cudf_related/
```
- Change the permission
```shell
chmod u+x cudf_submit.sh
```
- Submit the job:
```shell
qsub -O cudf_submit -e cudf_submit.error cudf_submit.sh
```


### References
- [cudf installation docs](https://github.com/rapidsai/cudf)
- [nvidia rapids kit cheatsheet](https://images.nvidia.com/aem-dam/Solutions/ai-data-science/rapids-kit/accelerated-data-science-print-getting-started-cheat-sheets.pdf)
- [blog article on conda usage](https://carpentries-incubator.github.io/introduction-to-conda-for-data-scientists/02-working-with-environments/index.html)
- [cugraph installation docs](https://github.com/rapidsai/cugraph#conda)