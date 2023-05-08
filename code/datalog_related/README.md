## Souffle
### Installation on Ubuntu 22.04
```
sudo wget https://souffle-lang.github.io/ppa/souffle-key.public -O /usr/share/keyrings/souffle-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/souffle-archive-keyring.gpg] https://souffle-lang.github.io/ppa/ubuntu/ stable main" | sudo tee /etc/apt/sources.list.d/souffle.list
sudo apt update
sudo apt install souffle
```

### Running Souffle program
```
souffle --version
```
- Run interpreter/compiler in parallel using N threads, N=auto for system default:
```
souffle -F . -D . tc.dl
souffle -F . -D . -o tc_dl tc.dl -j auto

souffle -F . -D . -o tc_dl tc.dl -j 128
time ./tc_dl
path    74619885
./tc_dl  177.81s user 1.97s system 764% cpu 23.505 total


g++ tc_dl.cpp -I . -O3 -fopenmp
time ./a.out -j 128
path	74619885
real	0m13.442s

cp sf.data edge.facts 
time ./a.out -j 128
cp p2p31.data edge.facts 
time ./a.out -j 128
cp p2p09.data edge.facts 
time ./a.out -j 128
cp p2p04.data edge.facts 
time ./a.out -j 128
cp cal.data edge.facts 
time ./a.out -j 128
cp tg.data edge.facts 
time ./a.out -j 128
cp ol.data edge.facts 
time ./a.out -j 128

```

- Souffle performance using different number of threads:

```shell

g++ tc_dl.cpp -I . -O3 -fopenmp

time ./a.out -j 1
path	884179859

real	9m7.514s
user	9m1.992s
sys	0m5.018s

arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 32
path	884179859

real	2m2.175s
user	11m32.363s
sys	0m10.700s
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 64
path	884179859

real	2m4.879s
user	11m55.119s
sys	0m17.087s
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 16
path	884179859

real	1m57.251s
user	9m34.587s
sys	0m7.887s
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 128
path	884179859

real	2m12.129s
user	13m40.888s
sys	0m22.758s

```

- Souffle performance using different number of threads for the same graph:

| Dataset | Number of rows | TC size | Iterations | Threads(Souffle) | Souffle(s) | 
| --- | --- | --- |----|------------------| --- |  
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 1                | 547.514 | 
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 2                | 321.823 | 
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 4                | 202.916 | 
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 8                | 143.917 |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 16               | 117.251 |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 32               | 122.175 |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 64               | 124.879 |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 128              | 132.129 | 


### Running Souffle in ThetaGPU
```shell
arsho@thetalogin4:~> ssh thetagpusn1
qsub -I -n 1 -t 60 -q single-gpu -A dist_relational_alg
# create directory and clone spack from the above section if that is not installed already 
cd ~/spack/bin/
./spack install gcc@11.3.0
./spack module tcl find gcc@11.3.0
gcc-11.3.0-gcc-9.4.0-tqxatvi
module use ~/spack/share/spack/modules/linux-ubuntu20.04-zen2
arsho@thetagpu06:~/spack/bin$ module avail gcc
module load gcc-11.3.0-gcc-9.4.0-tqxatvi
arsho@thetagpu06:~/spack/bin$ g++ --version
g++ (Spack GCC) 11.3.0
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ g++ tc_dl.cpp -I .
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out
path	80498014

real	3m55.989s
user	3m54.894s
sys	0m1.052s


for default gcc
module unload gcc-11.3.0-gcc-9.4.0-tqxatvi
module load openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0
```

### Error
```shell
time ./tc_dl
-bash: ./tc_dl: Permission denied

real	0m0.001s
user	0m0.001s
sys	0m0.000s
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ chmod +x tc_dl
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./tc_dl
./tc_dl: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by ./tc_dl)
./tc_dl: /usr/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found (required by ./tc_dl)
./tc_dl: /usr/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.33' not found (required by ./tc_dl)
./tc_dl: /usr/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found (required by ./tc_dl)

/GPUJoin/datalog_related$ g++ tc_dl.cpp -I . -O3
time ./a.out
path	74619885

real	1m19.381s
user	1m18.715s
sys	0m0.640s

```


### References
- [Install Souffle](https://souffle-lang.github.io/install.html)
- [Run Souffle](https://souffle-lang.github.io/execute)
