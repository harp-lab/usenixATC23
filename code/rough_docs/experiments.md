
### SuiteSparse Data Collection
- Go to [https://sparse.tamu.edu/?per_page=All](https://sparse.tamu.edu/?per_page=All)
- Select Undirected Graph type
- See the Nonzeros column, number of edges will be half on Nonzeros in actual graph
- Click on the graph link and download Matrix Markey format (open the Matrix Market as new window and then reload)
- Open a text editor and delete the meta data
- Replace space with \t using regular expression and save it

### Sparse graphs
- [fe_sphere: data_49152](https://sparse.tamu.edu/DIMACS10/fe_sphere)
- [fe_body: data_163734](https://sparse.tamu.edu/DIMACS10/fe_body)
- [loc-Brightkite: data_214078](https://sparse.tamu.edu/SNAP/loc-Brightkite)


## Old run output
```
Iteration: 9
join_result_rows: 1,533,673,887
projection_rows: 735,564,909
concatenated_rows: 1,399,870,967
deduplicated_result_rows:754,453,946

Iteration: 10
join_result_rows: 1,749,702,039
========= Program hit cudaErrorMemoryAllocation (error 2) due to "out of memory" on CUDA API call to cudaMalloc.
=========     Saved host backtrace up to driver entry point at error
=========     Host Frame:/lib/x86_64-linux-gnu/libcuda.so.1 [0x3d6d23]
=========     Host Frame:./tc_cuda.out [0x5dbdb]
=========     Host Frame:./tc_cuda.out [0x169b4]
=========     Host Frame:./tc_cuda.out [0x1b5bd]
=========     Host Frame:./tc_cuda.out [0xfa6e]
=========     Host Frame:./tc_cuda.out [0x115ef]
=========     Host Frame:./tc_cuda.out [0xc869]
=========     Host Frame:/lib/x86_64-linux-gnu/libc.so.6 (__libc_start_main + 0xf3) [0x24083]
=========     Host Frame:./tc_cuda.out [0xc9ce]
=========
========= Program hit cudaErrorMemoryAllocation (error 2) due to "out of memory" on CUDA API call to cudaGetLastError.
=========     Saved host backtrace up to driver entry point at error
=========     Host Frame:/lib/x86_64-linux-gnu/libcuda.so.1 [0x3d6d23]
=========     Host Frame:./tc_cuda.out [0x56134]
=========     Host Frame:./tc_cuda.out [0x16a5d]
=========     Host Frame:./tc_cuda.out [0x1b5bd]
=========     Host Frame:./tc_cuda.out [0xfa6e]
=========     Host Frame:./tc_cuda.out [0x115ef]
=========     Host Frame:./tc_cuda.out [0xc869]
=========     Host Frame:/lib/x86_64-linux-gnu/libc.so.6 (__libc_start_main + 0xf3) [0x24083]
=========     Host Frame:./tc_cuda.out [0xc9ce]
=========
terminate called after throwing an instance of 'thrust::system::detail::bad_alloc'
what():  std::bad_alloc: cudaErrorMemoryAllocation: out of memory
========= Error: process didn't terminate successfully
========= No CUDA-MEMCHECK results found

```
### Memory management
- Pageable memory vs pinned memory:
```shell
# Pageable memory
CUDA memcpy HtoD: 25.856us
CUDA memcpy DtoH: 180.32ms

# Pinned memory
CUDA memcpy HtoD: 25.983us
CUDA memcpy DtoH: 28.998ms
```

### Comparison between load factor 0.1 to 0.4 for transitive closure
| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | LF 0.1 (s) | LF 0.4 (s) |
| --- | --- | --- | --- | --- | --- |------------|
| CA-HepTh | 51,971 | 74,619,885 | 18 | 3,456 x 512 | 3.1588 | 3.0651 |
| SF.cedge | 223,001 | 80,498,014 | 287 | 3,456 x 512 | 11.8883 |11.7513 |
| ego-Facebook | 88,234 | 2,508,102 | 17 | 3,456 x 512 | 0.6426 | 0.6051 |
| wiki-Vote | 103,689 | 11,947,132 | 10 | 3,456 x 512 | 1.2424 | 1.2527 |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 512 | 0.8389 | 0.7525 |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 512 | 2.1293 | 2.1292 |
| cal.cedge | 21,693 | 501,755 | 195 | 3,456 x 512 | 0.4825 | 0.4371 |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 512 | 0.1625 | 0.1227 |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 512 | 0.0969 | 0.0546 |
| luxembourg_osm | 119,666 | 5,022,084 | 426 | 3,456 x 512 | 1.4788 | 1.3063 |
| fe_sphere | 49,152 | 78,557,912 | 188 | 3,456 x 512 | 13.5965 | 13.3869 |
| fe_body | 163,734 | 156,120,489 | 188 | 3,456 x 512 | 48.4381 | 48.4559 |
| cti | 48,232 | 6,859,653 | 53 | 3,456 x 512 | 0.4786 | 0.4248 |
| fe_ocean | 409,593 | 1,669,750,513 | 247 | 3,456 x 512 | 146.4821 | 145.0205 |
| wing | 121,544 | 329,438 | 11 | 3,456 x 512 | 0.0950 | 0.0681 |
| loc-Brightkite | 214,078 | 138,269,412 | 24 | 3,456 x 512 | 16.5086 | 16.4641 |
| delaunay_n16 | 196,575 | 6,137,959 | 101 | 3,456 x 512 | 1.3421 | 1.3160 |
| usroads | 165,435 | 871,365,688 | 606 | 3,456 x 512 | 367.2879 | 369.8636 |



### Comparison between CUDA and cuDF for isolated join operation

| Dataset | Number of rows | #Join    | Blocks x Threads | CUDA(s)  | cuDF(s)   | 
| --- |----------------|----------| --- |----------|-----------|
| random 1000000 | 1,000,000 | 30,511,908 | 3,456 x 512 | 0.118800 | 1.009139 |
| random 2000000 | 2,000,000 | 122,065,482 | 3,456 x 512 | 0.443422 | 3.559351 |
| random 3000000 | 3,000,000 | 274,633,933 | 3,456 x 512 | 0.986914 | 7.486542 |
| random 4000000 | 4,000,000 | 488,266,157 | 3,456 x 512 | 1.727381 | 13.013476 |
| random 5000000 | 5,000,000 | 762,962,523 | 3,456 x 512 | 2.640382 | 20.446236 |
| string 1000000 | 1,000,000 | 999,999 | 3,456 x 512 | 0.014831 | 0.144650 |
| string 2000000 | 2,000,000 | 1,999,999 | 3,456 x 512 | 0.022604 | 0.363000 |
| string 3000000 | 3,000,000 | 2,999,999 | 3,456 x 512 | 0.031009 | 0.577207 |
| string 4000000 | 4,000,000 | 3,999,999 | 3,456 x 512 | 0.038893 | 0.826158 |
| string 5000000 | 5,000,000 | 4,999,999 | 3,456 x 512 | 0.049386 | 1.062329 |
| CA-HepTh | 51,971         | 651,469 | 3,456 x 512 | 0.019241 | 0.011729  |
| SF.cedge | 2,23,001       | 273,550 | 3,456 x 512 | 0.050398 | 0.014935  |
| ego-Facebook | 88,234         | 2,690,019 | 3,456 x 512 | 0.031749 | 0.016591  |
| wiki-Vote | 1,03,689       | 4,542,805 | 3,456 x 512 | 0.040988 | 0.019290  |
| p2p-Gnutella09 | 26,013 | 108,864 | 3,456 x 512 | 0.008062 | 0.008280  |
| p2p-Gnutella04 | 39,994 | 180,230 | 3,456 x 512 | 0.010563 | 0.009489  |
| cal.cedge | 21,693 | 19,836 | 3,456 x 512 | 0.006451 |  0.004164 |
| TG.cedge | 23,874 | 24,274 | 3,456 x 512 | 0.008955 | 0.004714 |
| OL.cedge | 7,035 | 7,445 | 3,456 x 512 | 0.002515 | 0.004074 |
| luxembourg_osm | 119,666 | 114,532 | 3,456 x 512 | 0.026753 | 0.008111 |
| fe_sphere | 49,152 | 146,350 | 3,456 x 512 | 0.011760 | 0.009437 |
| fe_body | 163,734 | 609,957 | 3,456 x 512 | 0.037617 | 0.014670 |
| cti | 48,232 | 130,492 | 3,456 x 512 | 0.011542 | 0.007038 |
| fe_ocean | 409,593 | 1,175,076 | 3,456 x 512 | 0.091541 | 0.019214 |
| wing | 121,544 | 116,371 | 3,456 x 512 | 0.034421 | 0.008459 |
| loc-Brightkite | 214,078 | 3,368,451 | 3,456 x 512 | 0.099064 | 0.025354 |
| delaunay_n16 | 196,575 | 393,028 | 3,456 x 512 | 0.058860 | 0.014754 |
| usroads | 165,435 | 206,898 | 3,456 x 512 | 0.037773 | 0.013676 |


### Running NCCL
```
nvcc -o nccl_example nccl_example.cu -lnccl
./nccl_example
```

### References
- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)