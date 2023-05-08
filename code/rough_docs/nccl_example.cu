#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#define N 10000000
#define MAX_ERR 1e-6

__global__ void initBuffer(int *out, int n) {
    for(int i = 0; i < n; i++){
        out[i] = 1;
    }
}

int main(int argc, char* argv[])
{
    //managing 4 devices
    int nDev = 4;
    int size = 32*1024*1024;
    int devs[nDev] = { 0, 1, 2, 3};
    ncclComm_t comms[nDev];
    int check[1];

    for (int i = 0; i < nDev; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

    //allocating and initializing device buffers
    int** sendbuff = (int**)malloc(nDev * sizeof(int*));
    int** recvbuff = (int**)malloc(nDev * sizeof(int*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


    for (int i = 0; i < nDev; ++i) {
        cudaSetDevice(i);
        cudaMalloc(sendbuff + i, size * sizeof(int));
        cudaMalloc(recvbuff + i, size * sizeof(int));
        cudaStreamCreate(s+i);
        initBuffer<<<1,1>>>(sendbuff[i], size);
    }


    //initializing NCCL
    ncclCommInitAll(comms, nDev, devs);


    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
    ncclGroupStart();
    for (int i = 0; i < nDev; ++i)
        ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
                      comms[i], s[i]);
    ncclGroupEnd();


    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(s[i]);
    }

    cudaMemcpy(check, recvbuff[0], sizeof(int), cudaMemcpyDeviceToHost);
    printf("Value of check %d\n", check[0]);

    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        cudaSetDevice(i);
        cudaFree(sendbuff[i]);
        cudaFree(recvbuff[i]);
    }


    //finalizing NCCL
    for(int i = 0; i < nDev; ++i)
        ncclCommDestroy(comms[i]);


    printf("Success \n");
    return 0;
}
