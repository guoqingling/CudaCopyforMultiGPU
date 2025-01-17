/*
HOW TO USE?
nvcc -I "cuda-samples\Common" -o output_file.exe mysampleforcudacopy.cu
Of course you can use output_file to replace output_file.exe in linux
Author : Eloim, ewppple1999@qq.com
*/

#include <cstdio>
#include <vector>

#include <helper_cuda.h>
#include <helper_timer.h>

#define NUM_ELEMENTS 400000000

__global__ void add(int *a, int *b, int *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        c[idx] = a[idx] + b[idx];
    }
}


int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("  I have %d GPUs.\n\n", deviceCount);

    if (deviceCount < 3) return 0;

    int *d_a;
    cudaSetDevice(0);
    cudaMalloc(&d_a, NUM_ELEMENTS * sizeof(int));
    cudaMemset(d_a, 1, NUM_ELEMENTS * sizeof(int));
    cudaStream_t stream1; cudaEvent_t start1; cudaEvent_t stop1;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaEventCreate(&start1); cudaEventCreate(&stop1);
    // temp output the d_a
    /*
    int *host_buffer1 = new int[NUM_ELEMENTS];
    cudaMemcpy(host_buffer1, d_a, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
    for (int j = 0; j < NUM_ELEMENTS; j++)
    {
        printf("%d ",host_buffer1[j]);
        if ( j % 10 == 9){
            printf("\n");
        }
        if (j % 400 == 399) break;
    }
    */

    int *d_b;
    cudaSetDevice(1);
    cudaMalloc(&d_b, NUM_ELEMENTS * sizeof(int));
    cudaMemset(d_b, 2, NUM_ELEMENTS * sizeof(int));
    cudaStream_t stream2; cudaEvent_t start2; cudaEvent_t stop2;
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    cudaEventCreate(&start2); cudaEventCreate(&stop2);

    int *d_c;
    cudaSetDevice(2);
    cudaMalloc(&d_c, NUM_ELEMENTS * sizeof(int));
    cudaStream_t stream3; cudaEvent_t start3; cudaEvent_t stop3;
    cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking);
    cudaEventCreate(&start3); cudaEventCreate(&stop3);

    int *tmp_a, *tmp_b;
    cudaMalloc(&tmp_a, NUM_ELEMENTS * sizeof(int));
    cudaMalloc(&tmp_b, NUM_ELEMENTS * sizeof(int));


    // copy
    cudaEventRecord(start1, stream1);
    cudaMemcpyAsync(tmp_a, d_a, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToDevice, stream1);
    cudaEventRecord(stop1, stream1);
    cudaStreamSynchronize(stream1);
    cudaEventRecord(start2, stream2);
    cudaMemcpyAsync(tmp_b, d_b, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToDevice, stream2);
    cudaEventRecord(stop2, stream2);
    cudaStreamSynchronize(stream2);

    // add
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_ELEMENTS + threadsPerBlock - 1) / threadsPerBlock;
    add<<<blocksPerGrid, threadsPerBlock>>>(tmp_a, tmp_b, d_c);

    // copy back
    cudaEventRecord(start3, stream3);
    cudaMemcpyAsync(d_a, d_c, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToDevice, stream3);
    cudaEventRecord(stop3, stream3);
    cudaStreamSynchronize(stream3);

    float time_ms1; float time_ms2; float time_ms3;
    cudaEventElapsedTime(&time_ms1, start1, stop1);
    cudaEventElapsedTime(&time_ms2, start2, stop2);
    cudaEventElapsedTime(&time_ms3, start3, stop3);
    double time_s1 = time_ms1 / 1e3;
    double time_s2 = time_ms2 / 1e3;
    double time_s3 = time_ms3 / 1e3;

    double gb1 = NUM_ELEMENTS * sizeof(int) / (double)1e9;
    double gb2 = NUM_ELEMENTS * sizeof(int) / (double)1e9;
    double gb3 = NUM_ELEMENTS * sizeof(int) / (double)1e9;
    double bandwidth1 = gb1 / time_s1;
    double bandwidth2 = gb2 / time_s2;
    double bandwidth3 = gb3 / time_s3;

    printf("  OP1 use %6.02f second and datasize is %6.02f GB, bandwidth is %6.02f GB/s\n",time_s1, gb1, bandwidth1);
    printf("  OP2 use %6.02f second and datasize is %6.02f GB, bandwidth is %6.02f GB/s\n",time_s2, gb2, bandwidth2);
    printf("  OP3 use %6.02f second and datasize is %6.02f GB, bandwidth is %6.02f GB/s\n\n",time_s3, gb3, bandwidth3);

    int *h_result = new int[NUM_ELEMENTS];
    cudaMemcpy(h_result, d_a, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);


    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        printf("%d ",h_result[i]);
        if ( i % 10 == 9){
            printf("\n");
        }
        if (i % 400 == 399) break;
    }

    delete[] h_result;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(tmp_a); cudaFree(tmp_b);
    cudaEventDestroy(start1);cudaEventDestroy(start2);cudaEventDestroy(start3);
    cudaEventDestroy(stop1);cudaEventDestroy(stop2);cudaEventDestroy(stop3);
    cudaStreamDestroy(stream1);cudaStreamDestroy(stream2);cudaStreamDestroy(stream3);

    return 0;
}