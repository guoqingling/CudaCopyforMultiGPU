# README
This document is for the mysampleforcudacopy.cu file.

## How to run
We use nvcc （Cuda）and some source files in CudaSamples （https://github.com/NVIDIA/cuda-samples）
1. Use `git clone https://github.com/NVIDIA/cuda-samples.git` to download CudaSamples
2. Make sure you have Cuda installed, use `nvcc -V` to check
3. Use `nvcc -I "cuda-samples\Common" -o output_file.exe mysampleforcudacopy.cu` to compile in Windows environment
4. Use `./output_file.exe` to run

## Attention
Our environment have 3 GPUs, and we want to verify the data copy between GPUs.

