#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

echo "Running in directory: $(pwd)"

echo "Creating directory minitorch/cuda_kernels..."
mkdir -p minitorch/cuda_kernels

echo "Compiling combine.cu..."
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC

echo "Compiling softmax_kernel.cu..."
nvcc -o minitorch/cuda_kernels/softmax_kernel.so --shared src/softmax_kernel.cu -Xcompiler -fPIC

echo "Compiling layernorm_kernel.cu..."
nvcc -o minitorch/cuda_kernels/layernorm_kernel.so --shared src/layernorm_kernel.cu -Xcompiler -fPIC

echo "Compilation finished successfully."
