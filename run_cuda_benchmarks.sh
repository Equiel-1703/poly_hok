#!/bin/bash

RUNS_PER_BENCHMARK=30 # CHANGE TO 30 LATER ----------------------------------------------------<<<
USE_TEST_VALUES=false # Set to true to use smaller test values
MEASURE_TRUE_TIME=false # Set to true to measure the entire execution time of the benchmark with the "time" command

# Define directories for benchmarks
BENCHMARKS_DIR="benchmarks"
CUDA_BENCHMARKS_DIR="$BENCHMARKS_DIR/cuda"

# --- Functions for running benchmarks ---
run_CUDA_benchmark() {
    local benchmark_name="$1"
    local benchmark_input="$2"
    
    local output_name="${benchmark_name%.cu}.out"
    
    # Check if the compiled file does not exist
    if [ ! -f "$CUDA_BENCHMARKS_DIR/$output_name" ]; then
        # Compile the CUDA benchmark if it does not exist
        nvcc -o "$CUDA_BENCHMARKS_DIR/$output_name" "$CUDA_BENCHMARKS_DIR/$benchmark_name"
    fi
    
    local i # Loop variable
    
    # Run the compiled CUDA benchmark
    if [ "$MEASURE_TRUE_TIME" = true ]; then
        for ((i=1; i<=RUNS_PER_BENCHMARK; i++)); do
            time "./$CUDA_BENCHMARKS_DIR/$output_name" $benchmark_input 2>&1
        done
    else
        for ((i=1; i<=RUNS_PER_BENCHMARK; i++)); do
            "./$CUDA_BENCHMARKS_DIR/$output_name" $benchmark_input 2>&1
        done
    fi
}

# This function will invoke the benchmark with the provided inputs
# It will provide the test inputs if the flag USE_TEST_VALUES is true
#
# CALL SIGNATURE:
#       run_benchmark <benchmark_file> <title> <inputs> <test_inputs>
run_benchmark() {
    local benchmark_file=$1
    local title=$2
    local inputs=$3
    local test_inputs=$4
    
    # If we are using test values, "inputs" will have the test values instead
    if [ "$USE_TEST_VALUES" = true ]; then
        inputs="$test_inputs"
    fi
    
    echo -e "$title\n"
    
    local val
    for val in $inputs; do
        run_CUDA_benchmark "$benchmark_file" "$val"
    done
    echo ""
}

# ------------------ Script Start ------------------
echo -e "- CUDA Benchmarks Results -"
date
echo -e "Tests conducted by: Andre R. Du Bois & Henrique G. Rodrigues\n"
echo -e "Runs per benchmark: $RUNS_PER_BENCHMARK\n"
echo ""  # Add a blank line for readability

# ------------------ Dot Product Benchmark ------------------
BENCH_FILE="dot_product.cu"
TITLE="Dot Product (DP) benchmark"
INPUTS="800000000 900000000 1000000000"
TEST_INPUTS="1024 2048 4096"

run_benchmark "$BENCH_FILE" "$TITLE" "$INPUTS" "$TEST_INPUTS"

# ------------------ Julia Benchmark ------------------
BENCH_FILE="julia.cu"
TITLE="Julia (JL) benchmark"
INPUTS="10000 17000 20000"
TEST_INPUTS="512 1024 2048"

run_benchmark "$BENCH_FILE" "$TITLE" "$INPUTS" "$TEST_INPUTS"

# ------------------ MM Benchmarks ------------------
BENCH_FILE="mm.cu"
TITLE="Matrix Multiplication (MM) benchmark"
INPUTS="10000 15000 20000"
TEST_INPUTS="128 256 512"

run_benchmark "$BENCH_FILE" "$TITLE" "$INPUTS" "$TEST_INPUTS"

# ------------------ NBody Benchmarks ------------------
# BENCH_FILE="nbodies.cu"
# TITLE="nBodies (NB) benchmark"
# INPUTS="100000 200000 400000"
# TEST_INPUTS="128 256 512"

# run_benchmark "$BENCH_FILE" "$TITLE" "$INPUTS" "$TEST_INPUTS"

# ------------------ Nearest Neighbor Benchmarks ------------------
BENCH_FILE="nearest_neighbor_double.cu"
TITLE="Nearest Neighbor (NN) benchmark (double precision)"
INPUTS="300000000 400000000 500000000"
TEST_INPUTS="1024 2048 4096"

run_benchmark "$BENCH_FILE" "$TITLE" "$INPUTS" "$TEST_INPUTS"

# ------------------ Raytracer Benchmarks ------------------
BENCH_FILE="raytracer.cu"
TITLE="Raytracer (RT) benchmark"
INPUTS="10000 17000 20000"
TEST_INPUTS="512 1024 2048"

run_benchmark "$BENCH_FILE" "$TITLE" "$INPUTS" "$TEST_INPUTS"

# ------------------ Saxpy Benchmarks ------------------
# BENCH_FILE="saxpy.cu"
# TITLE="Saxpy (SP) benchmark"
# INPUTS="300000000 400000000 500000000"
# TEST_INPUTS="512 1024 2048"

# run_benchmark "$BENCH_FILE" "$TITLE" "$INPUTS" "$TEST_INPUTS"

# ------------------ Script End ------------------