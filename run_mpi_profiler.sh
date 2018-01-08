#!/usr/bin/env bash


for i in 3 4 6 12; do
    mpirun -np 12 --map-by ppr:${i}:socket -output-filename mpi_result_${i}_core_per_socket python run_mpi_test.py
done

