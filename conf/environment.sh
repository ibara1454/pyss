#!/usr/bin/env bash

# Disable blas multi-threading, this would cause undefined behavior
# when applying multi-processing with MPI
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

exit 0
