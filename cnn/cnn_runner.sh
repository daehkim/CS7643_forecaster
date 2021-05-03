#!/bin/bash

TICKERS=(IBM JNJ VZ XOM)

for tick in ${TICKERS[@]}
do
    grid_run --grid_submit=batch --grid_mem=30g --grid_quiet "./cnn_main.py $tick 200 3 2>&1 | tee /work/pl2669/taq_project_cnn/code/outfiles/${tick}_cnn.txt"
done
