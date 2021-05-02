#!/bin/bash

#NOTE: This script expects to launch on a computing cluster (e.g. see grid commands below)

to_run=$1
TICKERS=(AAPL IBM VZ JNJ XOM)

for tick in ${TICKERS[@]}
do
    if [[ $to_run -eq 1 ]]
    then
        #fit arima
        grid_run --grid_submit=batch --grid_mem=200g --grid_ncpus=20 --grid_quiet "./init_arima_fit.r $tick 20 0 &> /work/pl2669/taq_project/code/outfiles/${tick}_arima.txt"
    elif [[ $to_run -eq 2 ]]
    then
        #fit (iterative) garch
        grid_run --grid_submit=batch --grid_mem=200g --grid_ncpus=20 --grid_quiet "./init_garch_fit.r $tick 20 &> /work/pl2669/taq_project/code/outfiles/${tick}_garch.txt"
    else
        #forecast
        grid_run --grid_submit=batch --grid_mem=200g --grid_ncpus=20 --grid_quiet "./forecast.r $tick 20 &> /work/pl2669/taq_project/code/outfiles/${tick}_forecast.txt"
    fi
done
