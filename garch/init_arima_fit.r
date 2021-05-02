#!/apps/R-3.6.0/lib64/R/bin/Rscript

source("/work/pl2669/taq_project/code/arima_helpers.r")

args <- commandArgs(trailingOnly = TRUE)
ticker <- args[1]
cores <- args[2]
do_prelim <- as.integer(args[3])

panel_path <- paste("/work/pl2669/taq_project/data/agg_1hr/",ticker,"_1hr.csv", sep="")

#get train/test split
splits <- getTrainTestSplit(panel_path)
train <- splits$train
test <- splits$test

## Prelimnary analysis (stationarity)
#   * ACF, PACF
#   * Ljung-Box
if (as.logical(do_prelim)) {
    prelimAnalysis(train, ticker)
}

## Fit inital ARIMA on training data (Parallelized)
registerDoParallel(cores=cores)     #cores=20 for my fittings

#about 3-days of lags (by hour)
#   p>10 has trouble converging with higher orders
p_vec <- c(1:10); q_vec <- c(1:20)

arima_res <-
    foreach(p=p_vec, .combine="rbind") %:%
        foreach(q=q_vec, .combine="rbind") %dopar% {
            tmp_res <- fitArima(train, p, q, d=0)
            data.frame(p=p, q=q, aic=tmp_res$aic)
        }

#get best arima by aic
best_arima <- arima_res[which.min(arima_res$aic), ]

# analysis of arima fit
orders <- c(best_arima$p, 0, best_arima$q)
saveRDS(best_arima, paste("/work/pl2669/taq_project/data/arima_fits/init_arima_fit_",ticker,".rds", sep=""))
analysisArima(train, orders, ticker)
