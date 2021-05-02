#!/apps/R-3.6.0/lib64/R/bin/Rscript

source("/work/pl2669/taq_project/code/global_helpers.r")


args <- commandArgs(trailingOnly = TRUE)
ticker <- args[1]
cores <- as.integer(args[2])

panel_path <- paste("/work/pl2669/taq_project/data/agg_1hr/",ticker,"_1hr.csv", sep="")

#get train/test split
splits <- getTrainTestSplit(panel_path)
train <- splits$train
test <- splits$test

## Fit and forecast with best orders
df_garch <- read.csv(paste("/work/pl2669/taq_project/data/garch_fits/",ticker,"_garch_results.csv", sep=""))

best_arma <- c(df_garch[2, ]$p, df_garch[2, ]$q)
best_garch <- c(df_garch[3, ]$p, df_garch[3, ]$q)
final_spec <- ugarchspec(variance.model=list("garchOrder"=best_garch),
                         mean.model=list("armaOrder"=best_arma, include.mean=TRUE),
                         distribution.model="std")

full_data <- rbind.xts(train, test)

#parallel forecasting
cl <- makeCluster(cores)
forecasts <- ugarchroll(final_spec, full_data, n.ahead=1, forecast.length=length(test),
                        refit.every=1, refit_window="recursize", solver="hybrid",
                        calculate.VaR=FALSE, cluster=cl)
stopCluster(cl)

#gather up the predictions
mu <- list(); sigma <- list()
for (i in 1:length(forecasts@forecast)) {
    obs <- forecasts@forecast[[i]]
    if (obs$converge) {
        mu[[i]] <- obs$y$Mu
        sigma[[i]] <- obs$y$Sigma
    } else {
        mu[[i]] <- NA; sigma[[i]] <- NA
    }
}

#perform interpolation on non-convergence
df_fore <- as.data.frame(test)
df_fore["pred_lret"] <- unlist(mu)
df_fore["pred_vol"] <- unlist(sigma)

message(paste("% of nonconverged forecasts: ",round(dim(df_fore[is.na(df_fore$pred_lret), ])[1]/dim(df_fore)[1], 2), sep=""))

#interpolate with cubic spline, rule is the extrapolation point
df_fore$pred_lret <- na.approx(df_fore$pred_lret, rule=2)
df_fore$pred_vol <- na.approx(df_fore$pred_vol, rule=2)

#save forecast to dataframe
write.csv(df_fore, paste("/work/pl2669/taq_project/data/forecast_res/",ticker,"_forecasts.csv", sep=""))
