#!/apps/R-3.6.0/lib64/R/bin/Rscript

source("/work/pl2669/taq_project/code/garch_helpers.r")


args <- commandArgs(trailingOnly = TRUE)
ticker <- args[1]
cores <- as.integer(args[2])

panel_path <- paste("/work/pl2669/taq_project/data/agg_1hr/",ticker,"_1hr.csv", sep="")
arima_path <- paste("/work/pl2669/taq_project/data/arima_fits/init_arima_fit_",ticker,".rds", sep="")

#get train/test split
splits <- getTrainTestSplit(panel_path)
train <- splits$train
test <- splits$test

#get initial arima fit
arima_fit <- readRDS(arima_path)


## Initial GARCH fit (Parallelized)
#   1. Fit Garch with orders from ARMA
#   2. Fit ARMA again with Garch orders
#   3. Fit Garch again with new ARMA orders (this is the final GARCH to be used)
#       * Ideally, we want these GARCH orders to match ones in (1)

#fit orders in parallel: (p=1, q=1), (p=1, q=2), ... , (p=10, q=10)
registerDoParallel(cores=cores)     #cores=20 for my fittings

#about 3-days of lags (by hour)
#   p > 10 has trouble converging for ARIMA
gp_vec <- c(1:20); gq_vec <- c(1:20)
ap_vec <- c(1:10); aq_vec <- c(1:20)

arima_orders <- c(arima_fit$p, arima_fit$q)

# 1. 
garch_v1 <- 
    foreach(p=gp_vec, .combine="rbind") %:%
        foreach(q=gq_vec, .combine="rbind") %dopar% {
            tmp_res <- fitGarch(train, p, q, arma_orders=arima_orders)
            data.frame(p=p, q=q, aic=tmp_res$aic)
        }

#get best garch fit
best_v1 <- garch_v1[which.min(garch_v1$aic), ]

# 2.
garch_v2 <-
    foreach(p=ap_vec, .combine="rbind") %:%
        foreach(q=aq_vec, .combine="rbind") %dopar% {
            tmp_res <- fitFixedGarch(train, p, q, garch_orders=c(best_v1$p, best_v1$q))
            data.frame(p=p, q=q, aic=tmp_res$aic)
        }

best_v2 <- garch_v2[which.min(garch_v2$aic), ]

# 3.
garch_final <- 
    foreach(p=gp_vec, .combine="rbind") %:%
        foreach(q=gq_vec, .combine="rbind") %dopar% {
            tmp_res <- fitGarch(train, p, q, arma_orders=c(best_v2$p, best_v2$q))
            data.frame(p=p, q=q, aic=tmp_res$aic)
        }

best_final <- garch_final[which.min(garch_final$aic), ]

#combine and save best results together
df_res <- rbind(best_v1, best_v2, best_final)
df_res["version"] <- c(1:3)
df_res <- df_res[, c("version", "aic", "p", "q")]

write.csv(df_res, paste("/work/pl2669/taq_project/data/garch_fits/",ticker,"_garch_results.csv", sep=""), row.names=FALSE)

## residual analysis for best model
#fit with best orders
best_arma <- c(best_v2$p, best_v2$q)
best_garch <- c(best_final$p, best_final$q)
message(paste("ARMA(",best_arma[1],",",best_arma[2],") + GARCH(",best_garch[1],",",best_garch[2],")", sep=""))

final_spec <- ugarchspec(variance.model=list("garchOrder"=best_arma),
                       mean.model=list("armaOrder"=best_garch, include.mean=TRUE),
                       distribution.model="std")
garch_fit <- ugarchfit(final_spec, train, solver="hybrid")

#resid analysis
analysisGarch(garch_fit, ticker)
