#!/apps/R-3.6.0/lib64/R/bin/Rscript

source("/work/pl2669/taq_project/code/global_helpers.r")


calcAcc <- function(df_res, save_path, ticker) {
    #calculate MSE/MAE/Other
    vol_mse <- mean((df_res$pred_vol - df_res$vol)^2)
    vol_mae <- mean(abs(df_res$pred_vol - df_res$vol))
    
    df_acc <- t(data.frame(c(vol_mse, vol_mae)))
    colnames(df_acc) <- c("mse", "mae")
    write.csv(df_acc, paste(save_path,ticker,"_forecast_acc.csv", sep=""), row.names=FALSE)
}

plotForecast <- function(df_res, save_path, ticker) {
    ymin <- min(c(min(df_res$vol), min(df_res$pred_vol)))
    ymax <- max(c(max(df_res$vol), max(df_res$pred_vol)))

    pdf(paste(save_path,ticker,"_forecast_plot.pdf", sep=""), width=20, height=10)
    plot(df_res$vol, type="l", ylim=c(ymin, ymax), xaxt="n", xlab="", ylab="Realized Volatility",
         main=paste("Realized Volatility Forecasts for ",ticker, sep=""))
    lines(df_res$pred_vol, col="red")
    axis(1, at=c(1, dim(df_res)[1]), labels=FALSE)
    text(c(1, dim(df_res)[1]), par("usr")[3] - 0.2,
         labels=c(min(df_res$datetime), max(df_res$datetime)), srt=15, pos=1, cex=.7, xpd=TRUE)
    dev.off()
}

args <- commandArgs(trailingOnly = TRUE)
ticker <- args[1]

panel_path <- paste("/work/pl2669/taq_project/data/agg_1hr/",ticker,"_1hr.csv", sep="")
fore_path <- paste("/work/pl2669/taq_project/data/forecast_res/",ticker,"_forecasts.csv", sep="")

df_panel <- read.csv(panel_path)
df_fore <- read.csv(fore_path)
colnames(df_fore) <- c("datetime", "lret", "pred_lret", "pred_vol")

#merge realized vol and predicted vol
df_panel$date <- ymd(df_panel$date)
df_fore$datetime <- ymd_hms(df_fore$datetime)
df_fore["date"] <- date(df_fore$datetime)
df_fore["hour"] <- hour(df_fore$datetime)

df_panel <- df_panel[, c("date", "hour", "vol")]
df_fore <- df_fore %>% left_join(df_panel, by=c("date", "hour"))

save_path <- "/work/pl2669/taq_project/data/acc_metrics/"
calcAcc(df_fore, save_path, ticker)
plotForecast(df_fore, save_path, ticker)
