#!/apps/R-3.6.0/lib64/R/bin/Rscript

source("/work/pl2669/taq_project/code/global_helpers.r")

args <- commandArgs(trailingOnly = TRUE)
ticker <- args[1]

panel_path <- paste("/work/pl2669/taq_project/data/agg_1hr/",ticker,"_1hr.csv", sep="")

#unaltered panel
df <- read.csv(panel_path)
df$date <- ymd(df$date)
train_start <- ymd("2015-01-01"); train_end <- ymd("2019-01-01")
test_end <- ymd("2020-01-01")

df_train <- subset(df, df$date >= train_start & df$date < train_end)
df_test <- subset(df, df$date >= train_end & df$date < test_end)


#get train/test split as xts objects
splits <- getTrainTestSplit(panel_path)
train <- splits$train
test <- splits$test

#get best orders for ARIMA+GARCH
df_garch <- read.csv(paste("/work/pl2669/taq_project/data/garch_fits/",ticker,"_garch_results.csv", sep=""))

best_arma <- c(df_garch[2, ]$p, df_garch[2, ]$q)
best_garch <- c(df_garch[3, ]$p, df_garch[3, ]$q)
final_spec <- ugarchspec(variance.model=list("garchOrder"=best_garch),
                         mean.model=list("armaOrder"=best_arma, include.mean=TRUE),
                         distribution.model="std")

#fit garch and get sigmas
model <- ugarchfit(final_spec, train, solver="hybrid")
df_train["garch_vol"] <- model@fit$sigma
df_test["garch_vol"] <- NA

df <- rbind(df_train, df_test)

write.csv(df, paste("/work/pl2669/taq_project/data/agg_1hr_garchvol/",ticker,"_garchvol_1hr.csv", sep=""),
          row.names=FALSE)
