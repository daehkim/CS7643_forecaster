#!/apps/R-3.6.0/lib64/R/bin/Rscript

library(lubridate)
library(rugarch)
library(xts)
library(aTSA)
library(doParallel)
library(dplyr)
#library(pbapply)

getTrainTestSplit <- function(panel_path) {
    df <- read.csv(panel_path)
    df$date <- ymd(df$date)

    ## Traing/Test
    #   * train: [2015/01/01, 2019/01/01)
    #   * test: [2019/01/01, 2020/01/01)
    train_start <- ymd("2015-01-01"); train_end <- ymd("2019-01-01")
    test_end <- ymd("2020-01-01")

    df_train <- subset(df, df$date >= train_start & df$date < train_end)
    df_test <- subset(df, df$date >= train_end & df$date < test_end)

    getDatetime <- function(dft) {ymd_hms(paste(dft$date," ",dft$hour,":00:00", sep=""))}
    df_train["datetime"] <- getDatetime(df_train)
    df_test["datetime"] <- getDatetime(df_test)

    #need to make train and test xts
    train <- xts(df_train[, 4], df_train[, 6])
    test <- xts(df_test[, 4], df_test[, 6])

    colnames(train) <- c("lret"); colnames(test) <- c("lret")

    ## TODO: decide on missing values (impute or drop?)
    #   * Drop NaNs
    #   * Can do some imputations
    train <- train[!is.na(train[, 1]), ]
    test <- test[!is.na(test[, 1]), ]


    return(list("train"=train, "test"=test))
}
