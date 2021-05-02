#!/apps/R-3.6.0/lib64/R/bin/Rscript

library(progress)
library(doParallel)
library(foreach)
library(dplyr)
library(RPostgres)
library(highfrequency)
library(lubridate)
library(quantmod)

#NOTE: Code adapted from: https://onnokleen.de/post/taq_via_wrds/


getDates <- function() {
    res <- dbSendQuery(wrds, "select distinct table_name
                       from information_schema.columns
                       where table_schema='taqmsec'
                       order by table_name")
    df_dates <- dbFetch(res, n = -1)
    dbClearResult(res)

    dates_trades <- df_dates %>%
        filter(grepl("ctm",table_name), !grepl("ix_ctm",table_name)) %>%
        mutate(table_name = substr(table_name, 5, 12)) %>%
        unlist()

    df_tdates <- data.frame(dates_trades)
    colnames(df_tdates) <- c("milli_sdate")
    df_tdates$milli_ldate <- ymd(df_tdates$milli_sdate)
    return(df_tdates)
}

getDataHelper <- function(ticker, dd, agg_interval, sd_interval) {
    ## We clean the data during this step, and follow the Barndorff-Nielson et al. (2009).
    ## Also utilize the highfrequency R package to clean and aggregate data.
    
    #restrict to uncorrected data
    res <- dbSendQuery(wrds,
        paste0("select concat(date, ' ',time_m) as DT,", 
                         " ex, sym_root, sym_suffix, price, size, tr_scond",
                         " from taqmsec.ctm_", dd,
                         " where sym_root = '", ticker, "'",
                         " and price != 0 and tr_corr = '00'"))
    df_stock <- dbFetch(res, n = -1)
    dbClearResult(res)

    #get observations for open hours only, convert to data.table (more eff. vs df)
    dt_stock <- df_stock %>%
        rename(DT = dt, PRICE = price, SYM_ROOT = sym_root, SYM_SUFFIX = sym_suffix, 
               SIZE = size, EX = ex, COND = tr_scond) %>%
        mutate(DT = lubridate::ymd_hms(DT, tz = "UTC")) %>%
        data.table::as.data.table() %>%
        exchangeHoursOnly() %>%              #only observations from 9:30 to 16:00
        tradesCondition() %>%                #from highfrequency package
        mergeTradesSameTimestamp() 

    #only concerned with price and volume at the moment
    dt_stock <- dt_stock[, c("DT", "SYMBOL", "PRICE", "SIZE")]

    #aggregate to minute level 
    dt_agg_stock <- dt_stock %>% aggregateTrades(alignPeriod = agg_interval)

    #calculate returns and volatility
    dt_agg_stock <- dt_agg_stock %>% mutate(lret=100 * (log(PRICE) - log(lag(PRICE, order_by=DT))))
    dt_agg_stock["lret_sq"] <- dt_agg_stock$lret^2

    #dt_agg_stock["vol"] <- runSD(dt_agg_stock$lret, n=sd_interval)
    colnames(dt_agg_stock) <- c("datetime", "ticker", "price", "size", "lret", "lret_sq")

    return(dt_agg_stock)
}

getData <- function(ticker, dd, agg_interval) {
    #a helper to save the data while using parallel. Sometimes wrds fails, so catch and rerun.
    tryCatch(
        expr={
            dt_stock <- getDataHelper(ticker, dd, agg_interval)

            #create real value, and we also create several seasonal variables
            dt_stock$datetime <- ymd_hms(dt_stock$datetime, tz="EST")
            dt_stock["date"] <- date(dt_stock$datetime)
            dt_stock["hour"] <- hour(dt_stock$datetime)
            dt_stock <- dt_stock[, c(-1, -2, -3)]

            dt_stock <- dt_stock %>% group_by(date, hour) %>% summarise_all(sum, na.rm=TRUE)
            dt_stock["vol"] <- sqrt(dt_stock$lret_sq) 
            dt_stock <- dt_stock[, -5]

            write.csv(dt_stock, paste("/work/pl2669/taq_project/data/tmp/",ticker,"_",dd,".csv", sep=""), row.names=FALSE)
        },
        error=function(e){
            message(paste("Error for date: ",dd, sep=""))
            #print(e)
            #getData(ticker, dd, agg_interval)
        }
    )
}

##############
#####MAIN#####
##############

#connect to wrds database
wrds <- dbConnect(Postgres(),
                  host = 'wrds-pgdata.wharton.upenn.edu',
                  port = 9737,
                  dbname = 'wrds',
                  sslmode = 'require',
                  user = 'pl2669')

args <- commandArgs(trailingOnly = TRUE)
ticker <- args[1]
agg_interval <- as.integer(args[2])        #default 5 (minute)

df_tdates <- getDates()
df_tdates <- subset(df_tdates, milli_ldate < ymd("2020-01-01"))
tdates <- df_tdates$milli_sdate
tdates <- as.character(tdates)

#getData(ticker, '20030924', agg_interval, sd_interval)

#pb <- progress_bar$new(total=length(tdates), force=TRUE)
#for (i in 1:length(tdates)) {
#    pb$tick()
#    getData(ticker, tdates[i], agg_interval)
#}

#parallel computing
registerDoParallel(4)
print(paste("Number of workers: ",getDoParWorkers(), sep=""))

foreach(i=1:length(tdates)) %dopar% getData(ticker, tdates[i], agg_interval)   #wrds kicks with too many calls
