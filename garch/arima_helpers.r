#!/apps/R-3.6.0/lib64/R/bin/Rscript

source("/work/pl2669/taq_project/code/global_helpers.r")


prelimAnalysis <- function(train, ticker) {
    #these tests are for stationarity of our data, and homoskedasticity
    pdf(paste("/work/pl2669/taq_project/data/plots/prelim_plots/",ticker,"_ACF_PACF.pdf", sep=""))
    par(mfrow=c(2,2)) 
    lret_acf <- acf(train, main="ACF: log-returns")
    lret_pacf <- pacf(train, main="PACF: log-returns")
    lret2_acf <- acf(train^2, main="ACF: squared log-returns")
    lret2_pacf <- pacf(train^2, main="PACF: squared log-returns")
    dev.off()

    #Box-Ljung Test. The null: data are idependently distributed (i.e. uncorrelated/WN) 
    sink(paste("/work/pl2669/taq_project/data/plots/prelim_plots/boxljung_test_",ticker,".txt", sep=""))
    print(Box.test(train$lret, type="Ljung"))
    print(Box.test(train$lret^2, type="Ljung"))
    sink()
}

fitArima <- function(xts_data, p, q, d) {
    # we use Akaike information criteria to find a resonable fit
    final_aic <- Inf
   
    message(paste("iteration = ",p,d,q, "\n"), sep="")

    converged <- TRUE

    arima_fit <- tryCatch(arima(xts_data, order=c(p, d, q)),
                    error=function(e){message(paste("Error fitting ARIMA(",p,",",d,",",q,")", sep=""))
                                      converged <<- FALSE},
                    warning=function(w){message(paste("Warning fitting ARIMA(",p,",",d,",",q,")", sep=""))
                                        converged <<- FALSE})
    
    #if successfully fit arima get AIC
    if (converged) {
        curr_aic <- AIC(arima_fit)
        if (curr_aic < final_aic) {
            final_aic <- curr_aic
        }
    }
    return(list("aic"=final_aic))
}

analysisArima <- function(xts_data, orders, ticker) {
    # We test for the presence of heteroskedasticity here
    final_arima <- arima(xts_data, order=orders)

    #box tests for residuals (fitdf subtracts the degrees of freedom (p + q))
    #we also restrict lag = lag - fitdf = 3, which gives at least 3 df to chi-square.
    #this is consistent with what Hyndman and Athanasopoulos do.
    fitdf <- sum(orders)
    lags <- fitdf + 3 
    resids <- residuals(final_arima)
    sink(paste("/work/pl2669/taq_project/data/arima_fits/boxljung_test_resids_",ticker,".txt", sep=""))
    print(Box.test(resids, lag=lags, fitdf=fitdf))
    print(Box.test(resids^2, lag=lags, fitdf=fitdf))
    sink()

    #arch.lm test. In both of the following tests, the null is squared-residuals are WN.
    #   * Portmanteau Q test
    #   * Lagrange Multiplier 
    pdf(paste("/work/pl2669/taq_project/data/arima_fits/init_arima_fit_",ticker,"_archLMtest.pdf", sep=""))
    arch.test(final_arima)
    dev.off()
}

