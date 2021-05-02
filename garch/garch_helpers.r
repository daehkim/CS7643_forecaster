#!/apps/R-3.6.0/lib64/R/bin/Rscript

source("/work/pl2669/taq_project/code/global_helpers.r")


fitGarch <- function(xts_data, p, q, arma_orders=c(0,0)) {
    # we use Akaike information criteria to find a resonable fit
    final_aic <- Inf

    message(paste("iteration = ",p,q, "\n"), sep="")

    #distribution is standard-t, may want to experiment with this if time
    spec <- ugarchspec(variance.model=list("garchOrder"=c(p,q)),
                       mean.model=list("armaOrder"=c(arma_orders[1], arma_orders[2]), include.mean=TRUE),
                       distribution.model="std") 

    converged <- TRUE

    garch_fit <- tryCatch(ugarchfit(spec, xts_data, solver="hybrid"),
                    error=function(e){message(paste("Error fitting GARCH(",p,",",q,")", sep=""))
                                      converged <<- FALSE},
                    warning=function(w){message(paste("Warning fitting GARCH(",p,",",q,")", sep=""))
                                        converged <<- FALSE})

    #if successfully fit garch get AIC
    if (converged) {
        curr_aic <- infocriteria(garch_fit)[1]     #Akaike
        if (curr_aic < final_aic) {
            final_aic <- curr_aic
        }
    }
    return(list("aic"=final_aic))
}

fitFixedGarch <- function(xts_data, p, q, garch_orders=c(0,0)) {
    # We use Akaike information criteria to find a resonable fit
    final_aic <- Inf
    
    message(paste("iteration = ",p,q, "\n"), sep="")
    
    #distribution is standard-t, may want to experiment with this if time
    spec <- ugarchspec(variance.model=list("garchOrder"=c(garch_orders[1], garch_orders[2])),
                       mean.model=list("armaOrder"=c(p, q), include.mean=TRUE),
                       distribution.model="std")
   
    converged <- TRUE
 
    garch_fit <- tryCatch(ugarchfit(spec, xts_data, solver="hybrid"),
                    error=function(e){message(paste("Error fitting GARCH(",p,",",q,")", sep=""))
                                      converged <<- FALSE},
                    warning=function(w){message(paste("Warning fitting GARCH(",p,",",q,")", sep=""))
                                        converged <<- FALSE})
    
    #if successfully fit garch get AIC
    if (converged) {
        curr_aic <- infocriteria(garch_fit)[1]     #Akaike
        if (curr_aic < final_aic) {
            final_aic <- curr_aic
        }
    }
    return(list("aic"=final_aic))
}

analysisGarch <- function(garch_fit, ticker) {
    resids <- residuals(garch_fit)

    pdf(paste("/work/pl2669/taq_project/data/garch_fits/",ticker,"_garch_resids_analysis.pdf", sep=""))
    par(mfcol=c(3,1), cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
    acf(resids, main="ACF of GARCH Residuals")
    acf(resids^2, main="ACF of Squared GARCH Residuals")
    qqnorm(resids)
    dev.off()

    bt_res <- Box.test(resids, lag=10, type='Ljung')
    bt_res2 <- Box.test(resids^2, lag=10, type='Ljung')

    df_bt <- rbind(c(as.numeric(bt_res[1]), as.numeric(bt_res[2]), as.numeric(bt_res[3])),
                   c(as.numeric(bt_res2[1]), as.numeric(bt_res2[2]), as.numeric(bt_res2[3])))
    colnames(df_bt) <- c("X-sq", "df", "p-value")
    write.csv(df_bt, paste("/work/pl2669/taq_project/data/garch_fits/",ticker,"_garch_resids_boxtests.csv", sep=""), row.names=FALSE)
}

