#!/apps/R-3.6.0/lib64/R/bin/Rscript

source("/work/pl2669/taq_project/code/global_helpers.r")


garchForecast <- function(iter, spec, train_data, test_data) {
    data <- train_data
    if (iter > 1) {
        data <- c(data, test_data[1:(iter-1)])
    }

    fitted <- TRUE

    #NOTE: For some reason pbapply has trouble picking up loaded libraries sometimes
    #   * Could be my compute cluster. Need to reference actual library::method,
    #     i.e. rugarch::ugarchfit
    model <- tryCatch(rugarch::ugarchfit(spec, data, solver="hybrid"),
                      error=function(e){message("Error fitting garch. Cannot run forecast.")
                                        message(e)
                                        fitted <<- FALSE},
                      warning=function(w){message("Warning fitting garch. Cannot run forecast.")
                                          message(w)
                                          fitted <<- FALSE})

    mu <- NA; sigma <- NA
    if (fitted) {
        fore <- rugarch::ugarchforecast(model, n.ahead=1)
        mu <- fore@forecast$seriesFor
        sigma <- fore@forecast$sigmaFor
    }
    return(list("mu"=mu, "sigma"=sigma))
}
