
#' Benchmark the unmixing, print some interesting information, plot out some
#' plots
#'
#' @export
nougad.benchmark <- function(n=10240, k=30, d=40,
  n_pheno=20, pheno_positive_probability=0.2,
  plot.k=min(c(k,5)),
  plotf=plot) {


  pheno <- matrix(sample(c(0,1),
      prob=c(1-pheno_positive_probability,pheno_positive_probability),
      replace=T, n_pheno*k),
    n_pheno,k)

  spectra <- matrix(exp(4*runif(k*d)),k,d) * exp(rnorm(k, sd=0.2))

  spectra <- spectra/sqrt(rowSums(spectra^2))
  exprs <- 10^(4*(pheno[sample(n_pheno,n, replace=T),] + rnorm(k*n, sd=0.1)))
  emitted <- exprs %*% spectra
  received <- emitted + rnorm(length(emitted),sd=0.0005*sqrt(rowSums(emitted^2)))
  
  sw <- t(t(spectra^2)/sqrt(colSums(spectra^4)))
  res <- nougad(received, spectra, 2*sw, 1*sw, 5, iters=500)$unmixed

  trans <- function(x)asinh(x/100)

  par(mfrow=c(1,3))
  plotf(main=paste("Original data -",k,"markers in",d,"channels"), trans(exprs[,1:2]), pch='.', xlab='', ylab='')
  plotf(main="OLS", trans(t(lm(t(received)~t(spectra)+0)$coefficients)[,1:2]), pch='.', xlab='', ylab='')
  plotf(main="nougad", trans(res[,1:2]), pch='.', xlab='', ylab='')
}
