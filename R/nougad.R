
#' Non-linear unmixing by gradient descent
#' 
#' Run a gradient descent for each (row) measurement in `mixed`, extracting how
#' much of `spectra` is contained in each measurement.
#' Gradient descent runs for `iters` iterations, with Adam momentum estimation.
#'
#' Additionally, the result may be weighted towards non-negative region in each
#' result dimension by weights `nw`. Residual error is weighted by vectors
#' `rnw` (in case the residual in the dimension is negative) and `rpw`
#' (in case the residual is positive). The latter allows one to implicitly
#' force a non-negative or non-positive residual.
#'
#' The method should behave like OLS for rnw,rpw=1 and nw=0.
#'
#' @param mixed n*d matrix of measurements
#' @param spectra k*d matrix of spectra
#' @param rnw negative weights for residual (converted to vector of size d)
#' @param rpw positive weights for residual (converted to vector of size d)
#' @param nw weights of non-negative learning factor (converted to a
#'           vector of size k)
#' @param start starting points for the gradient descent
#' @param eta,betas,epsilon Adam parameters
#' @param iters number of iterations
#' @return a list with `unmixed` n*k matrix and `residuals` n*d matrix, so that
#'         `mixed = unmixed %*% spectra + residuals`
#' @useDynLib nougad, .registration=True
#' @export
nougad <- function(mixed, spectra,  
  rpw=1, rnw=1, nw=1, start=0,
  eta=0.1, betas=c(0.9,0.999), epsilon=1e-6, iters=250L) {
  if(!is.matrix(mixed)) stop("Mixed must be a matrix")
  n <- nrow(mixed)
  d <- ncol(mixed)
  k <- nrow(spectra)
  if (ncol(spectra) != d) stop("Wrong size of spectra")
  mixed <- t(mixed)
  spectra <- t(spectra)
  rpw <- { tmp <- rep(0, d); tmp[] <- rpw; tmp }
  rnw <- { tmp <- rep(0, d); tmp[] <- rnw; tmp }
  nw <- { tmp <- rep(0, k); tmp[] <- nw; tmp }
  
  x <- matrix(start, ncol=n, nrow=k)
  r <- matrix(0, ncol=n, nrow=d)

  res <- .C("nougad_c",
    n=as.integer(n),
    d=as.integer(d),
    k=as.integer(k),
    iters=as.integer(iters),
    eta=as.single(eta),
    betas=as.single(betas),
    eps=as.single(epsilon),
    s=as.single(spectra),
    rpw=as.single(rpw),
    rnw=as.single(rnw),
    nw=as.single(nw),
    y=as.single(mixed),
    x=as.single(x),
    r=as.single(r))

  list(unmixed = matrix(res$x, n, k, byrow=T),
    residuals = matrix(res$r, n, d, byrow=T))
}
