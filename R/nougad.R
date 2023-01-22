#' Non-linear unmixing by gradient descent
#'
#' Run a gradient descent for each (row) measurement in `mixed`, extracting how
#' much of `spectra` is contained in each measurement.
#' Gradient descent runs for `iters` iterations, with learning rate `alpha` and
#' AdaProp-style acceleration factor `accel` in each dimension.
#'
#' Additionally, the result may be weighted towards non-negative region in each
#' result dimension by weights `nw`. Influence of each input measurement on
#' each output parameter is weighted by matrices `rnw` (in case the residual in
#' the dimension is negative) and `rpw` (in case the residual is positive). The
#' latter allows one to implicitly force a non-negative or non-positive
#' residual.
#'
#' The method should behave like OLS for rnw,rpw=1 and nw=0.
#'
#' Caveat: Row and column names of all matrices are ignored, the correct order
#' of channels/markers must be ensured manually.
#'
#' @param mixed n*d matrix of measurements
#' @param spectra k*d matrix of spectra, norm of rows must be 1.
#' @param rnw negative weights for residual, will be converted to vector of length d
#' @param rpw positive weights for residual, will be converted to vector of length d
#' @param nw weights of non-negative learning factor, gets converted to a
#'           vector of size k
#' @param start starting points for the gradient descent
#' @param alpha learning rate, preferably low to prevent numeric problems
#' @param accel acceleration factor applied independently for each dimension if
#'              the convergence direction in that dimension is the same as in
#'              the last iteration.
#' @param iters number of iterations
#' @param threads number of threads to use for computation, defaults to 0 (auto-detection), 1 disables all threading.
#' @return a list with `unmixed` n*k matrix and `residuals` n*d matrix, so that
#'         `mixed = unmixed %*% spectra + residuals`
#' @useDynLib nougad, .registration=True
#' @export
nougad <- function(
    mixed, spectra,
    rnw = 1, rpw = 1, nw = 1, start = 0,
    alpha = 0.01, accel = 1, iters = 250L, threads = 0L) {
  if (!is.matrix(mixed)) stop("Mixed must be a matrix")
  n <- nrow(mixed)
  d <- ncol(mixed)
  k <- nrow(spectra)
  if (ncol(spectra) != d) stop("Wrong size of spectra")
  mixed <- t(mixed)
  spectra <- t(spectra)
  rnw <- rep(rnw, length.out = d)
  rpw <- rep(rpw, length.out = d)
  nw <- {
    tmp <- rep(0, k)
    tmp[] <- nw
    tmp
  }

  x <- matrix(start, ncol = n, nrow = k)
  r <- matrix(0, ncol = n, nrow = d)

  res <- .C("nougad_c",
    threads = as.integer(threads),
    n = as.integer(n),
    d = as.integer(d),
    k = as.integer(k),
    iters = as.integer(iters),
    alpha = as.single(alpha),
    accel = as.single(accel),
    s = as.single(spectra),
    rnw = as.single(t(rnw * spectra)),
    rpw = as.single(t(rpw * spectra)),
    nw = as.single(nw),
    y = as.single(mixed),
    x = as.single(x),
    r = as.single(r)
  )

  list(
    unmixed = matrix(res$x, n, k, byrow = T),
    residuals = matrix(res$r, n, d, byrow = T)
  )
}
