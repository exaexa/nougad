#' Benchmark the unmixing, print some interesting information, plot out some
#' plots
#'
#' @export
nougad.benchmark <- function(
    n = 10240, k = 30, d = 40,
    rnw = 3, rpw = 1, nw = 5,
    rnwf = function(sp) rnw, rpwf = function(sp) rpw, nwf = function(sp) nw,
    iters = 250L, alpha = 0.05, accel = 1, threads = 0L,
    n_pheno = 20, pheno_positive_probability = 0.2,
    spectra_sdev = 0.2, spectra_intensity = 4,
    positive_population_intensity = 4,
    population_intensity_sdev = 0.1,
    crosstalk_sdev = 0.0005,
    asinh_cofactor = 100,
    plot.k = min(c(k, 5)),
    plotf = plot) {
  pheno <- matrix(
    sample(c(0, 1),
      prob = c(1 - pheno_positive_probability, pheno_positive_probability),
      replace = T, n_pheno * k
    ),
    n_pheno, k
  )

  spectra <- matrix(exp(spectra_intensity * runif(k * d)), k, d) * exp(rnorm(k, sd = spectra_sdev))

  spectra <- spectra / sqrt(rowSums(spectra^2))
  exprs <- 10^(positive_population_intensity * (pheno[sample(n_pheno, n, replace = T), ] + rnorm(k * n, sd = population_intensity_sdev)))
  emitted <- exprs %*% spectra
  received <- emitted + rnorm(length(emitted), sd = crosstalk_sdev * sqrt(rowSums(emitted^2)))

  olst <- system.time(olsres <- t(lm(t(received) ~ t(spectra) + 0)$coefficients))
  t <- system.time(res <- nougad(mixed = received, spectra = spectra, rnw = rnwf(spectra), rpw = rpwf(spectra), nw = nwf(spectra), alpha = alpha, accel = accel, iters = iters, threads = threads)$unmixed)

  trans <- function(x) asinh(x / asinh_cofactor)

  par(mfrow = c(1, 3))
  plotf(main = paste("Original data -", k, "markers in", d, "channels"), trans(exprs[, 1:2]), pch = '.', xlab = '', ylab = '')
  plotf(main = paste0("OLS (elapsed: ", round(olst['elapsed'], digits = 3), ")"), trans(olsres[, 1:2]), pch = '.', xlab = '', ylab = '')
  plotf(main = paste0("nougad (elapsed: ", round(t['elapsed'], digits = 3), ")"), trans(res[, 1:2]), pch = '.', xlab = '', ylab = '')
}
