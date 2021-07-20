#include <R.h>
#include <R_ext/Rdynload.h>
#include <Rmath.h>

#include "nougad.hpp"

void nougad_c(const int* n, const int* dim, const int* spectraN, const int* iterations, const float* alpha, const float* acceleration,
			  const float* s_dk, const float* spw_dk, const float* snw_dk, const float* nw_k, const float* y_dn, float* x_kn, float* r_dn)
{
	CudaExecParameters exec;
	{
		cudaDeviceProp props;
		CUCH(cudaGetDeviceProperties(&props, 0));

		exec.blockSize = 64;
		exec.sharedMemorySize = (unsigned int)props.sharedMemPerBlock;
	}

	GradientDescendCudaAlgorithm<float, NougadBaseSharedKernel<float>> algorithm(exec);
	DataPoints<float> measurements(*dim, *n, const_cast<float*>(y_dn));
	DataPoints<float> spectra(*dim, *spectraN, const_cast<float*>(s_dk));
	DataPoints<float> spectraPW(*dim, *spectraN, const_cast<float*>(spw_dk));
	DataPoints<float> spectraNW(*dim, *spectraN, const_cast<float*>(snw_dk));
	DataPoints<float> resultNW(*spectraN, 1, const_cast<float*>(nw_k));
	DataPoints<float> resultsInitial(*spectraN, *n, x_kn);
	DataPoints<float> residuals(*dim, *n, r_dn);

	algorithm.initialize(measurements, spectra, spectraPW, spectraNW, resultNW, resultsInitial, residuals, *iterations, *alpha, *acceleration);

	algorithm.run();
	(void)algorithm.getResults();
	algorithm.cleanup();
}

static const R_CMethodDef cMethods[] = { { "nougad_c", (DL_FUNC)&nougad_c, 13 }, { NULL, NULL, 0 } };

void R_init_nougad(DllInfo* info)
{
	R_registerRoutines(info, cMethods, NULL, NULL, NULL);
	R_useDynamicSymbols(info, FALSE);
}