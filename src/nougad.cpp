#include <R.h>
#include <R_ext/Rdynload.h>
#include <Rmath.h>

#include "unmixscent-cuda.hpp"

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

	GradientDescendCudaAlgorithm<float, UnmixscentBaseSharedKernel<float>> algorithm(exec);
	DataPoints<float> measurements(*dim, *n);
	DataPoints<float> spectra(*dim, *spectraN);
	DataPoints<float> spectraPW(*dim, *spectraN);
	DataPoints<float> spectraNW(*dim, *spectraN);
	DataPoints<float> resultNW(*spectraN, 1);
	DataPoints<float> resultsInitial(*spectraN, *n);
	DataPoints<float> residuals(*dim, *n);

	algorithm.initialize(measurements, spectra, spectraPW, spectraNW, resultNW, resultsInitial, *iterations, *alpha, *acceleration);

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