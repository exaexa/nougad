#pragma once

#include <algorithm>

#include "interface.hpp"
#include "unmixscent.cuh"
#include <cuda/cuda.hpp>

/**
 * Serial implementation of GradientDescend algorithm.
 */
template <typename F, class KERNEL>
class GradientDescendCudaAlgorithm : public IGradientDescendAlgorithm<F>
{
protected:
	bpp::CudaBuffer<F> mCuPoints, mCuSpectra, mCuSpectraPositiveWeights, mCuSpectraNegativeWeights, mCuResultWeights, mCuResultResiduals, mCuResult, mCuGradientMemory;

private:
	CudaExecParameters& mCudaExec;
	const DataPoints<F>*mPoints, *mSpectra, *mSpectraPositiveWeights, *mSpectraNegativeWeights, *mResultWeights;
	bool mResultsLoaded;

public:
	GradientDescendCudaAlgorithm(CudaExecParameters& cudaExec)
		: mCudaExec(cudaExec),
		  mPoints(nullptr),
		  mSpectra(nullptr),
		  mSpectraPositiveWeights(nullptr),
		  mSpectraNegativeWeights(nullptr),
		  mResultWeights(nullptr),
		  mResultsLoaded(false)
	{}

	virtual void initialize(const DataPoints<F>& points, const DataPoints<F>& spectra, const DataPoints<F>& spectraPositiveWeights,
							const DataPoints<F>& spectraNegativeWeights, const DataPoints<F>& resultWeights, DataPoints<F>& initialResult,
							const std::size_t iterations, const F alpha, const F acceleration)
	{
		std::size_t devices = bpp::CudaDevice::count();
		if (devices == 0)
		{
			throw bpp::RuntimeError("No CUDA devices found!");
		}

		IGradientDescendAlgorithm::initialize(points, spectra, spectraPositiveWeights, spectraNegativeWeights, resultWeights, initialResult,
											  iterations, alpha, acceleration);
		mPoints = &points;
		mSpectra = &spectra;
		mSpectraPositiveWeights = &spectraPositiveWeights;
		mSpectraNegativeWeights = &spectraNegativeWeights;
		mResultWeights = &resultWeights;
		mResult = std::move(initialResult);
	}

	void prepareInputs() override
	{
		CUCH(cudaSetDevice(0));

		mCuPoints.realloc(this->mN * this->mDim);
		mCuSpectra.realloc(this->mSpectrumN * this->mDim);
		mCuSpectraPositiveWeights.realloc(this->mSpectrumN * this->mDim);
		mCuSpectraNegativeWeights.realloc(this->mSpectrumN * this->mDim);
		mCuResultWeights.realloc(this->mSpectrumN);
		mCuResult.realloc(this->mN * this->mSpectrumN);
		mCuResultResiduals.realloc(this->mN * this->mDim);
		mCuGradientMemory.realloc(this->mN * this->mSpectrumN);

		mCuPoints.write(mPoints->data(), this->mN * this->mDim);
		mCuSpectra.write(mSpectra->data(), this->mSpectrumN * this->mDim);
		mCuSpectraPositiveWeights.write(mSpectraPositiveWeights->data(), this->mSpectrumN * this->mDim);
		mCuSpectraNegativeWeights.write(mSpectraNegativeWeights->data(), this->mSpectrumN * this->mDim);
		mCuResultWeights.write(mResultWeights->data(), this->mSpectrumN);
		mCuResult.write(mResult.data(), this->mN * this->mSpectrumN);
		mCuGradientMemory.memset(0);

		mResultsLoaded = false;
	}


	void run() override
	{
		GradientDescendProblemInstance<F> gdProblem(*mCuPoints, *mCuSpectra, *mCuSpectraPositiveWeights, *mCuSpectraNegativeWeights,
													*mCuResultWeights, *mCuResult, *mCuResultResiduals, this->mDim, this->mN, this->mSpectrumN,
													this->mIterations, this->mAlpha, this->mAcceleration, *mCuGradientMemory);
		KERNEL::run(gdProblem, mCudaExec);
		CUCH(cudaDeviceSynchronize());
	}

	const ResultT getResults() override
	{
		if (!mResultsLoaded)
		{
			mCuResult.read(this->mResult.data());
			mCuResultResiduals.read(this->mResultResiduals.data());
			mResultsLoaded = true;
		}

		return IGradientDescendAlgorithm::getResults();
	}

	void cleanup() override
	{
		mCuPoints.free();
		mCuSpectra.free();
		mCuSpectraPositiveWeights.free();
		mCuSpectraNegativeWeights.free();
		mCuResultWeights.free();
		mCuResult.free();
		mCuResultResiduals.free();
		mCuGradientMemory.free();
	}
};
