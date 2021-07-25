#pragma once

#include <algorithm>

#include "cuda.hpp"
#include "interface.hpp"
#include "nougad.cuh"

/**
 * Serial implementation of GradientDescent algorithm.
 */
template<typename F, class KERNEL>
class GradientDescentCudaAlgorithm : public IGradientDescentAlgorithm<F>
{
protected:
  static constexpr std::size_t problemBatchSize = 1024 * 20;

  bpp::CudaBuffer<F> mCuSpectra, mCuSpectraPositiveWeights,
    mCuSpectraNegativeWeights, mCuResultWeights;
  bpp::HostBuffer<F> mHostPoints, mHostInitialResults;

  struct StreamContextT
  {
    bpp::CudaBuffer<F> mCuPoints, mCuResult, mCuResultResiduals;
    bpp::CudaStream mStream;

    StreamContextT()
      : mStream(0)
    {}

    void initialize(std::size_t dim, std::size_t spectraN)
    {
      mCuPoints.realloc(problemBatchSize * dim);

      mCuResult.realloc(problemBatchSize * spectraN);
      mCuResultResiduals.realloc(problemBatchSize * dim);
    }
    ~StreamContextT()
    {
      mCuPoints.free();
      mCuResult.free();
      mCuResultResiduals.free();
    }
  };

  StreamContextT streamContext[2];

private:
  CudaExecParameters &mCudaExec;
  const DataPoints<F> *mPoints, *mSpectra, *mSpectraPositiveWeights,
    *mSpectraNegativeWeights, *mResultWeights;

public:
  GradientDescentCudaAlgorithm(CudaExecParameters &cudaExec)
    : mCudaExec(cudaExec)
    , mPoints(nullptr)
    , mSpectra(nullptr)
    , mSpectraPositiveWeights(nullptr)
    , mSpectraNegativeWeights(nullptr)
    , mResultWeights(nullptr)
  {}

  virtual void initialize(const DataPoints<F> &points,
                          const DataPoints<F> &spectra,
                          const DataPoints<F> &spectraPositiveWeights,
                          const DataPoints<F> &spectraNegativeWeights,
                          const DataPoints<F> &resultWeights,
                          DataPoints<F> &initialResult,
                          DataPoints<F> &resultResiduals,
                          const std::size_t iterations,
                          const F alpha,
                          const F acceleration)
  {
    std::size_t devices = bpp::CudaDevice::count();
    if (devices == 0) {
      throw bpp::RuntimeError("No CUDA devices found!");
    }

    IGradientDescentAlgorithm<F>::initialize(points,
                                             spectra,
                                             spectraPositiveWeights,
                                             spectraNegativeWeights,
                                             resultWeights,
                                             initialResult,
                                             resultResiduals,
                                             iterations,
                                             alpha,
                                             acceleration);
    mPoints = &points;
    mSpectra = &spectra;
    mSpectraPositiveWeights = &spectraPositiveWeights;
    mSpectraNegativeWeights = &spectraNegativeWeights;
    mResultWeights = &resultWeights;
    this->mResult = std::move(initialResult);
    this->mResultResiduals = std::move(resultResiduals);
  }

  void prepareInputs() override
  {
    CUCH(cudaSetDevice(0));

    mHostPoints.realloc(this->mN * this->mDim, false, false, true);
    mHostInitialResults.realloc(
      this->mN * this->mSpectrumN, false, false, true);
    mCuSpectra.realloc(this->mSpectrumN * this->mDim);
    mCuSpectraPositiveWeights.realloc(this->mSpectrumN * this->mDim);
    mCuSpectraNegativeWeights.realloc(this->mSpectrumN * this->mDim);
    mCuResultWeights.realloc(this->mSpectrumN);

    memcpy(&*mHostPoints, mPoints->data(), this->mN * this->mDim * sizeof(F));
    memcpy(&*mHostInitialResults,
           this->mResult.data(),
           this->mN * this->mSpectrumN * sizeof(F));
    mCuSpectra.write(mSpectra->data(), this->mSpectrumN * this->mDim);
    mCuSpectraPositiveWeights.write(mSpectraPositiveWeights->data(),
                                    this->mSpectrumN * this->mDim);
    mCuSpectraNegativeWeights.write(mSpectraNegativeWeights->data(),
                                    this->mSpectrumN * this->mDim);
    mCuResultWeights.write(mResultWeights->data(), this->mSpectrumN);

    streamContext[0].initialize(this->mDim, this->mSpectrumN);
    streamContext[1].initialize(this->mDim, this->mSpectrumN);
  }

  void run() override
  {
    for (std::size_t batchOffset = 0; batchOffset < this->mN;
         batchOffset += problemBatchSize) {
      const auto batchSize = std::min(problemBatchSize, this->mN - batchOffset);

      auto &currentContext =
        streamContext[(batchOffset / problemBatchSize) % 2];

      currentContext.mCuPoints.writeAsync(*currentContext.mStream,
                                          &*mHostPoints +
                                            batchOffset * this->mDim,
                                          batchSize * this->mDim);
      currentContext.mCuResult.writeAsync(*currentContext.mStream,
                                          &*mHostInitialResults +
                                            batchOffset * this->mSpectrumN,
                                          batchSize * this->mSpectrumN);

      GradientDescentProblemInstance<F> gdProblem(
        *currentContext.mCuPoints,
        *mCuSpectra,
        *mCuSpectraPositiveWeights,
        *mCuSpectraNegativeWeights,
        *mCuResultWeights,
        *currentContext.mCuResult,
        *currentContext.mCuResultResiduals,
        this->mDim,
        batchSize,
        this->mSpectrumN,
        this->mIterations,
        this->mAlpha,
        this->mAcceleration);

      mCudaExec.stream = *currentContext.mStream;
      KERNEL::run(gdProblem, mCudaExec);

      currentContext.mCuResult.readAsync(*currentContext.mStream,
                                         this->mResult.data() +
                                           batchOffset * this->mSpectrumN,
                                         batchSize * this->mSpectrumN);
      currentContext.mCuResultResiduals.readAsync(
        *currentContext.mStream,
        this->mResultResiduals.data() + batchOffset * this->mDim,
        batchSize * this->mDim);
    }

    CUCH(cudaDeviceSynchronize());
  }

  const typename IGradientDescentAlgorithm<F>::ResultT getResults() override
  {
    return IGradientDescentAlgorithm<F>::getResults();
  }

  void cleanup() override
  {
    mCuSpectra.free();
    mCuSpectraPositiveWeights.free();
    mCuSpectraNegativeWeights.free();
    mCuResultWeights.free();
  }
};
