#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include "exception.hpp"
#include "points.hpp"
#include "structs.cuh"

template<typename F>
using DataPointsRef = std::reference_wrapper<DataPoints<F>>;

/**
 * Interface (base class) for all ESOM algorithms.
 */
template<typename F = float>
class IGradientDescendAlgorithm
{
protected:
  std::size_t mDim, mN, mSpectrumN, mIterations;
  F mAlpha, mAcceleration;
  DataPoints<F> mResultResiduals, mResult;

public:
  using ResultT = std::pair<DataPointsRef<F>, DataPointsRef<F>>;

  IGradientDescendAlgorithm()
    : mDim(0)
    , mN(0)
    , mSpectrumN(0)
    , mIterations(0)
    , mAlpha(0)
    , mAcceleration(0)
    , mResultResiduals(0)
    , mResult(0)
  {}

  virtual ~IGradientDescendAlgorithm() {}

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
    mDim = points.getDim();
    mN = points.size();
    mSpectrumN = spectra.size();

    if (initialResult.getDim() != mSpectrumN || initialResult.size() != mN) {
      throw(bpp::RuntimeError()
            << "Result points dimension and size are different than such "
               "implied by measurement points and spectra. Result: "
            << initialResult.size() << "x" << initialResult.getDim()
            << "Measurements: " << points.size() << "x" << points.getDim()
            << " Spectra: " << spectra.size() << "x" << spectra.getDim());
    }

    if (spectra.getDim() != mDim) {
      throw(bpp::RuntimeError()
            << "Measurement points and spectra have different dimensions: "
            << mDim << " != " << spectra.getDim());
    }

    if (spectra.getDim() != spectraNegativeWeights.getDim() ||
        spectra.getDim() != spectraPositiveWeights.getDim() ||
        spectra.size() != spectraNegativeWeights.size() ||
        spectra.size() != spectraPositiveWeights.size()) {
      throw(bpp::RuntimeError()
            << "Spectra weights dimension and size are different than spectra. "
               "Spectra: "
            << spectra.size() << "x" << spectra.getDim()
            << "Positive Weights: " << spectraPositiveWeights.size() << "x"
            << spectraPositiveWeights.getDim()
            << " Negative Weights: " << spectraNegativeWeights.size() << "x"
            << spectraNegativeWeights.getDim());
    }

    if (resultWeights.getDim() != mSpectrumN || resultWeights.size() != 1) {
      throw(bpp::RuntimeError()
            << "Results points and result weights have different dimensions: "
            << mSpectrumN
            << " != " << resultWeights.size() * resultWeights.getDim());
    }

    if (resultResiduals.getDim() != mDim || resultResiduals.size() != mN) {
      throw(bpp::RuntimeError()
            << "Residuals dimension and size are different than such implied "
               "by measurement points. Residuals: "
            << resultResiduals.size() << "x" << resultResiduals.getDim()
            << "Measurements: " << points.size() << "x" << points.getDim());
    }

    mIterations = iterations;
    mAlpha = alpha;
    mAcceleration = acceleration;

    // Derived class should save the pointers to inputs or copy them...
    // This part of the algorithm is not measured
  }

  virtual void prepareInputs()
  {
    // Transpose input data, copy them to GPU, ...
    // This part is measured separately from the run.
  }

  virtual void run() = 0;

  virtual const ResultT getResults()
  {
    return std::make_pair(std::ref(mResult), std::ref(mResultResiduals));
  }

  static const DataPoints<F> &getResult(size_t i, const ResultT &result)
  {
    return i == 0 ? result.first.get() : result.second.get();
  }

  bool verifyResult(IGradientDescendAlgorithm<F> &refAlgorithm,
                    std::ostream &out)
  {
    refAlgorithm.prepareInputs();
    refAlgorithm.run();
    auto refResults = refAlgorithm.getResults();
    auto results = getResults();

    std::size_t errors = 0;

    for (size_t resultI = 0; resultI < 2; resultI++) {
      const auto &refResult = getResult(resultI, refResults);
      const auto &result = getResult(resultI, results);
      size_t N = resultI == 0 ? mSpectrumN : mDim;

      for (std::size_t p = 0; p < mN; ++p) {
        std::size_t i = 0;
        while (i < N &&
               compare_floats_relative(refResult[p][i], result[p][i]) < 0.001)
          ++i;

        if (i < N) {
          if (++errors < 16) { // only first 16 errors are printed out
            out << (resultI == 0 ? "Result" : "Residual") << " Error [" << p
                << "]:" << std::endl;
            for (std::size_t j = i; j < N; ++j)
              out << "\t[" << j << "]";
            out << std::endl;

            for (std::size_t j = i; j < N; ++j)
              out << "\t" << result[p][j] << " ";
            out << std::endl;

            for (std::size_t j = i; j < N; ++j)
              out << "\t" << refResult[p][j] << " ";
            out << std::endl;
          }
        }
      }
    }

    if (errors) {
      out << "Total errors: " << errors << std::endl;
    } else {
      out << "Verification OK." << std::endl;
    }

    refAlgorithm.cleanup();
    return errors == 0;
  }

  virtual void cleanup() = 0;
};
