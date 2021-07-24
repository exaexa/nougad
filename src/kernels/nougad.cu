#ifdef __INTELLISENSE__
#define __CUDACC__
#endif

#include <climits>
#include <cstdint>

#include "cooperative_groups.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nougad.cuh"

namespace cg = cooperative_groups;

template<typename F>
__inline__ __device__ void
computeResiduals(const F *const __restrict__ points,
                 const F *const __restrict__ spectra,
                 const F *const __restrict__ result,
                 F *const __restrict__ residuals,
                 const std::uint32_t dim,
                 const std::uint32_t spectrumN)
{
  for (std::uint32_t d_idx = 0; d_idx < dim; ++d_idx)
    residuals[d_idx] = -points[d_idx];

  for (std::uint32_t k_idx = 0; k_idx < spectrumN; ++k_idx)
    for (size_t d_idx = 0; d_idx < dim; ++d_idx)
      residuals[d_idx] += result[k_idx] * spectra[k_idx * dim + d_idx];
}

/**
 * Each thread performs descent for one point.
 */
template<typename F>
__global__ void
nougadBaseKernel(const F *__restrict__ points,
                 const F *const __restrict__ spectra,
                 const F *const __restrict__ spectraPositiveWeights,
                 const F *const __restrict__ spectraNegativeWeights,
                 const F *const __restrict__ resultWeights,
                 F *__restrict__ result,
                 F *__restrict__ resultResiduals,
                 F *__restrict__ gradientMemory,
                 const std::uint32_t dim,
                 const std::uint32_t n,
                 const std::uint32_t spectrumN,
                 const std::uint32_t iterations,
                 const F alpha,
                 const F acceleration)
{
  // set variables for each thread specifically
  {
    const auto threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    points = points + threadIndex * dim;
    result = result + threadIndex * spectrumN;
    resultResiduals = resultResiduals + threadIndex * dim;
    gradientMemory = gradientMemory + threadIndex * spectrumN;
  }

  for (std::uint32_t i = 0; i < iterations; ++i) {
    computeResiduals(points, spectra, result, resultResiduals, dim, spectrumN);

    for (std::uint32_t k_idx = 0; k_idx < spectrumN; ++k_idx) {
      F gradient = result[k_idx] > 0 ? 0 : resultWeights[k_idx] * result[k_idx];

      for (std::uint32_t d_idx = 0; d_idx < dim; ++d_idx) {
        const F w = (resultResiduals[d_idx] > 0
                       ? spectraPositiveWeights
                       : spectraNegativeWeights)[k_idx * dim + d_idx];
        gradient += resultResiduals[d_idx] * spectra[k_idx * dim + d_idx] * w;
      }

      // apply gradient
      {
        gradient *= alpha;
        if (gradient * gradientMemory[k_idx] > 0)
          gradient += acceleration * gradientMemory[k_idx];
        result[k_idx] -= gradient;
        gradientMemory[k_idx] = gradient;
      }
    }
  }
  computeResiduals(points, spectra, result, resultResiduals, dim, spectrumN);
}

// runner wrapped in a class
template<typename F>
void
NougadBaseKernel<F>::run(const GradientDescendProblemInstance<F> &in,
                         CudaExecParameters &exec)
{
  unsigned int blockCount = (in.n + exec.blockSize - 1) / exec.blockSize;
  nougadBaseKernel<F><<<blockCount, exec.blockSize>>>(in.points,
                                                      in.spectra,
                                                      in.spectraPositiveWeights,
                                                      in.spectraNegativeWeights,
                                                      in.resultWeights,
                                                      in.result,
                                                      in.resultResiduals,
                                                      in.gradientMemory,
                                                      in.dim,
                                                      in.n,
                                                      in.spectrumN,
                                                      in.iterations,
                                                      in.alpha,
                                                      in.acceleration);
}

/**
 * Each thread performs descent for one point.
 * Common arrays are stored to shared memory.
 */
template<typename F>
__global__ void
nougadBaseSharedKernel(const F *__restrict__ points,
                       const F *const __restrict__ spectra,
                       const F *const __restrict__ spectraPositiveWeights,
                       const F *const __restrict__ spectraNegativeWeights,
                       const F *const __restrict__ resultWeights,
                       F *__restrict__ result,
                       F *__restrict__ resultResiduals,
                       F *__restrict__ gradientMemory,
                       const std::uint32_t dim,
                       const std::uint32_t n,
                       const std::uint32_t spectrumN,
                       const std::uint32_t iterations,
                       const F alpha,
                       const F acceleration)
{
  extern __shared__ char sharedMemory[];
  F *const __restrict__ spectraCache = reinterpret_cast<F *>(sharedMemory);
  F *const __restrict__ spectraPositiveWeightsCache =
    spectraCache + dim * spectrumN;
  F *const __restrict__ spectraNegativeWeightsCache =
    spectraPositiveWeightsCache + dim * spectrumN;
  F *const __restrict__ resultWeightsCache =
    spectraNegativeWeightsCache + dim * spectrumN;
  F *const __restrict__ gradientMemoryCache =
    resultWeightsCache + spectrumN + threadIdx.x * spectrumN;
  F *const __restrict__ resultResidualsCache =
    resultWeightsCache + spectrumN + blockDim.x * spectrumN + threadIdx.x * dim;
  F *const __restrict__ resultCache =
    resultWeightsCache + spectrumN + blockDim.x * spectrumN + blockDim.x * dim +
    threadIdx.x * spectrumN;

  // set variables for each thread specifically
  {
    const auto threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    points = points + threadIndex * dim;
    result = result + threadIndex * spectrumN;
    resultResiduals = resultResiduals + threadIndex * dim;
  }

  // store to shared memory
  {
    memset(gradientMemoryCache, 0, spectrumN * sizeof(F));
    memcpy(resultCache, result, spectrumN * sizeof(F));
    if (threadIdx.x == 0) {
      memcpy(spectraCache, spectra, dim * spectrumN * sizeof(F));
      memcpy(spectraPositiveWeightsCache,
             spectraPositiveWeights,
             dim * spectrumN * sizeof(F));
      memcpy(spectraNegativeWeightsCache,
             spectraNegativeWeights,
             dim * spectrumN * sizeof(F));
      memcpy(resultWeightsCache, resultWeights, spectrumN * sizeof(F));
    }
    __syncthreads();
  }

  for (std::uint32_t i = 0; i < iterations; ++i) {
    computeResiduals(
      points, spectraCache, resultCache, resultResidualsCache, dim, spectrumN);

    for (std::uint32_t k_idx = 0; k_idx < spectrumN; ++k_idx) {
      F gradient = resultCache[k_idx] > 0
                     ? 0
                     : resultWeightsCache[k_idx] * resultCache[k_idx];

      for (std::uint32_t d_idx = 0; d_idx < dim; ++d_idx) {
        const F w = (resultResidualsCache[d_idx] > 0
                       ? spectraPositiveWeightsCache
                       : spectraNegativeWeightsCache)[k_idx * dim + d_idx];
        gradient +=
          resultResidualsCache[d_idx] * spectraCache[k_idx * dim + d_idx] * w;
      }

      // apply gradient
      {
        gradient *= alpha;
        if (gradient * gradientMemoryCache[k_idx] > 0)
          gradient += acceleration * gradientMemoryCache[k_idx];
        resultCache[k_idx] -= gradient;
        gradientMemoryCache[k_idx] = gradient;
      }
    }
  }
  computeResiduals(
    points, spectraCache, resultCache, resultResiduals, dim, spectrumN);
  memcpy(result, resultCache, spectrumN * sizeof(F));
}

// runner wrapped in a class
template<typename F>
void
NougadBaseSharedKernel<F>::run(const GradientDescendProblemInstance<F> &in,
                               CudaExecParameters &exec)
{
  unsigned int blockCount = (in.n + exec.blockSize - 1) / exec.blockSize;
  unsigned int sharedMemory =
    (3 * (in.dim * in.spectrumN) + in.spectrumN) * sizeof(F) +
    exec.blockSize * sizeof(F) * (in.spectrumN + in.dim + in.spectrumN);

  nougadBaseSharedKernel<F>
    <<<blockCount, exec.blockSize, sharedMemory>>>(in.points,
                                                   in.spectra,
                                                   in.spectraPositiveWeights,
                                                   in.spectraNegativeWeights,
                                                   in.resultWeights,
                                                   in.result,
                                                   in.resultResiduals,
                                                   in.gradientMemory,
                                                   in.dim,
                                                   in.n,
                                                   in.spectrumN,
                                                   in.iterations,
                                                   in.alpha,
                                                   in.acceleration);
}

/*
 * Explicit template instantiation.
 */
template<typename F>
void
instantiateKernelRunnerTemplates()
{
  GradientDescendProblemInstance<F> instance(nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0,
                                             0);
  CudaExecParameters exec;

  NougadBaseKernel<F>::run(instance, exec);
  NougadBaseSharedKernel<F>::run(instance, exec);
}

template void
instantiateKernelRunnerTemplates<float>();
#ifndef NO_DOUBLES
template void
instantiateKernelRunnerTemplates<double>();
#endif
