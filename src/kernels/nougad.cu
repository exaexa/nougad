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

template<typename F, typename T>
__inline__ __device__ void
computeResiduals(T &group,
                 const F *const __restrict__ points,
                 const F *const __restrict__ spectra,
                 const F *const __restrict__ result,
                 F *const __restrict__ residuals,
                 const std::uint32_t dim,
                 const std::uint32_t spectrumN)
{
  for (std::uint32_t d_idx = group.thread_rank(); d_idx < dim;
       d_idx += group.size())
    residuals[d_idx] = -points[d_idx];

  group.sync();

  for (std::uint32_t idx = group.thread_rank(); idx < spectrumN * dim;
       idx += group.size()) {
    const auto k_idx = idx / dim;
    const auto d_idx = idx % dim;
    residuals[d_idx] += result[k_idx] * spectra[idx];
  }
}

/**
 * A group of threads performs descent for one point.
 * Common arrays are stored to shared memory.
 */
template<typename F, std::size_t group_size>
__global__ void
nougadGroupSharedKernel(const F *__restrict__ points,
                        const F *const __restrict__ spectra,
                        const F *const __restrict__ spectraPositiveWeights,
                        const F *const __restrict__ spectraNegativeWeights,
                        const F *const __restrict__ resultWeights,
                        F *__restrict__ result,
                        F *__restrict__ resultResiduals,
                        const std::uint32_t dim,
                        const std::uint32_t n,
                        const std::uint32_t spectrumN,
                        const std::uint32_t iterations,
                        const F alpha,
                        const F acceleration)
{
  extern __shared__ char sharedMemory[];

  auto group = cg::tiled_partition<group_size>(cg::this_thread_block());

  F *const __restrict__ spectraCache = reinterpret_cast<F *>(sharedMemory);
  F *const __restrict__ spectraPositiveWeightsCache =
    spectraCache + dim * spectrumN;
  F *const __restrict__ spectraNegativeWeightsCache =
    spectraPositiveWeightsCache + dim * spectrumN;
  F *const __restrict__ resultWeightsCache =
    spectraNegativeWeightsCache + dim * spectrumN;

  F *const __restrict__ gradientMemoryCache =
    resultWeightsCache + spectrumN + group.meta_group_rank() * spectrumN;
  F *const __restrict__ resultResidualsCache =
    resultWeightsCache + spectrumN + group.meta_group_size() * spectrumN +
    group.meta_group_rank() * dim;
  F *const __restrict__ resultCache =
    resultWeightsCache + spectrumN + group.meta_group_size() * spectrumN +
    group.meta_group_size() * dim + group.meta_group_rank() * spectrumN;
  F *const __restrict__ pointCache =
    resultWeightsCache + spectrumN + group.meta_group_size() * spectrumN +
    group.meta_group_size() * dim + group.meta_group_size() * spectrumN +
    group.meta_group_rank() * dim;

  // set variables for each group specifically
  {
    const auto groupIndex =
      group.meta_group_rank() + blockIdx.x * group.meta_group_size();
    points = points + groupIndex * dim;
    result = result + groupIndex * spectrumN;
    resultResiduals = resultResiduals + groupIndex * dim;
  }

  // store to shared memory
  {
    if (group.thread_rank() == 0) {
      memset(gradientMemoryCache, 0, spectrumN * sizeof(F));
      memcpy(resultCache, result, spectrumN * sizeof(F));
      memcpy(pointCache, points, dim * sizeof(F));
    }
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
    computeResiduals(group,
                     pointCache,
                     spectraCache,
                     resultCache,
                     resultResidualsCache,
                     dim,
                     spectrumN);

    for (std::uint32_t k_idx = group.thread_rank(); k_idx < spectrumN;
         k_idx += group.size()) {
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

    group.sync();
  }

  computeResiduals(group,
                   pointCache,
                   spectraCache,
                   resultCache,
                   resultResiduals,
                   dim,
                   spectrumN);

  if (group.thread_rank() == 0) {
    memcpy(result, resultCache, spectrumN * sizeof(F));
  }
}

// runner wrapped in a class
template<typename F>
void
NougadGroupSharedKernel<F>::run(const GradientDescentProblemInstance<F> &in,
                                CudaExecParameters &exec)
{
  constexpr std::size_t groupSize = 8;
  const unsigned int blockSize = exec.blockSize / groupSize;
  unsigned int blockCount = (in.n + blockSize - 1) / blockSize;
  unsigned int sharedMemory =
    (3 * (in.dim * in.spectrumN) + in.spectrumN) * sizeof(F) +
    blockSize * sizeof(F) * (in.spectrumN + in.dim + in.spectrumN + in.dim);

  nougadGroupSharedKernel<F, groupSize>
    <<<blockCount, exec.blockSize, sharedMemory, exec.stream>>>(
      in.points,
      in.spectra,
      in.spectraPositiveWeights,
      in.spectraNegativeWeights,
      in.resultWeights,
      in.result,
      in.resultResiduals,
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
  GradientDescentProblemInstance<F> instance(nullptr,
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

  NougadGroupSharedKernel<F>::run(instance, exec);
}

template void
instantiateKernelRunnerTemplates<float>();
#ifndef NO_DOUBLES
template void
instantiateKernelRunnerTemplates<double>();
#endif
