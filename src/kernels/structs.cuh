#pragma once

#include <cstdint>

/**
 * Common CUDA kernel execution parameters. Each kernel runner may interpret them slightly differently,
 * but blockSize and sharedMemorySize are usually taken literally.
 */
struct CudaExecParameters
{
public:
	unsigned int blockSize;
	unsigned int sharedMemorySize;
	std::uint32_t privatizedCopies; // affects data replication/privatization (if the kernel employs it)
	std::uint32_t itemsPerThread;	// affects workload division among the threads
	std::uint32_t regsCache;		// affects size of the data cached in registers

	CudaExecParameters(unsigned int _blockSize = 256, unsigned int _sharedMemorySize = 0, std::uint32_t _privatizedCopies = 1,
					   std::uint32_t _itemsPerThread = 1, std::uint32_t _regsCache = 1)
		: blockSize(_blockSize),
		  sharedMemorySize(_sharedMemorySize),
		  privatizedCopies(_privatizedCopies),
		  itemsPerThread(_itemsPerThread),
		  regsCache(_regsCache)
	{}
};


/**
 * Structure that aggregates all data (parameters, input buffers, output buffers) required for kernel execution.
 */
template <typename F>
struct GradientDescendProblemInstance
{
	const F* points;
	const F* spectra;
	const F* spectraPositiveWeights;
	const F* spectraNegativeWeights;
	const F* resultWeights;
	F* result;
	F* resultResiduals;
	const std::uint32_t dim;
	const std::uint32_t n;
	const std::uint32_t spectrumN;
	const std::uint32_t iterations;
	const F alpha;
	const F acceleration;

	// optional
	F* gradientMemory;

	GradientDescendProblemInstance(const F* points, const F* spectra, const F* spectraPositiveWeights, const F* spectraNegativeWeights,
								   const F* resultWeights, F* result, F* resultResiduals, const std::size_t dim, const std::size_t n,
								   const std::size_t spectrumN, const std::size_t iterations, const F alpha, const F acceleration,
								   F* gradientMemory = nullptr)
		: points(points),
		  spectra(spectra),
		  spectraPositiveWeights(spectraPositiveWeights),
		  spectraNegativeWeights(spectraNegativeWeights),
		  resultWeights(resultWeights),
		  result(result),
		  resultResiduals(resultResiduals),
		  dim((std::uint32_t)dim),
		  n((std::uint32_t)n),
		  spectrumN((std::uint32_t)spectrumN),
		  iterations((std::uint32_t)iterations),
		  alpha(alpha),
		  acceleration(acceleration),
		  gradientMemory(gradientMemory)
	{}
};
