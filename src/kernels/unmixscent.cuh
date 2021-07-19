#pragma once

#include "structs.cuh"


#define DECLARE_UNMIXSCENT_KERNEL(NAME)                                                                                                                      \
	template <typename F>                                                                                                                            \
	class NAME                                                                                                                                       \
	{                                                                                                                                                \
	public:                                                                                                                                          \
		static void run(const GradientDescendProblemInstance<F>& in, CudaExecParameters& exec);                                                      \
	};

DECLARE_UNMIXSCENT_KERNEL(UnmixscentBaseKernel)
DECLARE_UNMIXSCENT_KERNEL(UnmixscentBaseSharedKernel)
