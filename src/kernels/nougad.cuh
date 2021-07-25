#pragma once

#include "structs.cuh"

#define DECLARE_NOUGAD_KERNEL(NAME)                                            \
  template<typename F>                                                         \
  class NAME                                                                   \
  {                                                                            \
  public:                                                                      \
    static void run(const GradientDescendProblemInstance<F> &in,               \
                    CudaExecParameters &exec);                                 \
  };

DECLARE_NOUGAD_KERNEL(NougadBaseKernel)
DECLARE_NOUGAD_KERNEL(NougadBaseSharedKernel)
