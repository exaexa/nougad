#pragma once

#include "structs.cuh"

#define DECLARE_NOUGAD_KERNEL(NAME)                                            \
  template<typename F>                                                         \
  class NAME                                                                   \
  {                                                                            \
  public:                                                                      \
    static void run(const GradientDescentProblemInstance<F> &in,               \
                    CudaExecParameters &exec);                                 \
  };

DECLARE_NOUGAD_KERNEL(NougadGroupSharedKernel)
