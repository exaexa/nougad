
#include <cstddef>
#include <vector>

static void
nougad_impl(const size_t n,
            const size_t d,
            const size_t k,
            const size_t iters,
            const float alpha,
            const float accel,
            const float *s_dk,
            const float *rnw_kd,
            const float *rpw_kd,
            const float *nw_k,
            const float *y_dn,
            float *x_kn,
            float *r_dn)
{
  size_t ni, di, ki, ii;

  std::vector<float> lastg_k(k);
  std::vector<float> g_k(k);

  for (ni = 0; ni < n; ++ni) {
    float *x_k = x_kn + ni * k;
    float *r_d = r_dn + ni * d;
    const float *y_d = y_dn + ni * d;

    for (ki = 0; ki < k; ++ki)
      lastg_k[ki] = 0;

    for (ii = 0;; ++ii) {
      for (di = 0; di < d; ++di)
        r_d[di] = -y_d[di]; // residual is "negated" here for simplicity
      for (ki = 0; ki < k; ++ki)
        for (di = 0; di < d; ++di)
          r_d[di] += x_k[ki] * s_dk[di + d * ki];

      if (ii >= iters)
        break;

      for (ki = 0; ki < k; ++ki)
        g_k[ki] = x_k[ki] > 0 ? 0 : nw_k[ki] * x_k[ki];

      for (di = 0; di < d; ++di)
        if (r_d[di] > 0) // condition flipped by the negated residual
          for (ki = 0; ki < k; ++ki)
            g_k[ki] += r_d[di] * rnw_kd[ki + k * di];
        else
          for (ki = 0; ki < k; ++ki)
            g_k[ki] += r_d[di] * rpw_kd[ki + k * di];

      for (ki = 0; ki < k; ++ki) {
        float gki = g_k[ki] * alpha;
        if (gki * lastg_k[ki] > 0)
          gki += accel * lastg_k[ki];
        x_k[ki] -= gki;
        lastg_k[ki] = gki;
      }
    }
  }
}

#include <thread>

extern "C" void
nougad_c(const int *threadsp,
         const int *np,
         const int *dp,
         const int *kp,
         const int *itersp,
         const float *alphap,
         const float *accelp,
         const float *s_dk,
         const float *rpw_kd,
         const float *rnw_kd,
         const float *nw_k,
         const float *y_dn,
         float *x_kn,
         float *r_dn)
{
  int threads = *threadsp;
  if (threads <= 0)
    threads = std::thread::hardware_concurrency();
  if (threads == 1)
    nougad_impl(*np,
                *dp,
                *kp,
                *itersp,
                *alphap,
                *accelp,
                s_dk,
                rpw_kd,
                rnw_kd,
                nw_k,
                y_dn,
                x_kn,
                r_dn);
  else {
    size_t n = *np, d = *dp, k = *kp;
    std::vector<std::thread> ts(threads);
    for (size_t i = 0; i < threads; ++i)
      ts[i] = std::thread(
        [&](size_t thread_id) {
          size_t beginT = thread_id * n / threads,
                 endT = (thread_id + 1) * n / threads;
          size_t nT = endT - beginT;
          const float *y_dnT = y_dn + d * beginT;
          float *x_knT = x_kn + k * beginT;
          float *r_dnT = r_dn + d * beginT;
          nougad_impl(nT,
                      d,
                      k,
                      *itersp,
                      *alphap,
                      *accelp,
                      s_dk,
                      rpw_kd,
                      rnw_kd,
                      nw_k,
                      y_dnT,
                      x_knT,
                      r_dnT);
        },
        i);
    for (auto &t : ts)
      t.join();
  }
}

#include <R.h>
#include <R_ext/Rdynload.h>
#include <Rmath.h>

static const R_CMethodDef cMethods[] = { { "nougad_c", (DL_FUNC)&nougad_c, 14 },
                                         { NULL, NULL, 0 } };

void
R_init_nougad(DllInfo *info)
{
  R_registerRoutines(info, cMethods, NULL, NULL, NULL);
  R_useDynamicSymbols(info, FALSE);
}
