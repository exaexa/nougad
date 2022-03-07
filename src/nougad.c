
#include <R.h>
#include <R_ext/Rdynload.h>
#include <Rmath.h>

void
nougad_c(const int *np,
         const int *dp,
         const int *kp,
         const int *itersp,
         const float *alphap,
         const float *accelp,
         const float *s_dk,
         const float *rpw_d,
         const float *rnw_d,
         const float *nw_k,
         const float *y_dn,
         float *x_kn,
         float *r_dn)
{
  const size_t n = *np, d = *dp, k = *kp, iters = *itersp;
  const float alpha = *alphap, accel = *accelp;
  size_t ni, di, ki, ii;

  float *lastg_k = malloc(sizeof(float) * k);

  for (ni = 0; ni < n; ++ni) {
    float *restrict x_k = x_kn + ni * k;
    float *restrict r_d = r_dn + ni * d;
    const float *restrict y_d = y_dn + ni * d;

    for (ki = 0; ki < k; ++ki)
      lastg_k[ki] = 0;

    for (ii = 0;; ++ii) {
      for (di = 0; di < d; ++di)
        r_d[di] = -y_d[di];
      for (ki = 0; ki < k; ++ki)
        for (di = 0; di < d; ++di)
          r_d[di] += x_k[ki] * s_dk[di + d * ki];
      for (di = 0; di < d; ++di)
        r_d[di] *= r_d[di] > 0 ? rpw_d[di] : rnw_d[di];

      if (ii >= iters)
        break;

      for (ki = 0; ki < k; ++ki) {
        float gki = (x_k[ki] > 0 ? 0 : nw_k[ki] * x_k[ki]);
        for (di = 0; di < d; ++di)
          gki +=
            r_d[di] * s_dk[di + d * ki];

        gki *= alpha;
        if (gki * lastg_k[ki] > 0)
          gki += accel * lastg_k[ki];
        x_k[ki] -= gki;
        lastg_k[ki] = gki;
      }
    }
  }

  free(lastg_k);
}

static const R_CMethodDef cMethods[] = { { "nougad_c", (DL_FUNC)&nougad_c, 13 },
                                         { NULL, NULL, 0 } };

void
R_init_nougad(DllInfo *info)
{
  R_registerRoutines(info, cMethods, NULL, NULL, NULL);
  R_useDynamicSymbols(info, FALSE);
}
