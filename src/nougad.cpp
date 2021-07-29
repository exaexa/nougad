
#include <R.h>
#include <R_ext/Rdynload.h>
#include <Rmath.h>
#include <cstdint>
#include <iterator>
#include <vector>
#include <vuh/array.hpp>
#include <vuh/vuh.h>

#include "unmix.comph"

const size_t local_size = 64;
const size_t batch_size = 5120;

/*
 * transposed matrix view -- iterating on transposed(ptr,rows,cols) that is
 * stored by columns outputs (or ingests) the numbers as if it was read by
 * rows.
 */

template<typename T>
struct transposed
{

  T *ptr;
  const size_t a, b;

  transposed(T *ptr, size_t a, size_t b)
    : ptr(ptr)
    , a(a)
    , b(b)
  {}

  size_t size() const { return a * b; }

  struct tr_iter
  {
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using reference = T &;

    size_t off;
    transposed base;
    tr_iter(size_t off, transposed base)
      : off(off)
      , base(base)
    {}

    inline reference operator*() const
    {
      const size_t blk = off / base.b;
      const size_t item = off % base.b;
      return base.ptr[blk + base.a * item];
    }

    tr_iter &operator++()
    {
      ++off;
      return *this;
    }

    ptrdiff_t operator-(tr_iter other) const { return off - other.off; }
    bool operator==(tr_iter other) const { return off == other.off; }
    bool operator!=(tr_iter other) const { return off != other.off; }
    bool operator<(tr_iter other) const { return off < other.off; }
  };

  tr_iter begin() const { return tr_iter(0, *this); }
  tr_iter end() const { return tr_iter(a * b, *this); }
  tr_iter at(size_t ia, size_t ib) const { return tr_iter(ib + b * ia, *this); }
};

/*
 * nougad entrypoint
 */

extern "C" void
nougad_c(const int *np,
         const int *dp,
         const int *kp,
         const int *itersp,
         const float *alphap,
         const float *accelp,
         const float *s_dk,
         const float *spw_dk,
         const float *snw_dk,
         const float *nw_k,
         const float *y_dn,
         float *x_kn,
         float *r_dn)
{
  const size_t n = *np, d = *dp, k = *kp, iters = *itersp;
  const float alpha = *alphap, accel = *accelp;

  auto instance = vuh::Instance();
  auto device = instance.devices().at(0); // TODO parametrize

  vuh::Array<float> vu_s_kd(device, k * d), vu_spw_kd(device, k * d),
    vu_snw_kd(device, k * d), vu_nw_k(device, k);

  // we're doing partial fills on these (vuh crashes if they're not host
  // visible)
  vuh::Array<float, vuh::mem::Host> vu_y_nd(device, batch_size * d),
    vu_x_nk(device, batch_size * k), vu_r_nd(device, batch_size * d);

  vu_s_kd.fromHost(transposed(s_dk, d, k).begin(),
                   transposed(s_dk, d, k).end());
  vu_spw_kd.fromHost(transposed(spw_dk, d, k).begin(),
                     transposed(spw_dk, d, k).end());
  vu_snw_kd.fromHost(transposed(snw_dk, d, k).begin(),
                     transposed(snw_dk, d, k).end());
  vu_nw_k.fromHost(nw_k, nw_k + k);

  using Specs = vuh::typelist<uint32_t /*n_blocks*/,
                              uint32_t /* k */,
                              uint32_t /* d */,
                              uint32_t /* iterations */>;
  struct Params
  {
    uint32_t n_cells;
    float alpha;
    float accel;
  };

  auto program =
    vuh::Program<Specs, Params>(device, sizeof(spirv_unmix), spirv_unmix);

  for (size_t batch_off = 0; batch_off < n; batch_off += batch_size) {
    size_t local_n = std::min(batch_size, n - batch_off);

    auto xv = transposed(x_kn + k * batch_off, k, local_n);
    auto yv = transposed(y_dn + d * batch_off, d, local_n);
    auto rv = transposed(r_dn + d * batch_off, d, local_n);

    std::copy(xv.begin(), xv.end(), vu_x_nk.begin());
    std::copy(yv.begin(), yv.end(), vu_y_nd.begin());

    program.grid(vuh::div_up(local_n, local_size))
      .spec(local_size, k, d, iters)({ uint32_t(local_n), alpha, accel },
                                     vu_s_kd,
                                     vu_spw_kd,
                                     vu_snw_kd,
                                     vu_nw_k,
                                     vu_y_nd,
                                     vu_x_nk,
                                     vu_r_nd);

    std::copy(vu_x_nk.begin(), vu_x_nk.begin() + xv.size(), xv.begin());
    std::copy(vu_r_nd.begin(), vu_r_nd.begin() + rv.size(), rv.begin());
  }
}

/*
 * R API connector
 */

extern "C"
{
  static const R_CMethodDef cMethods[] = {
    { "nougad_c", (DL_FUNC)&nougad_c, 13 },
    { NULL, NULL, 0 }
  };

  void R_init_nougad(DllInfo *info)
  {
    R_registerRoutines(info, cMethods, NULL, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
  }
} // extern "C"
