
#include <R.h>
#include <R_ext/Rdynload.h>
#include <Rmath.h>

#include <chrono>
#include <iostream> //TODO remove

#include "resource.hpp"
#include "transposed.hpp"
#include "vut.hpp"

#include "unmix.comph"

const size_t local_size = 64;
const size_t batch_size = 5120;

struct unmix_constants
{
  uint32_t n;
  float alpha;
  float accel;
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

  resource instance(vut::createInstance("nougad"), [](auto i) { i.destroy(); });
  auto phy = vut::findPhy(instance, getenv("NOUGAD_VULKAN_DEVICE"));
  auto computeQ = vut::findQueueFamilyIdx(
    phy, vk::QueueFlagBits::eCompute, vk::QueueFlagBits::eGraphics);
  resource dev(vut::createDevice<1>(phy, { { { computeQ, 1, 1.0 } } }),
               [](auto d) { d.destroy(); });

  auto mkBuf = [&](size_t elems) {
    return resource(vut::createStorageBuffer<float>(dev, elems),
                    [&](vk::Buffer b) { dev->destroyBuffer(b); });
  };

  resource bufS = mkBuf(d * k), bufSNW = mkBuf(d * k), bufSPW = mkBuf(d * k),
           bufNW = mkBuf(k), bufX = mkBuf(batch_size * k),
           bufY = mkBuf(batch_size * d), bufR = mkBuf(batch_size * d);

  auto cachedMemIdx =
    vut::findMemoryTypeIdx(phy, vk::MemoryPropertyFlagBits::eHostCached);

  auto mkCachedMem = [&](size_t elems) {
    return resource(vut::allocateMemory<float>(dev, elems, cachedMemIdx),
                    [&](vk::DeviceMemory m) { dev->freeMemory(m); });
  };

  resource memS = mkCachedMem(d * k), memSNW = mkCachedMem(d * k),
           memSPW = mkCachedMem(d * k), memNW = mkCachedMem(k),
           memX = mkCachedMem(batch_size * k),
           memY = mkCachedMem(batch_size * d),
           memR = mkCachedMem(batch_size * d);

  dev->bindBufferMemory(bufS, memS, 0);
  dev->bindBufferMemory(bufSNW, memSNW, 0);
  dev->bindBufferMemory(bufSPW, memSPW, 0);
  dev->bindBufferMemory(bufNW, memNW, 0);
  dev->bindBufferMemory(bufX, memX, 0);
  dev->bindBufferMemory(bufY, memY, 0);
  dev->bindBufferMemory(bufR, memR, 0);

  resource shader(
    vut::createShader(
      dev, 7, sizeof(unmix_constants), sizeof(spirv_unmix), spirv_unmix),
    [&](auto &sh) { sh.destroy(dev); });

  resource pipeline(vut::createComputePipeline(dev,
                                               shader,
                                               "main",
                                               uint32_t(local_size),
                                               uint32_t(k),
                                               uint32_t(d),
                                               uint32_t(iters)),
                    [&](auto p) { dev->destroyPipeline(p); });

  resource pool(vut::createStorageDescriptorPool(dev, 1, 7),
                [&](auto pool) { dev->destroyDescriptorPool(pool); });

  resource dss(vut::allocateDescriptorSets(dev, pool, *shader),
               [&](auto &dss) { dev->freeDescriptorSets(pool, dss); });
  auto &ds = dss->at(0);

  vut::writeDescriptorSet(
    dev, ds, bufS, bufSNW, bufSPW, bufNW, bufX, bufY, bufR);

  auto withMem = [&dev](vk::DeviceMemory &mem,
                        auto fn,
                        bool invalidate = false,
                        bool flush = false) {
    auto start = std::chrono::high_resolution_clock::now();

    resource m(vut::mapMemory<float>(dev, mem),
               [&dev, &mem](float *) { dev->unmapMemory(mem); });

    if (invalidate)
      vut::invalidateMemory(dev, mem);
    fn(*m);
    if (flush)
      vut::flushMemory(dev, mem);

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout
      << "mem transfer (flushes " << invalidate << flush << ") took: "
      << std::chrono::duration<double, std::milli>(finish - start).count()
      << std::endl;
  };

  withMem(
    memS,
    [&](float *data) {
      std::copy(
        transposed(s_dk, d, k).begin(), transposed(s_dk, d, k).end(), data);
    },
    false,
    true);

  withMem(
    memSNW,
    [&](float *data) {
      std::copy(
        transposed(snw_dk, d, k).begin(), transposed(snw_dk, d, k).end(), data);
    },
    false,
    true);

  withMem(
    memSPW,
    [&](float *data) {
      std::copy(
        transposed(spw_dk, d, k).begin(), transposed(spw_dk, d, k).end(), data);
    },
    false,
    true);

  withMem(
    memNW, [&](float *data) { std::copy(nw_k, nw_k + k, data); }, false, true);

  resource cmdpool(vut::createCmdPool(dev, computeQ),
                   [&](auto cp) { dev->destroyCommandPool(cp); });

  resource cbufs(vut::allocateCmdBufs(dev, cmdpool, 1),
                 [&](auto &cbufs) { dev->freeCommandBuffers(cmdpool, cbufs); });
  auto &cbuf = cbufs->at(0);

  auto cq = dev->getQueue(computeQ, 0);

  auto mkMap = [&](auto &mem) {
    return resource(vut::mapMemory<float>(dev, mem),
                    [&](auto) { dev->unmapMemory(mem); });
  };

  resource mapX = mkMap(memX), mapY = mkMap(memY), mapR = mkMap(memR);

  for (size_t batch_off = 0; batch_off < n; batch_off += batch_size) {
    auto start = std::chrono::high_resolution_clock::now();

    size_t local_n = std::min(batch_size, n - batch_off);

    // send the data
    auto xv = transposed(x_kn + k * batch_off, k, local_n);
    auto yv = transposed(y_dn + d * batch_off, d, local_n);
    auto rv = transposed(r_dn + d * batch_off, d, local_n);

    std::copy(xv.begin(), xv.end(), *mapX);
    std::copy(yv.begin(), yv.end(), *mapY);
    vut::flushMemory(dev, memX);
    vut::flushMemory(dev, memY);

    // run the shader
    auto cstart = std::chrono::high_resolution_clock::now();
    unmix_constants consts = { uint32_t(local_n), alpha, accel };

    cbuf.begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    cbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    cbuf.pushConstants(shader->pl,
                       vk::ShaderStageFlagBits::eCompute,
                       0,
                       sizeof(unmix_constants),
                       &consts);
    cbuf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute, shader->pl, 0, { ds }, {});
    cbuf.dispatch(vut::div_up(local_n, local_size), 1, 1);
    cbuf.end();
    cq.submit(
      { vk::SubmitInfo({}, {}, std::array<vk::CommandBuffer, 1>{ cbuf }) });
    cq.waitIdle();
    auto cfinish = std::chrono::high_resolution_clock::now();
    std::cout
      << "computation @" << batch_off << " took: "
      << std::chrono::duration<double, std::milli>(cfinish - cstart).count()
      << std::endl;

    // get the results out
    vut::invalidateMemory(dev, memX);
    vut::invalidateMemory(dev, memR);
    std::copy(*mapX, *mapX + xv.size(), xv.begin());
    std::copy(*mapR, *mapR + rv.size(), rv.begin());

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout
      << "batch @" << batch_off << " (size " << local_n << ") took: "
      << std::chrono::duration<double, std::milli>(finish - start).count()
      << std::endl;
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
