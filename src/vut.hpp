
#ifndef _vut_hpp
#define _vut_hpp

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace vut {

constexpr size_t
div_up(size_t n, size_t blk)
{
  return (n + blk - 1) / blk;
}

vk::Instance
createInstance(const char *app)
{
  vk::ApplicationInfo appInfo(app);
  return vk::createInstance(
    vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &appInfo));
}

vk::PhysicalDevice
findPhy(vk::Instance instance, const char *name)
{
  vk::PhysicalDevice phy;
  for (auto &d : instance.enumeratePhysicalDevices()) {
    auto props = d.getProperties();
    if (name && !strstr(props.deviceName, name))
      continue;
    return d;
  }

  return vk::PhysicalDevice(); // converts to boolean false
}

uint32_t
findQueueFamilyIdx(vk::PhysicalDevice phy,
                   vk::QueueFlags needed,
                   vk::QueueFlags undesired)
{
  // first try
  auto propss = phy.getQueueFamilyProperties();

  for (uint32_t i = 0; i < propss.size(); ++i) {
    auto &props = propss[i];
    if (~props.queueFlags & needed)
      continue;
    if (props.queueFlags & undesired)
      continue;
    return i;
  }

  // second try, ignore undesired

  for (uint32_t i = 0; i < propss.size(); ++i) {
    auto &props = propss[i];
    if (~props.queueFlags & needed)
      continue;
    return i;
  }

  // TODO throw?
  return -1;
}

template<size_t N>
vk::Device
createDevice(vk::PhysicalDevice phy,
             std::array<std::tuple<uint32_t, size_t, float>, N> queues)
{
  std::array<vk::DeviceQueueCreateInfo, N> a;
  for (size_t i = 0; i < N; ++i)
    a[i] = vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(),
                                     std::get<0>(queues[i]),
                                     std::get<1>(queues[i]),
                                     &std::get<2>(queues[i]));

  return phy.createDevice(vk::DeviceCreateInfo(
    vk::DeviceCreateFlags(), a, nullptr, nullptr, nullptr));
}

template<typename ELEM = char>
vk::Buffer
createStorageBuffer(vk::Device dev, size_t elems)
{
  return dev.createBuffer(
    vk::BufferCreateInfo(vk::BufferCreateFlags(),
                         elems * sizeof(ELEM),
                         vk::BufferUsageFlagBits::eStorageBuffer,
                         vk::SharingMode::eExclusive,
                         0,
                         nullptr));
}

uint32_t
findMemoryTypeIdx(vk::PhysicalDevice phy,
                  vk::MemoryPropertyFlags needed,
                  vk::MemoryPropertyFlags avoided = vk::MemoryPropertyFlags())
{
  uint32_t best = VK_MAX_MEMORY_TYPES;
  auto memTypes = phy.getMemoryProperties().memoryTypes;
  for (uint32_t i = 0; i < memTypes.size(); ++i) {
    auto &mt = memTypes[i];
    if (~mt.propertyFlags & (vk::MemoryPropertyFlagBits::eHostCached |
                             vk::MemoryPropertyFlagBits::eHostVisible))
      continue;
    if (best != VK_MAX_MEMORY_TYPES &&
        mt.propertyFlags > memTypes[best].propertyFlags)
      continue;
    best = i;
  }
  // TODO throw if best==VK_MAX_MEMORY_TYPES ?
  return best;
}

template<typename ELEM = char>
vk::DeviceMemory
allocateMemory(vk::Device dev, size_t elems, uint32_t memTypeIdx)
{
  return dev.allocateMemory(
    vk::MemoryAllocateInfo(elems * sizeof(ELEM), memTypeIdx));
}

struct Shader
{
  vk::DescriptorSetLayout dsl;
  vk::PipelineLayout pl;
  vk::ShaderModule mod;

  void destroy(vk::Device dev)
  {
    dev.destroyShaderModule(mod);
    dev.destroyPipelineLayout(pl);
    dev.destroyDescriptorSetLayout(dsl);
  }
};

Shader
createShader(vk::Device dev,
             size_t n_storage_buffers,
             size_t constants_size,
             size_t program_size,
             const uint32_t *program)
{
  Shader res;
  std::vector<vk::DescriptorSetLayoutBinding> v;
  v.reserve(n_storage_buffers);
  for (int i = 0; i < n_storage_buffers; ++i)
    v.emplace_back(i,
                   vk::DescriptorType::eStorageBuffer,
                   1,
                   vk::ShaderStageFlagBits::eCompute,
                   nullptr);

  res.dsl = dev.createDescriptorSetLayout(
    vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), v));

  res.pl = dev.createPipelineLayout(vk::PipelineLayoutCreateInfo(
    vk::PipelineLayoutCreateFlags(),
    std::array<vk::DescriptorSetLayout, 1>{ { res.dsl } },
    std::array<vk::PushConstantRange, 1>{ { { vk::ShaderStageFlagBits::eCompute,
                                              0,
                                              uint32_t(constants_size) } } }));

  res.mod = dev.createShaderModule(vk::ShaderModuleCreateInfo(
    vk::ShaderModuleCreateFlags(), program_size, program));

  return res;
}

template<class TUP, size_t... IS>
auto
tuple_spec_map_entries(TUP entries, std::index_sequence<IS...>)
  -> std::array<vk::SpecializationMapEntry, sizeof...(IS)>
{
  const char *entries_base =
    reinterpret_cast<char *>(&entries); // ooh this is bad
  return {
    { { uint32_t(IS),
        uint32_t(reinterpret_cast<const char *>(&get<IS>(entries)) -
                 entries_base),
        uint32_t(sizeof(typename std::tuple_element<IS, TUP>::type)) }... }
  };
}

template<typename... SpecArgs>
vk::Pipeline
createComputePipeline(vk::Device dev,
                      const Shader &shader,
                      const char *entrypoint,
                      SpecArgs... sa)
{

  std::tuple<SpecArgs...> entries(sa...);
  auto specs = tuple_spec_map_entries(
    entries, std::make_index_sequence<sizeof...(SpecArgs)>());
  auto spec_info = vk::SpecializationInfo(
    sizeof...(SpecArgs), specs.data(), sizeof(entries), &entries);

  return dev
    .createComputePipeline(
      nullptr,
      vk::ComputePipelineCreateInfo(
        vk::PipelineCreateFlags(),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                          vk::ShaderStageFlagBits::eCompute,
                                          shader.mod,
                                          entrypoint,
                                          &spec_info),
        shader.pl,
        nullptr,
        0))
    .value;
}

vk::DescriptorPool
createStorageDescriptorPool(vk::Device dev, size_t sets, size_t descriptors)
{
  std::array<vk::DescriptorPoolSize, 1> pools = {
    { { vk::DescriptorType::eStorageBuffer, uint32_t(descriptors) } }
  };
  return dev.createDescriptorPool(vk::DescriptorPoolCreateInfo(
    vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, sets, pools));
}

template<typename... SHADERS>
auto
allocateDescriptorSets(vk::Device dev,
                       vk::DescriptorPool pool,
                       SHADERS... shaders)
{
  auto res = dev.allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
    pool,
    std::array<vk::DescriptorSetLayout, sizeof...(SHADERS)>{
      { (shaders.dsl)... } }));
  std::array<vk::DescriptorSet, sizeof...(SHADERS)> a;
  for (size_t i = 0; i < sizeof...(SHADERS); ++i)
    a[i] = res[i]; // plain ugly
  return a;
}

template<size_t N>
void
writeDescriptorSet(vk::Device dev,
                   vk::DescriptorSet dset,
                   std::array<vk::Buffer, N> bufs)
{
  std::array<std::array<vk::DescriptorBufferInfo, 1>, N> dbis;
  std::array<vk::WriteDescriptorSet, N> writes;
  for (size_t i = 0; i < N; ++i) {
    dbis[i][0] = vk::DescriptorBufferInfo(bufs[i], 0, VK_WHOLE_SIZE);
    writes[i] = vk::WriteDescriptorSet(
      dset, i, 0, vk::DescriptorType::eStorageBuffer, nullptr, dbis[i]);
  }
  dev.updateDescriptorSets(writes, nullptr);
}

template<typename... BUFS>
void
writeDescriptorSet(vk::Device dev, vk::DescriptorSet dset, const BUFS &...bufs)
{
  writeDescriptorSet(
    dev,
    dset,
    std::array<vk::Buffer, sizeof...(BUFS)>{ { { (vk::Buffer)bufs }... } });
}

template<typename ELEM>
ELEM *
mapMemory(vk::Device dev, vk::DeviceMemory mem)
{
  return static_cast<ELEM *>(
    dev.mapMemory(mem, 0, VK_WHOLE_SIZE, vk::MemoryMapFlags()));
}

void
flushMemory(vk::Device dev, vk::DeviceMemory mem)
{
  auto range = std::array<vk::MappedMemoryRange, 1>{ { vk::MappedMemoryRange(
    mem, 0, VK_WHOLE_SIZE) } };
  dev.flushMappedMemoryRanges(range);
}

void
invalidateMemory(vk::Device dev, vk::DeviceMemory mem)
{
  auto range = std::array<vk::MappedMemoryRange, 1>{ { vk::MappedMemoryRange(
    mem, 0, VK_WHOLE_SIZE) } };
  dev.invalidateMappedMemoryRanges(range);
}

vk::CommandPool
createCmdPool(vk::Device dev, uint32_t queueFamIdx)
{
  return dev.createCommandPool(
    vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), queueFamIdx));
}

auto
allocateCmdBufs(vk::Device dev, vk::CommandPool pool, size_t n)
{
  return dev.allocateCommandBuffers(
    vk::CommandBufferAllocateInfo(pool, vk::CommandBufferLevel::ePrimary, 1));
}

}; // namespace vut

#endif // _vut_hpp
