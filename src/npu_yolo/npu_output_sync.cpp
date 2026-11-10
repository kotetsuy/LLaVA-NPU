// SPDX-License-Identifier: AGPL-3.0-only
// Compatibility shim for the fingerprinted Ryzen AI 1.8 runtime. Its runner
// invalidates output before execution; invalidate again after wait2, before
// CPU depadding. All BO references belong to a single runner invocation.
#include <xrt/xrt_bo.h>
#include <atomic>
#include <cstdint>
#include <dlfcn.h>
#include <stdexcept>
#include <vector>

namespace {
constexpr auto sync_symbol = "_ZN3xrt2bo4syncE18xclBOSyncDirectionmm";
constexpr auto run_symbol = "_ZN15DpuKernelRunner3runERKSt6vectorIPN4vart12TensorBufferESaIS3_EES7_";
constexpr auto pdi_symbol = "_ZN15DpuKernelRunner7run_pdiEi";
using Sync = void (*)(xrt::bo*, xclBOSyncDirection, size_t, size_t);
using Run = void (*)(void*, const void*, const void*);
using Pdi = void (*)(void*, int);

template<typename T> T next(const char* symbol) {
  auto address = dlsym(RTLD_NEXT, symbol);
  if (!address) throw std::runtime_error("NPU output sync: missing runtime symbol");
  return reinterpret_cast<T>(address);
}

struct Region { xrt::bo buffer; size_t bytes; size_t offset; };
std::atomic<uint64_t> runners{0}, pdis{0}, synchronized{0}, live_regions{0};
struct Frame;
thread_local Frame* active = nullptr;
struct Frame {
  void* runner;
  Frame* previous;
  std::vector<Region> outputs;
  uint64_t completions = 0;
  explicit Frame(void* owner) : runner(owner), previous(active) { active = this; }
  ~Frame() {
    active = previous;
    live_regions.fetch_sub(outputs.size());
    // outputs and their shared BO handles are destroyed here, on exceptions too.
  }
};
}

extern "C" void output_sync_bo(xrt::bo*, xclBOSyncDirection, size_t, size_t)
    asm("_ZN3xrt2bo4syncE18xclBOSyncDirectionmm");
extern "C" void output_sync_bo(xrt::bo* bo, xclBOSyncDirection direction,
                               size_t bytes, size_t offset) {
  static auto original = next<Sync>(sync_symbol);
  original(bo, direction, bytes, offset);
  if (!active || direction != XCL_BO_SYNC_BO_FROM_DEVICE || !bytes) return;
  for (const auto& r : active->outputs)
    if (r.buffer == *bo && r.bytes == bytes && r.offset == offset) return;
  active->outputs.push_back({*bo, bytes, offset});
  ++live_regions;
}

extern "C" void output_sync_run(void*, const void*, const void*)
    asm("_ZN15DpuKernelRunner3runERKSt6vectorIPN4vart12TensorBufferESaIS3_EES7_");
extern "C" void output_sync_run(void* runner, const void* inputs, const void* outputs) {
  static auto original = next<Run>(run_symbol);
  Frame frame(runner);
  ++runners;
  original(runner, inputs, outputs);
  if (!frame.completions)
    throw std::runtime_error("NPU output sync: runner completed without synchronized NPU execution");
}

extern "C" void output_sync_pdi(void*, int) asm("_ZN15DpuKernelRunner7run_pdiEi");
extern "C" void output_sync_pdi(void* runner, int index) {
  static auto original = next<Pdi>(pdi_symbol);
  static auto sync = next<Sync>(sync_symbol);
  if (!active || active->runner != runner || active->outputs.empty())
    throw std::runtime_error("NPU output sync: no output buffers for current runner");
  original(runner, index);  // Includes successful wait2; do not read outputs yet.
  ++pdis;
  for (auto& r : active->outputs)
    sync(&r.buffer, XCL_BO_SYNC_BO_FROM_DEVICE, r.bytes, r.offset);
  ++synchronized;
  ++active->completions;
}

extern "C" void yolodemo_npu_sync_stats(uint64_t* result) {
  result[0] = runners.load();
  result[1] = pdis.load();
  result[2] = synchronized.load();
  result[3] = live_regions.load();
}

extern "C" const char* yolodemo_npu_sync_original_library(int which) {
  auto address = dlsym(RTLD_NEXT, which == 0 ? run_symbol : which == 1 ? pdi_symbol : sync_symbol);
  Dl_info info{};
  return address && dladdr(address, &info) ? info.dli_fname : nullptr;
}
