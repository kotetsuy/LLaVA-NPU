// SPDX-License-Identifier: AGPL-3.0-only
// Ryzen AI 1.9: synchronize recorded output BOs after XRT completion.
#include <xrt/xrt_bo.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <dlfcn.h>
#include <vector>

namespace {
using Sync = void (*)(xrt::bo*, xclBOSyncDirection, size_t, size_t);
using Wait = void (*)(const void*, const std::chrono::milliseconds&);
constexpr auto sync_name = "_ZN3xrt2bo4syncE18xclBOSyncDirectionmm";
constexpr auto wait_name = "_ZNK3xrt3run5wait2ERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE";
struct Region { xrt::bo bo; size_t bytes, offset; };
thread_local std::vector<Region> outputs;
thread_local bool active = false;
std::atomic<uint64_t> calls{0}, waits{0}, syncs{0}, live{0};
void clear() { live.fetch_sub(outputs.size()); outputs.clear(); }
}
extern "C" void sync_hook(xrt::bo*, xclBOSyncDirection, size_t, size_t) asm("_ZN3xrt2bo4syncE18xclBOSyncDirectionmm");
extern "C" void sync_hook(xrt::bo* bo, xclBOSyncDirection direction, size_t bytes, size_t offset) {
  static auto original = reinterpret_cast<Sync>(dlsym(RTLD_NEXT, sync_name));
  original(bo, direction, bytes, offset);
  if (!active || direction != XCL_BO_SYNC_BO_FROM_DEVICE || !bytes) return;
  for (const auto& r : outputs) if (r.bo == *bo && r.bytes == bytes && r.offset == offset) return;
  outputs.push_back({*bo, bytes, offset}); ++live;
}
extern "C" void wait_hook(const void*, const std::chrono::milliseconds&) asm("_ZNK3xrt3run5wait2ERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE");
extern "C" void wait_hook(const void* run, const std::chrono::milliseconds& timeout) {
  static auto original = reinterpret_cast<Wait>(dlsym(RTLD_NEXT, wait_name));
  static auto sync = reinterpret_cast<Sync>(dlsym(RTLD_NEXT, sync_name));
  original(run, timeout);
  if (!active) return;
  ++waits;
  if (!outputs.empty()) {
    for (auto& r : outputs) sync(&r.bo, XCL_BO_SYNC_BO_FROM_DEVICE, r.bytes, r.offset);
    ++syncs;
  }
  clear();
}
extern "C" void yolodemo_npu_sync_begin() { clear(); active = true; ++calls; }
extern "C" void yolodemo_npu_sync_end() { active = false; clear(); }
extern "C" void yolodemo_npu_sync_stats(uint64_t* v) { v[0]=calls; v[1]=waits; v[2]=syncs; v[3]=live; }
extern "C" const char* yolodemo_npu_sync_original_library(int which) {
  auto address = dlsym(RTLD_NEXT, which == 0 ? sync_name : wait_name);
  Dl_info info{}; return address && dladdr(address, &info) ? info.dli_fname : nullptr;
}
