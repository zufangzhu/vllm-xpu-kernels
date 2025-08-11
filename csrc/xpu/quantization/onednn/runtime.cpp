#include "runtime.h"
#include "xpu/quantization/onednn/onednn_utils.h"

/* XXX WARNING:
 * Do NOT put any kernels or call any device binaries here!
 * Only maintain oneDNN runtime states in this file.
 * */

namespace oneDNN {

GpuEngineManager& GpuEngineManager::Instance() {
  static GpuEngineManager myInstance;
  return myInstance;
}

GpuStreamManager& GpuStreamManager::Instance() {
  static thread_local GpuStreamManager myInstance;
  return myInstance;
}

bool set_onednn_verbose(int level) {
  dnnl::status rs = dnnl::set_verbose(level);
  return rs == dnnl::status::success;
}

}  // namespace oneDNN
