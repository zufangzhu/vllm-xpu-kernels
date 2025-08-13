#include <ATen/ATen.h>
#include <oneapi/dnnl/dnnl.hpp>

Tensor empty_dpcpp(IntArrayRef size, const TensorOptions& options,
                   c10::optional<MemoryFormat> optional_memory_format) {
  TORCH_INTERNAL_ASSERT(options.backend() == at::Backend::XPU ||
                        options.backend() == at::Backend::QuantizedXPU);
  // TORCH_INTERNAL_ASSERT(!options.is_variable()); // is_variable should have
  // been
  // "unpacked"

  auto* allocator = torch_ipex::xpu::dpcpp::getDeviceAllocator();
  int64_t nelements = torch_ipex::xpu::dpcpp::detail::prod_intlist(size);
  auto dtype = options.dtype();
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(), size_bytes,
      allocator->allocate(size_bytes), allocator,
      /*resizeable=*/true);
  auto tensor = detail::make_tensor<TensorImpl>(
      storage_impl, options.computeDispatchKey(), dtype);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  TORCH_CHECK(
      !(options.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; "
      "please delete "
      "the redundant setter.");

  auto memory_format = options.memory_format_opt().value_or(
      optional_memory_format.value_or(MemoryFormat::Contiguous));

  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  return tensor;
}

Tensor empty(IntArrayRef size, const TensorOptions& options,
             c10::optional<MemoryFormat> optional_memory_format) {
  return empty_dpcpp(size, options, optional_memory_format);
}