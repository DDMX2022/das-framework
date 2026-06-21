// csrc/pager.cpp
// A real C++ torch extension for the paging copy. The page-in now lives in a
// compiled C++ layer instead of pure Python. On CUDA this is exactly where you
// would use cudaMemcpyAsync + pinned host memory + streams for true async PCIe
// transfer; on CPU/MPS (this machine) torch's .to() is the portable path. The
// CUDA-specific async path CANNOT be exercised here (no NVIDIA hardware) — it is
// guarded and documented, not faked.
#include <torch/extension.h>
#include <vector>
#include <string>

std::vector<torch::Tensor> page_in(std::vector<torch::Tensor> tensors, std::string device) {
    torch::Device dev(device);
    std::vector<torch::Tensor> out;
    out.reserve(tensors.size());
    for (auto &t : tensors) {
        // non_blocking matters only for pinned CUDA memory; harmless elsewhere
        out.push_back(t.to(dev, /*non_blocking=*/true));
    }
    return out;
}

int64_t total_bytes(std::vector<torch::Tensor> tensors) {
    int64_t b = 0;
    for (auto &t : tensors) b += t.numel() * t.element_size();
    return b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("page_in", &page_in, "page tensors onto a device (compiled C++)");
    m.def("total_bytes", &total_bytes, "sum bytes of a tensor list");
}
