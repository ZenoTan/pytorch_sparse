#include <torch/extension.h>
#include <vector>

template<typename T> bool equal(torch::Tensor tensor, const std::vector<T> &vec) {
    if (tensor.size(0) != vec.size()) {
        return false;
    }
    auto ptr = tensor.data_ptr<T>();
    for (int i = 0; i < vec.size(); i++) {
        if (ptr[i] != vec[i]) {
            return false;
        }
    }
    return true;
}
