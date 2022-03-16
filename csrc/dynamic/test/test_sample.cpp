#include <algorithm>
#include <iostream>
#include <numeric>

#include <dynamic_store.h>
#include <torch/extension.h>

#include "utils.h"

void test_sample_static() {
  auto csr_options = torch::TensorOptions().dtype(torch::kInt64);
  auto val_options = torch::TensorOptions().dtype(torch::kInt32);

  torch::Tensor rowptr0 = torch::empty(4, csr_options);
  std::vector<int64_t> row_vec0 = {0, 3, 5, 9};
  std::copy(row_vec0.begin(), row_vec0.end(), rowptr0.data_ptr<int64_t>());
  torch::Tensor rowptr1 = torch::empty(4, csr_options);
  std::vector<int64_t> row_vec1 = {0, 1, 3, 5};
  std::copy(row_vec1.begin(), row_vec1.end(), rowptr1.data_ptr<int64_t>());

  torch::Tensor col0 = torch::empty(9, csr_options);
  std::vector<int64_t> col_vec0{1, 2, 3, 0, 2, 0, 1, 4, 5};
  std::copy(col_vec0.begin(), col_vec0.end(), col0.data_ptr<int64_t>());
  torch::Tensor col1 = torch::empty(5, csr_options);
  std::vector<int64_t> col_vec1{0, 2, 5, 2, 4};
  std::copy(col_vec1.begin(), col_vec1.end(), col1.data_ptr<int64_t>());

  torch::Tensor val0 = torch::empty(9, val_options);
  std::vector<int> val_vec0(9);
  std::iota(val_vec0.begin(), val_vec0.end(), 0);
  std::copy(val_vec0.begin(), val_vec0.end(), val0.data_ptr<int>());
  torch::Tensor val1 = torch::empty(5, val_options);
  std::vector<int> val_vec1(5);
  std::iota(val_vec1.begin(), val_vec1.end(), 9);
  std::copy(val_vec1.begin(), val_vec1.end(), val1.data_ptr<int>());

  torch::Tensor idx = torch::empty(4, csr_options);
  std::vector<int64_t> idx_vec(4);
  std::iota(idx_vec.begin(), idx_vec.end(), 2);
  std::copy(idx_vec.begin(), idx_vec.end(), idx.data_ptr<int64_t>());

  DynamicStore<int> store(2, 3, 6, 100);
  store.append_block(rowptr0, col0, val0);
  store.append_block(rowptr1, col1, val1);

  auto [out_rowptr0, out_col0, out_n_id0, out_val0] =
      store.sample_adj(idx, -1, false);
  std::vector<int64_t> expected_rowptr{0, 4, 5, 7, 9};
  std::vector<int64_t> expected_col{2, 3, 4, 5, 4, 0, 3, 0, 2};
  std::vector<int> expected_val{7, 8, 5, 6, 9, 10, 11, 12, 13};
  std::vector<int64_t> expected_n_id{2, 3, 4, 5, 0, 1};
  assert(equal(out_rowptr0, expected_rowptr));
  assert(equal(out_col0, expected_col));
  assert(equal(*out_val0, expected_val));
  assert(equal(out_n_id0, expected_n_id));

  auto [out_rowptr1, out_col1, out_n_id1, out_val1] =
      store.sample_adj(idx, 2, true);
  assert(out_col1.size(0) == 8);

  auto [out_rowptr2, out_col2, out_n_id2, out_val2] =
      store.sample_adj(idx, 2, false);
  assert(out_col2.size(0) == 7);
}

int main() {
  test_sample_static();
  return 0;
}
