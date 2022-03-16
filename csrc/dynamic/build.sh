rm -rf ../../build
mkdir -p ../../build
cd ../../build
Torch_DIR=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
cmake -DBUILD_TEST=1 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=. ../csrc/dynamic
make install
