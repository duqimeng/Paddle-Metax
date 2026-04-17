#!/bin/bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

pip uninstall paddlepaddle-metax -y || echo "No existing paddlepaddle-metax installation found, skipping uninstallation."

# bash clean_paddle.sh || { echo "Error: Failed to clean paddle!"; exit 1; }


# rm -rf /root/m01097/Paddle-metax_test/Paddle/third_party/eigen3
# cp -r /root/m01097/Paddle-metax_test/eigen3/ /root/m01097/Paddle-metax_test/Paddle/third_party/eigen3
# git -C /root/m01097/Paddle-metax_test/Paddle config submodule.third_party/eigen3.update none

PYTHON_VERSION=${PYTHON_VERSION:-$(python3 -V 2>&1|awk '{print $2}')}


export MACA_AI_VERSION=$(cat /opt/maca/Version.txt | cut -d':' -f2)
WITH_CINN=${WITH_CINN:-OFF}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# CURRENT_DIR=$(pwd)
PLATFORM_ID=$(uname -i)
PADDLE_SOURCE_DIR="${SCRIPT_DIR}/Paddle"
PADDLE_BUILD_DIR="${PADDLE_SOURCE_DIR}/build"
METAX_SOURCE_DIR="${SCRIPT_DIR}"
METAX_BUILD_DIR="${PADDLE_BUILD_DIR}/custom_device_build"


# WITH_CINN=${WITH_CINN:-OFF}
# export CMAKE_CUDA_ARCHITECTURES=${COREX_ARCH}

# SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# PADDLE_SOURCE_DIR="${SCRIPT_DIR}/Paddle"
# PADDLE_BUILD_DIR="${PADDLE_SOURCE_DIR}/build"
# ILUVATAR_SOURCE_DIR="${SCRIPT_DIR}"
# ILUVATAR_BUILD_DIR="${PADDLE_BUILD_DIR}/custom_device_build"
# PATCH_FILE="${SCRIPT_DIR}/patches/paddle-corex.patch"
# STATE_FILE="${SCRIPT_DIR}/.build_state"




if [[ ! -d "${PADDLE_BUILD_DIR}" ]]; then
  echo "Creating Paddle build directory at ${PADDLE_BUILD_DIR}"
  mkdir -p "${PADDLE_BUILD_DIR}"
fi
if [[ ! -d "${METAX_BUILD_DIR}" ]]; then
  echo "Creating MetaX build directory at ${METAX_BUILD_DIR}"
  mkdir -p "${METAX_BUILD_DIR}"
fi


export CUCC_CMAKE_ENTRY=2
export MACA_PATH=/opt/maca
if [ ! -d ${HOME}/cu-bridge ]; then
    ${MACA_PATH}/tools/cu-bridge/tools/pre_make
fi
export CUDA_PATH=/root/cu-bridge/CUDA_DIR/
export PATH=${CUDA_PATH}/bin:${PATH}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export PATH=${PATH}:${CUCC_PATH}/tools:${CUCC_PATH}/bin
export PATH=${MACA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}



echo "MetaX source dir: ${METAX_SOURCE_DIR}"
echo "Paddle source dir: ${PADDLE_SOURCE_DIR}"
echo "MACA AI version: ${MACA_AI_VERSION}"
echo "Platform: ${PLATFORM_ID}"


PADDLE_CMAKE_ARGS=(
  "-DPY_VERSION=${PYTHON_VERSION}"
  "-DWITH_GPU=OFF"
  "-DWITH_DISTRIBUTE=ON"
  "-DWITH_CUSTOM_DEVICE_SUB_BUILD=ON"
  "-DCUSTOM_DEVICE_SOURCE_DIR=${METAX_SOURCE_DIR}"
)

CUSTOM_DEVICE_CMAKE_ARGS=(
  "-DPADDLE_SOURCE_DIR=${PADDLE_SOURCE_DIR}"
  "-DPY_VERSION=${PYTHON_VERSION:-3}"             # 如果环境变量未设置，默认为 3，对应 $(which python3) 的版本
  "-DCUDA_ARCH_NAME=Manual"
  "-DCUDA_ARCH_BIN=80"
  "-DCMAKE_BUILD_TYPE=Release"
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
  "-DPython3_EXECUTABLE=$(which python3)"
  "-DWITH_NCCL=ON"
  "-DNCCL_VERSION=0"
)



if [[ "${PLATFORM_ID}" == "aarch64" ]]; then
  CUSTOM_DEVICE_CMAKE_ARGS+=("-DWITH_ARM=ON")
  echo " Building for ARM architecture, enabling ARM-specific optimizations."
else
  CUSTOM_DEVICE_CMAKE_ARGS+=("-DWITH_ARM=OFF")
  echo " Building for non-ARM architecture, skipping ARM-specific optimizations."
fi

CUSTOM_DEVICE_CMAKE_ARGS_STR=$(IFS=';'; echo "${CUSTOM_DEVICE_CMAKE_ARGS[*]}")
PADDLE_CMAKE_ARGS+=("-DCUSTOM_DEVICE_CMAKE_ARGS=${CUSTOM_DEVICE_CMAKE_ARGS_STR}")

cd Paddle/build || { echo "Error: Failed to enter Paddle/build directory!"; exit 1; }

# cmake_maca "${PADDLE_CMAKE_ARGS[@]}" "${PADDLE_SOURCE_DIR}" 2>&1 | tee cmake_config_319.log \
#   || { echo "Error: CMake configuration failed! Check cmake_config.log for details."; exit 1; }

# 在数组展开后追加 -DCMAKE_BUILD_TYPE=Debug
cmake_maca "${PADDLE_CMAKE_ARGS[@]}" -DCMAKE_BUILD_TYPE=Debug "${PADDLE_SOURCE_DIR}" 2>&1 | tee cmake_config_416.log \
  || { echo "Error: CMake configuration failed! Check cmake_config.log for details."; exit 1; }


if [[ "${PLATFORM_ID}" == "aarch64" ]]; then
  env TARGET=ARMV8 make_maca -j$(nproc) || { echo "Error: Paddle build failed!"; exit 1; }
else
  echo " Building for non-ARM architecture, using default target settings."
  make_maca -j$(nproc) VERBOSE=1 2>&1 | tee build_416.log
fi
# popd

echo "-------------------------------------------------------------------"

pip install python/dist/paddlepaddle_metax-3.4.0.dev20260415-cp310-cp310-linux_x86_64.whl --force-reinstall


# pip install python/dist/paddlepaddle_metax-3.4.0.dev20260312-cp310-cp310-linux_x86_64.whl --force-reinstall
# pip install Paddle/build/python/dist/paddlepaddle_metax-3.4.0.dev20260312-cp310-cp310-linux_x86_64.whl --force-reinstall

# # Revert patch
# if git -C "$PADDLE_SOURCE_DIR" apply --reverse --check "$PATCH_FILE" > /dev/null 2>&1; then
#   git -C "$PADDLE_SOURCE_DIR" apply --reverse "$PATCH_FILE" || { echo "Error: Failed to revert patch!"; exit 1; }
#   echo "Patch successfully reverted!"
# fi

# pushd ${PADDLE_SOURCE_DIR}/third_party/eigen3
# git reset --hard || { echo "Error: Failed to reset eigen repository!"; exit 1; }
# popd