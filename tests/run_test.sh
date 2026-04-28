# 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
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

# pip install scipy -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple some-package
SCRIPT_DIR=$(dirname "$0")
LEGACY_TEST_PATH="${SCRIPT_DIR}/../Paddle/test/legacy_test"
TEST_PATH1="${SCRIPT_DIR}/../python"
TEST_PATH2="${SCRIPT_DIR}/../python/tests"
TEST_PATH3="${SCRIPT_DIR}/../Paddle/test/rnn"
export PYTHONPATH="${LEGACY_TEST_PATH}:${PYTHONPATH}:${TEST_PATH1}:${TEST_PATH2}:${TEST_PATH3}"
export PADDLE_XCCL_BACKEND=metax_gpu
export CUDA_VISIBLE_DEVICES=0

# export
# sleep 1000000
# 2. 打印变量的值
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "LEGACY_TEST_PATH: $LEGACY_TEST_PATH"
echo "TEST_PATH1: $TEST_PATH1"
echo "TEST_PATH2: $TEST_PATH2"

# 3. (可选) 如果你想看它们解析后的绝对路径，可以使用 realpath
echo "--------------------------------"
echo "解析后的绝对路径:"
echo "SCRIPT_DIR (绝对): $(realpath "$SCRIPT_DIR")"
echo "LEGACY_TEST_PATH (绝对): $(realpath "$LEGACY_TEST_PATH")"
echo "TEST_PATH1 (绝对): $(realpath "$TEST_PATH1")"
echo "TEST_PATH2 (绝对): $(realpath "$TEST_PATH2")"
# exit 


rm -r build
mkdir -p build && cd build


TEST_LOG_LEVEL=0
TEST_LIST_FILE=""
TEST_LOG_OUTPUT_DIR=""
TEST_PARALLEL_NUM=1
IGNORE_BLOCKS=""   

while getopts "i:o:v:j:b:h" opt; do
  case "$opt" in
    i)
      TEST_LIST_FILE="$OPTARG"
      ;;
    o)
      TEST_LOG_OUTPUT_DIR="$OPTARG"
      echo "Set log output dir [ $TEST_LOG_OUTPUT_DIR ]"
      ;;
    v)
      TEST_LOG_LEVEL=$OPTARG
      ;;
    j)
      TEST_PARALLEL_NUM="$OPTARG"
      ;;
    b)
      IGNORE_BLOCKS="$OPTARG"
      echo "Set ignore blocks: $IGNORE_BLOCKS"
      ;;
    h)
      echo "用法：$0 -i <测试列表文件> -o <日志输出路径> ..."
      echo "  -i  测试程序列表文件"
      echo "  -o  日志输出路径"
      echo "  -v  GLOG_v 日志等级"
      echo "  -j  ctest 并行数量"
      echo "  -b  忽略的块列表，例如：\"elementwise;conv\" 或 NONE"
      echo "  -h  显示帮助"
      exit 0
      ;;
    \?)
      echo "error: unknow option '-$OPTARG'."
      exit 1
      ;;
    :)
      echo "error option '-$OPTARG' must have parameter."
      exit 1
      ;;
  esac
done


export GLOG_v=$TEST_LOG_LEVEL

# 定义日志文件路径
BUILD_LOG="build.log"
TEST_LOG="test.log"

echo ">>> 开始执行 CMake 配置 (Configure)..."
# 1. 配置阶段日志
cmake .. \
    -DTEST_LIST_FILE="$TEST_LIST_FILE" \
    -DLOG_OUTPUT_DIR="$TEST_LOG_OUTPUT_DIR" \
    -DIGNORE_BLOCKS="$IGNORE_BLOCKS" \
    2>&1 | tee cmake_config.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ CMake 配置失败，请查看 cmake_config.log"
    exit 1
fi

echo ">>> 开始执行编译 (Build)..."
cmake --build . \
    2>&1 | tee "$BUILD_LOG"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ 编译失败，请查看 $BUILD_LOG"
    exit 1
fi

echo ">>> 开始执行测试 (Test)..."
ctest -j"$TEST_PARALLEL_NUM" --output-on-failure \
    2>&1 | tee "$TEST_LOG"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ 测试未通过，请查看 $TEST_LOG"
fi

echo ">>> 删除编译缓存产物"
rm -r ../../tests/profiler_log/
rm -r ../../tests/log/
rm -r ../../tests/testdebug/


echo ">>> 所有操作完成。"
