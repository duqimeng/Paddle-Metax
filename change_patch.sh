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

cd patches

echo ">>> Unzipping Eigen_3.4.0_paddle.zip..."

unzip Eigen_3.4.0_paddle.zip

echo ">>> Renaming folder to 'eigen3'..."
mv Eigen_3.4.0_paddle eigen3

cd ..

echo ">>> Removing old eigen3 directory from Paddle/third_party..."
rm -r Paddle/third_party/eigen3

echo ">>> Copying new eigen3 to Paddle/third_party..."

cp -r patches/eigen3/ Paddle/third_party/eigen3


echo ">>> Cleaning up temporary patch files..."


rm -r patches/eigen3

cd Paddle/

echo ">>> Applying main patch: paddle.patch..."

git apply --verbose ../patches/paddle.patch
echo ">>> Applying fix patch: patch_nullptr.patch..."

git apply --verbose ../patches/patch_nullptr.patch

cd -