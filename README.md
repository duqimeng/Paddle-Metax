# Paddle-MetaX

MetaX GPU backend for PaddlePaddle.

## 构建说明

本仓库采用与 Paddle-iluvatar 相同的独立仓模式，支持与 Paddle 联合编译，最终产出单独的硬件版 Paddle whl。

### 目录结构

```
Paddle-MetaX/
├── Paddle/                    # Paddle 子模块
├── cmake/                     # CMake 配置文件
├── common/                   # 通用代码
├── runtime/                   # 运行时实现
├── kernels/                   # Kernel 实现
├── cinn/                      # CINN 支持
├── headers/                   # 头文件
├── patches/                   # 补丁文件
├── build_paddle.sh            # 构建脚本
├── install_paddle.sh          # 安装脚本
├── clean_paddle.sh            # 清理脚本
└── CMakeLists.txt             # MetaX 后端 CMake 配置
```

### 构建步骤

1. 确保 Paddle 子模块已正确初始化：
   ```bash
   git submodule update --init --recursive
   ```

2. 执行联合编译：
   ```bash
   ./build_paddle.sh
   # 或使用增量构建脚本
   ./build_inc.sh
   ```

3. 安装生成的 whl 包：
   ```bash
   ./install_paddle.sh
   ```

### 环境变量

- `PYTHON_VERSION`: Python 版本 (默认自动检测)
- `METAX_VERSION`: MetaX 版本号 (默认自动生成)
- `METAX_ARCH`: CUDA 架构 (默认 80)
- `WITH_CINN`: 是否启用 CINN (默认 OFF)

### 清理

```bash
./build_inc.sh --clean
```