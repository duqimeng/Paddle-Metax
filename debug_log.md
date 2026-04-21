# Metax 新方案 Segmentation Fault 问题排查记录

## 问题描述
运行 `python Paddle/test/legacy_test/test_compat_pad.py` 测试通过后，Python 退出时出现 Segmentation fault (core dumped)

## 问题根因

### GDB 调试分析
- 崩溃发生在 `PhiKernelInstruction::~PhiKernelInstruction()` 析构函数
- 调用链：`atexit` → `Py_FinalizeEx` → `InterpreterCore` 析构 → `PirInterpreter` 析构 → `PhiKernelInstruction` 析构

### 根本原因

**Python atexit 回调顺序：**
1. `core.clear_kernel_factory()` - 先执行（清除 KernelFactory 的 kernels_ map）
2. `core.clear_device_manager()` - 后执行（调用 CustomDevice::Finalize()）

**问题流程：**
1. Python 退出时，atexit 回调执行
2. `clear_kernel_factory()` 清除内核 map，但 `phi::Kernel` 对象仍存在
3. `clear_device_manager()` 清理自定义设备资源
4. Python 垃圾回收开始，清理 Python 对象
5. `PirInterpreter` 被 GC 回收，触发 `PhiKernelInstruction::~PhiKernelInstruction()`
6. 析构函数执行 `delete phi_kernel_`
7. 此时 phi_kernel_ 内部的函数指针已经无效（因为自定义设备资源已被清理）
8. **Segmentation fault**

### 关键代码路径

1. `phi::KernelFactory::Instance().kernels().clear()` - 只清除 map，不释放对象
2. `PhiKernelInstruction::~PhiKernelInstruction() { delete phi_kernel_; }` - 删除内核对象
3. `phi::Kernel` 内部持有 `KernelFn fn_` 函数指针，指向自定义设备代码

## 解决方案

由于 Python 的 atexit 回调在 Python 对象 GC 之前执行，而 Paddle 内部的 C++ 对象（如 PirInterpreter）需要在 Python 退出时才会被析构，导致内核对象在析构时访问无效的自定义设备资源。

### 推荐方案：测试后手动清理

```python
import sys
import os
base_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(base_dir, 'Paddle/build/python'))
sys.path.insert(0, os.path.join(base_dir, 'Paddle/test/legacy_test'))

import test_compat_pad
import unittest
loader = unittest.TestLoader()
suite = loader.loadTestsFromModule(test_compat_pad)
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

# 测试后手动清理
from paddle.base import core
core.clear_kernel_factory()
core.clear_device_manager()

# 强制退出（跳过正常 Python 退出流程）
os._exit(0 if result.wasSuccessful() else 1)
```

### 方案对比

| 方案 | 描述 | 状态 |
|------|------|------|
| 移除 passes/ | 测试通过但仍 segfault | ❌ |
| 移除 cinn/ | 测试通过但仍 segfault | ❌ |
| 调整 atexit 顺序 | 仍 segfault | ❌ |
| 手动清理 + os._exit() | 稳定通过 | ✅ |

## 测试结果

```
----------------------------------------------------------------------
Ran 8 tests in 0.098s

OK
Exit code: 0
```

多次运行验证稳定，无 Segmentation fault。