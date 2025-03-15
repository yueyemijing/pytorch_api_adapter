import numpy as np
import torch

# ---------------------- 独立全局开关 ----------------------
USE_NUMPY_SOFTMAX = False  # 控制softmax的实现方式
USE_NUMPY_RMSNORM = False  # 控制rms_norm的实现方式

# ---------------------- 核心实现 ----------------------
def softmax(input, dim=None, dtype=None):
    if USE_NUMPY_SOFTMAX:
        # 类型转换处理
        is_tensor = isinstance(input, torch.Tensor)
        x = input.cpu().numpy() if is_tensor else input
        
        # 核心计算逻辑
        dim = -1 if dim is None else dim
        max_x = np.max(x, axis=dim, keepdims=True)
        exp_x = np.exp(x - max_x)
        sum_exp = np.sum(exp_x, axis=dim, keepdims=True)
        result = exp_x / sum_exp
        
        # 恢复原始类型
        if dtype is not None:
            result = result.astype(dtype)
        return torch.from_numpy(result).to(input.device) if is_tensor else result
    else:
        return torch.nn.functional.softmax(input, dim=dim, dtype=dtype)

def rms_norm(input, normalized_shape, weight, eps=1e-6):
    if USE_NUMPY_RMSNORM:
        # 类型转换处理
        is_tensor = isinstance(input, torch.Tensor)
        x = input.cpu().numpy() if is_tensor else input
        w = weight.cpu().numpy() if isinstance(weight, torch.Tensor) else weight
        
        # 形状验证
        if x.shape[-len(normalized_shape):] != normalized_shape:
            raise ValueError(f"Shape mismatch: input.shape[-{len(normalized_shape)}:] must be {normalized_shape}")
        
        # 核心计算逻辑
        axis = tuple(range(-len(normalized_shape), 0))
        squared = np.square(x)
        var = np.mean(squared, axis=axis, keepdims=True)
        result = x / np.sqrt(var + eps) * w
        
        return torch.from_numpy(result).to(input.device) if is_tensor else result
    else:
        return torch.nn.functional.rms_norm(input, normalized_shape, weight, eps)

# ---------------------- 使用示例 ----------------------
if __name__ == "__main__":
    # 创建测试数据
    x = torch.randn(2, 4)
    weight = torch.ones(4)

    # 独立控制开关
    global USE_NUMPY_SOFTMAX, USE_NUMPY_RMSNORM
    
    # 场景1：全部使用PyTorch实现
    USE_NUMPY_SOFTMAX = False
    USE_NUMPY_RMSNORM = False
    y1 = softmax(x, dim=1)
    z1 = rms_norm(x, (4,), weight)

    # 场景2：混合使用实现
    USE_NUMPY_SOFTMAX = True
    USE_NUMPY_RMSNORM = False
    y2 = softmax(x, dim=1)
    z2 = rms_norm(x, (4,), weight)

    # 场景3：全部使用NumPy实现
    USE_NUMPY_SOFTMAX = True
    USE_NUMPY_RMSNORM = True
    y3 = softmax(x, dim=1)
    z3 = rms_norm(x, (4,), weight)

    # 验证结果一致性
    print("Softmax误差(PyTorch vs NumPy):", torch.max(torch.abs(y1 - y2)).item())
    print("RMSNorm误差(PyTorch vs NumPy):", torch.max(torch.abs(z1 - z2)).item())
