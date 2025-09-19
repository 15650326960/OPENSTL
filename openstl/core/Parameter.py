
#from openstl.core import calculate_model_size

def calculate_model_size(model, model_name="Model"):
    """
    打印模型大小 (MB)，基于模型参数数量和数据类型计算。

    Args:
        model: PyTorch 模型实例.
        model_name: 模型名称，用于打印信息.
    """
    # 计算模型所有参数数量
    total_params = sum(p.numel() for p in model.parameters())
    # 假设所有参数为 float32，每个参数占 4 字节
    param_size_bytes = total_params * 4
    model_size_mb = param_size_bytes / (1024 ** 2)  # 转换为 MB
    print(f"{model_name} size: {model_size_mb:.2f} MB")