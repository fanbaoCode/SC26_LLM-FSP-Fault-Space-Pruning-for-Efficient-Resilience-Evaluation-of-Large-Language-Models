"""pytorchfi.error_models 提供了多种开箱即用的错误模型"""
import random
import pytorchfi.core as core
import logging
import torch
import numpy as np
from pytorchfi.util import random_value


def random_weight_location(pfi, layer: int = -1):
    """
    随机选择一个权重位置进行错误注入
    
    参数:
        pfi: 故障注入对象
        layer: 指定层号，默认为-1表示随机选择
    
    返回:
        包含以下信息的元组:
        - 层号列表
        - 输出通道索引列表 
        - 输入通道索引列表
        - 高度维度索引列表
        - 宽度维度索引列表
    """
    if layer == -1:
        layer = random.randint(0, pfi.get_total_layers() - 1)

    dim = pfi.get_weights_dim(layer)  # 获取权重的维度
    shape = pfi.get_weights_size(layer)  # 获取权重的形状

    dim0_shape = shape[0]  # 输出通道数
    k = random.randint(0, dim0_shape - 1)  # 随机选择输出通道
    
    # 根据维度数初始化其他维度的随机值
    if dim > 1:
        dim1_shape = shape[1]
        dim1_rand = random.randint(0, dim1_shape - 1)  # 输入通道
    if dim > 2:
        dim2_shape = shape[2]
        dim2_rand = random.randint(0, dim2_shape - 1)  # 高度维度
    else:
        dim2_rand = None
    if dim > 3:
        dim3_shape = shape[3]
        dim3_rand = random.randint(0, dim3_shape - 1)  # 宽度维度
    else:
        dim3_rand = None

    return ([layer], [k], [dim1_rand], [dim2_rand], [dim3_rand])


# 权重扰动模型
def random_weight_inj(
    pfi, corrupt_layer: int = -1, min_val: int = -1, max_val: int = 1
):
    """
    随机权重注入 - 将随机值注入到随机权重位置
    
    参数:
        pfi: 故障注入对象
        corrupt_layer: 指定要注入错误的层，-1表示随机选择
        min_val: 随机值的最小范围
        max_val: 随机值的最大范围
    
    返回:
        配置好的故障注入对象
    """
    layer, k, c_in, kH, kW = random_weight_location(pfi, corrupt_layer)
    faulty_val = [random_value(min_val=min_val, max_val=max_val)]  # 生成随机错误值

    return pfi.declare_weight_fault_injection(
        layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW, value=faulty_val
    )


def zero_func_rand_weight(pfi: core.FaultInjection):
    """
    随机权重置零 - 将随机选择的权重位置置为0
    
    参数:
        pfi: 故障注入对象
    
    返回:
        配置好的故障注入对象
    """
    layer, k, c_in, kH, kW = random_weight_location(pfi)
    return pfi.declare_weight_fault_injection(
        function=_zero_rand_weight, layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW
    )


def _zero_rand_weight(data, location):
    """
    辅助函数 - 将指定位置的权重置零
    
    参数:
        data: 权重数据
        location: 要置零的位置
    
    返回:
        置零后的新数据
    """
    new_data = data[location] * 0
    return new_data

def multi_weight_inj(pfi, sdc_p=1e-5, function=_zero_rand_weight):
    """
    多权重注入 - 以一定概率在多个权重位置注入错误
    
    参数:
        pfi: 故障注入对象
        sdc_p: 每个权重位置发生错误的概率
        function: 使用的错误注入函数，默认为置零
    
    返回:
        配置好的故障注入对象
    """
    corrupt_idx = [[], [], [], [], []]  # 初始化错误位置索引
    
    # 遍历所有层和所有权重位置
    for layer_idx in range(pfi.get_total_layers()):
        shape = list(pfi.get_weights_size(layer_idx))
        dim_len = len(shape)  # 权重张量的实际维度
        shape.extend([1 for i in range(4 - len(shape))])  # 填充到4维
        
        # 遍历所有可能的权重位置
        for k in range(shape[0]):
            for dim1 in range(shape[1]):
                for dim2 in range(shape[2]):
                    for dim3 in range(shape[3]):
                        # 以sdc_p概率选择该位置进行错误注入
                        if random.random() < sdc_p:
                            idx = [layer_idx, k, dim1, dim2, dim3]
                            # 记录错误位置
                            for i in range(dim_len + 1):
                                corrupt_idx[i].append(idx[i])
                            for i in range(dim_len + 1, 5):
                                corrupt_idx[i].append(None)
    
    return pfi.declare_weight_fault_injection(
        layer_num=corrupt_idx[0],
        k=corrupt_idx[1],
        dim1=corrupt_idx[2],
        dim2=corrupt_idx[3],
        dim3=corrupt_idx[4],
        function=function,
    )
    
# 定义双比特翻转函数
def _flip_two_bits_signed(self, orig_value, bit_pos1, bit_pos2):
    """
    执行有符号数的双比特翻转
    """
    save_type = orig_value.dtype
    save_device = orig_value.device
    logging.info(f"Original Value: {orig_value}")
    
    if orig_value.numel() != 1:
        raise ValueError("只支持单个标量进行双比特翻转")
    
    if save_type == torch.float16:
        np_float = np.float16
        np_uint = np.uint16
        total_bits = 16
    elif save_type == torch.float32:
        np_float = np.float32
        np_uint = np.uint32
        total_bits = 32
    elif save_type == torch.float64:
        np_float = np.float64
        np_uint = np.uint64
        total_bits = 64
    else:
        raise TypeError(f"_flip_two_bits_signed 只支持 float16/32/64，但收到 {save_type}")
    
    if bit_pos1 < 0 or bit_pos1 >= total_bits or bit_pos2 < 0 or bit_pos2 >= total_bits:
        raise ValueError(f"bit_pos1 和 bit_pos2 必须在 [0, {total_bits-1}] 范围内，但收到 bit_pos1={bit_pos1}, bit_pos2={bit_pos2}")
    if bit_pos1 == bit_pos2:
        raise ValueError(f"bit_pos1 和 bit_pos2 不能相同，但收到 bit_pos1={bit_pos1}, bit_pos2={bit_pos2}")
    
    np_val = orig_value.detach().cpu().numpy().astype(np_float)
    raw_bits = np_val.view(np_uint).item()
    new_bits = raw_bits ^ (1 << bit_pos1) ^ (1 << bit_pos2)
    new_value = np.array([new_bits], dtype=np_uint).view(np_float)[0]
    
    return torch.tensor(new_value, dtype=save_type, device=save_device)

def double_bit_flip_signed_across_weights(self, layer, locations):
    """
    对指定层的权重执行双比特翻转
    """
    def hook(module, input, output):
        save_type = module.weight.dtype
        if save_type == torch.float16:
            self.bits = 16
        elif save_type == torch.float32:
            self.bits = 32
        elif save_type == torch.float64:
            self.bits = 64
        else:
            raise TypeError(f"Unsupported dtype {save_type}")

        new_locations = []
        for loc in locations:
            k, dim1, dim2, dim3 = loc
            prev_value = module.weight[k, dim1, dim2, dim3].clone() if dim3 is not None else \
                        module.weight[k, dim1, dim2].clone() if dim2 is not None else \
                        module.weight[k, dim1].clone()
            bit_pos1 = random.randint(0, self.bits - 1)
            possible_bit_pos = [i for i in range(self.bits) if i != bit_pos1]
            bit_pos2 = random.choice(possible_bit_pos)
            logging.info(f"Layer {layer}, Position ({k}, {dim1}, {dim2}, {dim3}), Bit Positions: {bit_pos1}, {bit_pos2}")
            new_value = self._flip_two_bits_signed(prev_value, bit_pos1, bit_pos2)
            
            if dim3 is not None:
                module.weight[k, dim1, dim2, dim3] = new_value
            elif dim2 is not None:
                module.weight[k, dim1, dim2] = new_value
            else:
                module.weight[k, dim1] = new_value
                
            new_locations.append({
                "layer": layer,
                "k": k,
                "dim1": dim1,
                "dim2": dim2,
                "dim3": dim3,
                "bit_position": (bit_pos1, bit_pos2),
                "prev_value": prev_value.item(),
                "faulty_value": new_value.item(),
                "weight_shape": tuple(module.weight.shape),
                "module_type": module.__class__.__name__
            })
        return output, new_locations

    return self._apply_hook(layer, hook)

def random_weight_two_bit_inj(pfi):
    """
    随机选择单个权重位置进行双比特翻转
    """
    layer, k, c_in, kH, kW = random_weight_location(pfi)
    pfi._last_rand_loc = (layer, k, c_in, kH, kW)
    
    return pfi.declare_weight_fault_injection(
        layer_num=[layer],
        k=[k],
        dim1=[c_in],
        dim2=[kH],
        dim3=[kW],
        function=double_bit_flip_signed_across_weights
    )