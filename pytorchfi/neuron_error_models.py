"""
pytorchfi.error_models 模块提供了多种开箱即用的错误模型，用于在PyTorch模型中进行故障注入测试。
该模块主要包含神经元扰动模型，支持随机值注入等多种错误类型。
"""

import logging  # 用于记录日志信息
import random   # 用于生成随机数和随机选择
import numpy as np  # 数值计算库
import torch    # PyTorch深度学习框架

from pytorchfi import core  # 导入故障注入核心模块
from pytorchfi.util import random_value  # 导入随机值生成工具
from copy import copy  # 用于对象的浅拷贝

# ---------------------------- 辅助函数 ----------------------------

def random_batch_element(pfi: core.FaultInjection):
    """
    随机选择一个batch中的元素索引
    
    参数:
        pfi: 故障注入对象，包含batch大小等信息
    
    返回:
        随机选择的batch索引(0到batch_size-1之间的整数)
    """
    return random.randint(0, pfi.batch_size - 1)

def random_neuron_location(pfi: core.FaultInjection, layer: int = -1):
    """
    随机选择神经元位置(层和维度索引)
    
    参数:
        pfi: 故障注入对象
        layer: 指定层号，-1表示随机选择
    
    返回:
        元组: (层号, 通道维度索引, 高度维度索引, 宽度维度索引)
        注: 高度和宽度维度索引可能为None，取决于层的维度
    """
    if layer == -1:
        layer = random.randint(0, pfi.get_total_layers() - 1)
        # layer = 4
    dim = pfi.get_layer_dim(layer)  # 获取层的维度数
    shape = pfi.get_layer_shape(layer)  # 获取层的形状

    dim1_shape = shape[1]
    dim1_rand = random.randint(0, dim1_shape - 1)  # 随机选择通道维度索引
    
    # 根据层的维度数决定是否随机选择高度和宽度维度
    if dim > 2:
        dim2_shape = shape[2]
        dim2_rand = random.randint(0, dim2_shape - 1)
    else:
        dim2_rand = None
        
    if dim > 3:
        dim3_shape = shape[3]
        dim3_rand = random.randint(0, dim3_shape - 1)
    else:
        dim3_rand = None

    return (layer, dim1_rand, dim2_rand, dim3_rand)

# ---------------------------- 神经元扰动模型 ----------------------------

def random_neuron_inj(pfi: core.FaultInjection, min_val: int = -1, max_val: int = 1):
    """
    单神经元随机值注入 - 在单个batch元素中随机选择一个神经元注入随机值
    
    参数:
        pfi: 故障注入对象
        min_val: 随机值下限
        max_val: 随机值上限
    
    返回:
        配置好的故障注入对象
    """
    b = random_batch_element(pfi)  # 随机选择batch元素
    (layer, C, H, W) = random_neuron_location(pfi)  # 随机选择神经元位置
    err_val = random_value(min_val=min_val, max_val=max_val)  # 生成随机错误值

    return pfi.declare_neuron_fault_injection(
        batch=[b], layer_num=[layer], dim1=[C], dim2=[H], dim3=[W], value=[err_val]
    )

def random_neuron_inj_batched(
    pfi: core.FaultInjection,
    min_val: int = -1,
    max_val: int = 1,
    rand_loc: bool = True,
    rand_val: bool = True
):
    """
    批量随机神经元注入 - 在每个batch元素中随机选择一个神经元注入错误
    
    参数:
        pfi: 故障注入对象
        min_val: 随机值下限
        max_val: 随机值上限
        rand_loc: 是否每个batch使用不同的随机位置
        rand_val: 是否每个batch使用不同的随机值
    
    返回:
        配置好的故障注入对象
    """
    # 初始化存储各维度索引和值的列表
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for _ in range(6))

    # 如果不随机位置，预先选择一个位置
    if not rand_loc:
        (layer, C, H, W) = random_neuron_location(pfi)
    # 如果不随机值，预先生成一个错误值
    if not rand_val:
        err_val = random_value(min_val=min_val, max_val=max_val)

    # 遍历batch中的每个元素
    for i in range(pfi.batch_size):
        if rand_loc:
            (layer, C, H, W) = random_neuron_location(pfi)
        if rand_val:
            err_val = random_value(min_val=min_val, max_val=max_val)

        # 收集各维度的索引和值
        batch.append(i)
        layer_num.append(layer)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(err_val)

    return pfi.declare_neuron_fault_injection(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )

def random_inj_per_layer(pfi: core.FaultInjection, min_val: int = -1, max_val: int = 1):
    """
    每层单神经元注入 - 在单个batch元素中每层随机选择一个神经元注入错误
    
    参数:
        pfi: 故障注入对象
        min_val: 随机值下限
        max_val: 随机值上限
    
    返回:
        配置好的故障注入对象
    """
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    b = random_batch_element(pfi)  # 随机选择batch元素
    for i in range(pfi.get_total_layers()):  # 遍历所有层
        (layer, C, H, W) = random_neuron_location(pfi, layer=i)  # 在当前层随机选择神经元
        batch.append(b)
        layer_num.append(layer)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(random_value(min_val=min_val, max_val=max_val))

    return pfi.declare_neuron_fault_injection(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )

def random_inj_per_layer_batched(
    pfi: core.FaultInjection,
    min_val: int = -1,
    max_val: int = 1,
    rand_loc: bool = True,
    rand_val: bool = True
):
    """
    批量每层单神经元注入 - 在每个batch元素中每层随机选择一个神经元注入错误
    
    参数:
        pfi: 故障注入对象
        min_val: 随机值下限
        max_val: 随机值上限
        rand_loc: 是否每个batch使用随机位置
        rand_val: 是否每个batch使用随机值
    
    返回:
        配置好的故障注入对象
    """
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    for i in range(pfi.get_total_layers()):  # 遍历所有层
        if not rand_loc:
            (layer, C, H, W) = random_neuron_location(pfi, layer=i)
        if not rand_val:
            err_val = random_value(min_val=min_val, max_val=max_val)

        for b in range(pfi.batch_size):  # 遍历batch中所有元素
            if rand_loc:
                (layer, C, H, W) = random_neuron_location(pfi, layer=i)
            if rand_val:
                err_val = random_value(min_val=min_val, max_val=max_val)

            batch.append(b)
            layer_num.append(layer)
            c_rand.append(C)
            h_rand.append(H)
            w_rand.append(W)
            value.append(err_val)

    return pfi.declare_neuron_fault_injection(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )


class single_bit_flip_func(core.FaultInjection):
    def __init__(self, model, batch_size, input_shape=None, target_generate=None, **kwargs):
        if input_shape is None:
            input_shape = [3, 224, 224]
        super().__init__(model, batch_size, input_shape=input_shape, **kwargs)
        # logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")

        self.bits = kwargs.get("bits", 32) 
        self.layer_ranges = []
        # 配置日志格式
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")

        # 获取量化位数，默认为8位
        self.bits = kwargs.get("bits", 32) # 采用8位量化
        # 存储各层激活值的最大范围
        self.layer_ranges = []
        # 存储目标generate，默认为1
        self.target_generate = target_generate if target_generate is not None else [1] * batch_size
        # 记录是第几次generate
        self.current_generate = 0
        # 记录错误信息
        self.last_faults = []

    def set_conv_max(self, data):
        """设置激活值的最大范围"""
        self.layer_ranges = data

    def reset_conv_max(self, data):
        """重置激活值范围数据"""
        self.layer_ranges = []

    def get_conv_max(self, layer):
        """获取指定层的激活值最大范围"""
        return self.layer_ranges[layer]
    
    def reset_generate(self):
        """重置generate"""
        self.current_generate = 1
    
    def reset_faults(self):
        self.last_faults = []
    
    # 处理注入位置超边界
    def check_inj_oob(self, i, output):
        if output.ndim == 4:  # CNN 层 [N, C, H, W]
            if self.corrupt_dim[0][i] >= output.shape[1]:
                self.corrupt_dim[0][i] = random.randint(0, output.shape[1] - 1)
            if self.corrupt_dim[1][i] >= output.shape[2]:
                self.corrupt_dim[1][i] = random.randint(0, output.shape[2] - 1)
            if self.corrupt_dim[2][i] >= output.shape[3]:
                self.corrupt_dim[2][i] = random.randint(0, output.shape[3] - 1)            
        elif output.ndim == 3:  # Transformer Linear 层 [N, seq_len, hidden_dim] 
            if self.corrupt_dim[0][i] >= output.shape[1]:
                self.corrupt_dim[0][i] = random.randint(0, output.shape[1] - 1)
            if self.corrupt_dim[1][i] >= output.shape[2]:
                self.corrupt_dim[1][i] = random.randint(0, output.shape[2] - 1)
        elif output.ndim == 2:  # 特殊情况 [N, hidden_dim]
            if self.corrupt_dim[0][i] >= output.shape[1]:
                self.corrupt_dim[0][i] = random.randint(0, output.shape[1] - 1)
        if self.corrupt_batch[i] >= output.shape[0]:
            raise ValueError(f"Unsupported batch={self.corrupt_batch[i]} >= output.shape[0]={output.shape[0]}")
            
    @staticmethod
    def _twos_comp(val, bits):
        """计算补码表示"""
        # 如果最高位为1（负数）
        if (val & (1 << (bits - 1))) != 0:
            # 计算补码值
            val = val - (1 << bits)
        return val

    def _twos_comp_shifted(self, val, nbits):
        """处理带偏移的补码转换"""
        # 负值处理：加上偏移量
        return (1 << nbits) + val if val < 0 else self._twos_comp(val, nbits)
    
    # 针对qwen2和LLAMA2进行FP32翻转显存优化
    def _flip_bit_signed(self, orig_value, bit_pos):
        save_type = orig_value.dtype
        save_device = orig_value.device
        logging.info(f"Original Value: {orig_value}")

        if orig_value.numel() != 1:
            raise ValueError("只支持单个标量进行比特翻转")

        # === 新增：统一转成 fp32 进行比特操作 ===
        if save_type != torch.float32:
            orig_value_fp32 = orig_value.to(torch.float32)
        else:
            orig_value_fp32 = orig_value

        # 下面全部使用 fp32 + numpy 的方式（最可靠、最统一）
        np_val = orig_value_fp32.detach().cpu().numpy().astype(np.float32)
        raw_bits = np_val.view(np.uint32).item()

        if bit_pos < 0 or bit_pos >= 32:
            raise ValueError(f"bit_pos 必须在 [0, 31] 范围内，但收到 {bit_pos}")

        orig_bit = (raw_bits >> bit_pos) & 1
        new_bits = raw_bits ^ (1 << bit_pos)

        new_value_fp32 = np.array([new_bits], dtype=np.uint32).view(np.float32)[0]
        # === 结束统一处理 ===

        # 转回原始类型
        new_tensor = torch.tensor(new_value_fp32, dtype=save_type, device=save_device)

        flip_info = {
            "bit_pos": bit_pos,
            "from": int(orig_bit),
            "to": int(1 - orig_bit)
        }

        return new_tensor, flip_info
    
    def _flip_two_bits_signed(self, orig_value, bit_pos1, bit_pos2):
        save_type = orig_value.dtype
        save_device = orig_value.device
        logging.info(f"Original Value: {orig_value}")

        if orig_value.numel() != 1:
            raise ValueError("只支持单个标量进行双比特翻转")

        # === 新增：统一转成 fp32 进行比特操作 ===
        if save_type != torch.float32:
            orig_value_fp32 = orig_value.to(torch.float32)
        else:
            orig_value_fp32 = orig_value

        np_val = orig_value_fp32.detach().cpu().numpy().astype(np.float32)
        raw_bits = np_val.view(np.uint32).item()

        if bit_pos1 == bit_pos2:
            raise ValueError("bit_pos1 和 bit_pos2 不能相同")
        if not (0 <= bit_pos1 < 32 and 0 <= bit_pos2 < 32):
            raise ValueError("bit_pos1 和 bit_pos2 必须在 [0, 31] 范围内")

        orig_bit1 = (raw_bits >> bit_pos1) & 1
        orig_bit2 = (raw_bits >> bit_pos2) & 1
        new_bits = raw_bits ^ (1 << bit_pos1) ^ (1 << bit_pos2)

        new_value_fp32 = np.array([new_bits], dtype=np.uint32).view(np.float32)[0]
        # === 结束统一处理 ===

        new_tensor = torch.tensor(new_value_fp32, dtype=save_type, device=save_device)

        flip_info = [
            {"bit_pos": bit_pos1, "from": int(orig_bit1), "to": int(1 - orig_bit1)},
            {"bit_pos": bit_pos2, "from": int(orig_bit2), "to": int(1 - orig_bit2)},
        ]

        return new_tensor, flip_info
    
    # def _flip_bit_signed(self, orig_value, bit_pos):
    #     """
    #     执行有符号浮点数的单比特翻转（支持 fp16/fp32/fp64/bf16）
        
    #     参数:
    #     orig_value: 原始浮点数值（标量张量）
    #     bit_pos: 要翻转的比特位置（0 为最低位）
        
    #     返回:
    #     (新张量, 翻转信息字典)
    #     """
    #     save_type = orig_value.dtype
    #     save_device = orig_value.device
    #     logging.info(f"Original Value: {orig_value}")

    #     if orig_value.numel() != 1:
    #         raise ValueError("只支持单个标量进行比特翻转")

    #     if save_type in (torch.float16, torch.float32, torch.float64):
    #         # fp16/32/64 使用 NumPy 处理
    #         if save_type == torch.float16:
    #             np_float, np_uint, total_bits = np.float16, np.uint16, 16
    #         elif save_type == torch.float32:
    #             np_float, np_uint, total_bits = np.float32, np.uint32, 32
    #         else:  # float64
    #             np_float, np_uint, total_bits = np.float64, np.uint64, 64

    #         if bit_pos < 0 or bit_pos >= total_bits:
    #             raise ValueError(f"bit_pos 必须在 [0, {total_bits-1}] 范围内，但收到 {bit_pos}")

    #         np_val = orig_value.detach().cpu().numpy().astype(np_float)
    #         raw_bits = np_val.view(np_uint).item()

    #         orig_bit = (raw_bits >> bit_pos) & 1
    #         new_bits = raw_bits ^ (1 << bit_pos)

    #         new_value = np.array([new_bits], dtype=np_uint).view(np_float)[0]

    #         flip_info = {
    #             "bit_pos": bit_pos,
    #             "from": int(orig_bit),
    #             "to": int(1 - orig_bit)
    #         }

    #         return torch.tensor(new_value, dtype=save_type, device=save_device), flip_info

    #     elif save_type == torch.bfloat16:
    #         # bf16 使用纯 PyTorch 处理（兼容所有 NumPy 版本）
    #         total_bits = 16
    #         if bit_pos < 0 or bit_pos >= total_bits:
    #             raise ValueError(f"bit_pos 必须在 [0, 15] 范围内，但收到 {bit_pos}")

    #         # 直接 view 为 uint16 获取底层比特
    #         raw_bits = orig_value.view(torch.uint16).item()

    #         orig_bit = (raw_bits >> bit_pos) & 1
    #         new_bits = raw_bits ^ (1 << bit_pos)

    #         # 构造新的 bfloat16 值
    #         new_value = torch.tensor(new_bits, dtype=torch.uint16, device='cpu').view(torch.bfloat16).item()

    #         flip_info = {
    #             "bit_pos": bit_pos,
    #             "from": int(orig_bit),
    #             "to": int(1 - orig_bit)
    #         }

    #         return torch.tensor(new_value, dtype=torch.bfloat16, device=save_device), flip_info

    #     else:
    #         raise TypeError(f"_flip_bit_signed 只支持 float16/32/64/bfloat16，但收到 {save_type}")


    # def _flip_two_bits_signed(self, orig_value, bit_pos1, bit_pos2):
    #     """
    #     执行有符号浮点数的双比特翻转（支持 fp16/fp32/fp64/bf16）
        
    #     参数:
    #     orig_value: 原始浮点数值（标量张量）
    #     bit_pos1, bit_pos2: 要翻转的两个比特位置（0 为最低位）
        
    #     返回:
    #     (新张量, 翻转信息列表)
    #     """
    #     save_type = orig_value.dtype
    #     save_device = orig_value.device
    #     logging.info(f"Original Value: {orig_value}")

    #     if orig_value.numel() != 1:
    #         raise ValueError("只支持单个标量进行双比特翻转")

    #     if save_type in (torch.float16, torch.float32, torch.float64):
    #         # fp16/32/64 使用 NumPy 处理
    #         if save_type == torch.float16:
    #             np_float, np_uint, total_bits = np.float16, np.uint16, 16
    #         elif save_type == torch.float32:
    #             np_float, np_uint, total_bits = np.float32, np.uint32, 32
    #         else:  # float64
    #             np_float, np_uint, total_bits = np.float64, np.uint64, 64

    #         if (bit_pos1 < 0 or bit_pos1 >= total_bits or
    #             bit_pos2 < 0 or bit_pos2 >= total_bits):
    #             raise ValueError(f"bit_pos1 和 bit_pos2 必须在 [0, {total_bits-1}] 范围内")
    #         if bit_pos1 == bit_pos2:
    #             raise ValueError("bit_pos1 和 bit_pos2 不能相同")

    #         np_val = orig_value.detach().cpu().numpy().astype(np_float)
    #         raw_bits = np_val.view(np_uint).item()

    #         orig_bit1 = (raw_bits >> bit_pos1) & 1
    #         orig_bit2 = (raw_bits >> bit_pos2) & 1
    #         new_bits = raw_bits ^ (1 << bit_pos1) ^ (1 << bit_pos2)

    #         new_value = np.array([new_bits], dtype=np_uint).view(np_float)[0]

    #         flip_info = [
    #             {"bit_pos": bit_pos1, "from": int(orig_bit1), "to": int(1 - orig_bit1)},
    #             {"bit_pos": bit_pos2, "from": int(orig_bit2), "to": int(1 - orig_bit2)},
    #         ]

    #         return torch.tensor(new_value, dtype=save_type, device=save_device), flip_info

    #     elif save_type == torch.bfloat16:
    #         # bf16 使用纯 PyTorch 处理
    #         total_bits = 16
    #         if (bit_pos1 < 0 or bit_pos1 >= total_bits or
    #             bit_pos2 < 0 or bit_pos2 >= total_bits):
    #             raise ValueError(f"bit_pos1 和 bit_pos2 必须在 [0, 15] 范围内")
    #         if bit_pos1 == bit_pos2:
    #             raise ValueError("bit_pos1 和 bit_pos2 不能相同")

    #         raw_bits = orig_value.view(torch.uint16).item()

    #         orig_bit1 = (raw_bits >> bit_pos1) & 1
    #         orig_bit2 = (raw_bits >> bit_pos2) & 1
    #         new_bits = raw_bits ^ (1 << bit_pos1) ^ (1 << bit_pos2)

    #         new_value = torch.tensor(new_bits, dtype=torch.uint16, device='cpu').view(torch.bfloat16).item()

    #         flip_info = [
    #             {"bit_pos": bit_pos1, "from": int(orig_bit1), "to": int(1 - orig_bit1)},
    #             {"bit_pos": bit_pos2, "from": int(orig_bit2), "to": int(1 - orig_bit2)},
    #         ]

    #         return torch.tensor(new_value, dtype=torch.bfloat16, device=save_device), flip_info

    #     else:
    #         raise TypeError(f"_flip_two_bits_signed 只支持 float16/32/64/bfloat16，但收到 {save_type}")


    def single_bit_flip_signed_across_batch_svd(self, module, input_val, output):
        """
        在批量数据中执行单比特翻转操作
        
        参数:
        module: 当前处理的模块
        input_val: 模块输入值
        output: 模块输出值（将在此张量上执行注入）
        """
        id_tuple = random.randint(0, 1)
        if isinstance(output, tuple):
            # 随机0和1
            output = output[id_tuple]
            
        # 获取要注入的卷积层设置
        corrupt_conv_set = self.corrupt_layer
        
        # 这两个变量用于记录翻转前后的值，便于后续比较和调试
        # 注意：如果是多位置注入，这里需要根据实际情况调整
        prev_value = None
        new_value = None
        rand_bit = -1

        need_inject = False

        # 处理多位置注入情况
        if (type(corrupt_conv_set) is list) and len(corrupt_conv_set) > 1:
            # 筛选当前层需要注入的位置
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.current_layer,
                    range(len(corrupt_conv_set)),
                )
            )    
            if inj_list:
                need_inject = True

        else:
            if (type(corrupt_conv_set) is list):
                corrupt_conv_set = corrupt_conv_set[0]
            if (self.current_generate == self.target_generate[self.corrupt_batch[0]]) and (self.current_layer == corrupt_conv_set):
                need_inject = True

        if need_inject:
            original_shape = output.shape
            original_dtype = output.dtype
            D = original_shape[-1]
            A = output.view(-1, D).to(torch.float16)
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            r = min(output.shape[-1] // 16, S.size(0))  # 假设 self.r 已定义，否则可硬编码如 r = 128
            V = Vh.T
            Z = A @ V[:, :r]

        # 处理多位置注入情况
        if (type(corrupt_conv_set) is list) and len(corrupt_conv_set) > 1:
            for i in inj_list:
                if self.current_generate == self.target_generate[self.corrupt_batch[i]]: # 进行故障注入
                    self.check_inj_oob(i, output)
                    if self.current_generate == 1:
                        current_generate_seq_len = self.corrupt_dim[0][i]
                    else:
                        current_generate_seq_len = 0
                    if (output.shape[1] == 1) and (current_generate_seq_len != 1):
                        current_generate_seq_len = 0
                    if output.ndim == 4:  # CNN 层 [N, C, H, W]
                        token_idx = self.corrupt_batch[i] * (output.shape[1] * output.shape[2]) + current_generate_seq_len * output.shape[2] + self.corrupt_dim[1][i]
                        dim_idx = self.corrupt_dim[2][i]
                    elif output.ndim == 3:  # Transformer Linear 层 [N, seq_len, hidden_dim]
                        token_idx = self.corrupt_batch[i] * output.shape[1] + current_generate_seq_len
                        dim_idx = self.corrupt_dim[1][i]
                    elif output.ndim == 2:  # 特殊情况 [N, hidden_dim]
                        token_idx = self.corrupt_batch[i]
                        dim_idx = current_generate_seq_len
                    else:
                        raise ValueError(f"Unsupported output.ndim={output.ndim}")
                    r_idx = dim_idx % r
                    prev_value = Z[token_idx, r_idx].clone()
                    # 随机选择要翻转的比特位置
                    rand_bit = random.randint(0, self.bits - 1)
                    logging.info(f"Random Bit: {rand_bit}")
                    # 执行比特翻转
                    new_value, flip_info = self._flip_bit_signed(prev_value, rand_bit)
                    
                    # 写回，新值代替原始值
                    Z[token_idx, r_idx] = new_value
                    self.last_faults.append({
                    "batch": self.corrupt_batch[i],
                    "forword": self.current_generate,
                    "layer_id": self.current_layer,
                    "neuron_id": current_generate_seq_len,
                    "h_id": self.corrupt_dim[1][i],
                    "w_id": self.corrupt_dim[2][i],
                    "is_tuple": self.output_isTuple[self.current_layer],
                    "id_tuple": id_tuple,
                    "inj_type": "neuron",
                    "bit_position": flip_info,
                    "prev_value": prev_value.item() if torch.is_tensor(prev_value) else prev_value,
                    "faulty_value": new_value.item() if torch.is_tensor(new_value) else new_value,
                    "tensor_shape": tuple(output.shape),
                    "module_type": module.__class__.__name__,
                    })

        # 处理单位置注入情况
        else:
            if (self.current_generate == self.target_generate[self.corrupt_batch[0]]) and (self.current_layer == corrupt_conv_set):  # 进行故障注入
                self.check_inj_oob(0, output)
                if self.current_generate == 1:
                    current_generate_seq_len = self.corrupt_dim[0][0]
                else:
                    current_generate_seq_len = 0
                if (output.shape[1] == 1) and (current_generate_seq_len != 1):
                        current_generate_seq_len = 0
                if output.ndim == 4:  # CNN 层 [N, C, H, W]
                    token_idx = self.corrupt_batch[0] * (output.shape[1] * output.shape[2]) + current_generate_seq_len * output.shape[2] + self.corrupt_dim[1][0]
                    dim_idx = self.corrupt_dim[2][0]
                elif output.ndim == 3:  # Transformer Linear 层 [N, seq_len, hidden_dim]
                    token_idx = self.corrupt_batch[0] * output.shape[1] + current_generate_seq_len
                    dim_idx = self.corrupt_dim[1][0]
                elif output.ndim == 2:  # 特殊情况 [N, hidden_dim]
                    token_idx = self.corrupt_batch[0]
                    dim_idx = current_generate_seq_len
                else:
                    raise ValueError(f"Unsupported output.ndim={output.ndim}")

                r_idx = dim_idx % r
                prev_value = Z[token_idx, r_idx].clone()
                rand_bit = random.randint(0, self.bits - 1)
                logging.info(f"Random Bit: {rand_bit}")
                new_value, flip_info = self._flip_bit_signed(prev_value, rand_bit)
                # 写回
                Z[token_idx, r_idx] = new_value

                # 记录 fault 信息，方便外部访问
                self.last_faults.append({
                "batch": self.corrupt_batch[0],
                "forword": self.current_generate,
                "layer_id": self.current_layer,
                "neuron_id": current_generate_seq_len,
                "h_id": self.corrupt_dim[1][0],
                "w_id": self.corrupt_dim[2][0],
                "is_tuple": self.output_isTuple[self.current_layer],
                "id_tuple": id_tuple,
                "inj_type": "neuron",
                "bit_position": flip_info,
                "prev_value": prev_value.item() if torch.is_tensor(prev_value) else prev_value,
                "faulty_value": new_value.item() if torch.is_tensor(new_value) else new_value,
                "tensor_shape": tuple(output.shape),
                "module_type": module.__class__.__name__,
                })

        if need_inject:
            approx_A = Z @ Vh[:r, :]
            output.copy_(approx_A.view(original_shape).to(original_dtype))

        # 更新当前层索引
        self.update_layer()
        # 如果已处理完所有层则重置
        if self.current_layer >= len(self.output_size):
            self.reset_current_layer()
            self.current_generate += 1
    
    # 分析激活值
    def single_bit_flip_signed_across_batch_analysis(self, module, input_val, output):
        """
        在批量数据中执行单比特翻转操作
        
        参数:
        module: 当前处理的模块
        input_val: 模块输入值
        output: 模块输出值（将在此张量上执行注入）
        """
        id_tuple = random.randint(0, 1)
        if isinstance(output, tuple):
            # 随机0和1
            output = output[id_tuple]
            
        # # ==================== 关键：计算每个 neuron 的平均激活强度 ====================
        # import os
        # save_dir = "neuron_activation_magnitude"
        # os.makedirs(save_dir, exist_ok=True)
        
        # # 文件名：一个 generate_step 一个文件，里面存所有层的 d_ff 向量
        # save_path = os.path.join(save_dir, f"gen_{self.current_generate}.pt")
        
        # # 加载已有数据（支持多层追加）
        # if os.path.exists(save_path):
        #     layer_scores = torch.load(save_path)
        # else:
        #     layer_scores = {}

        # t = output.detach().float()  # [B, S, H] 或 [B, H]

        # if t.ndim == 3:
        #     # [B, S, H] → 先在 batch 和 seq 上取绝对值平均 → [H]
        #     neuron_score = t.abs().mean(dim=0).mean(dim=0)  # → [d_ff]
        # elif t.ndim == 2:
        #     # [B, H] → 只在 batch 上平均
        #     neuron_score = t.abs().mean(dim=0)  # → [d_ff]
        # else:
        #     return

        # # 存下来：layer_id → [d_ff] 的激活强度向量
        # layer_scores[self.current_layer] = neuron_score.cpu()

        # # 保存（每层都覆盖写一次，没问题）
        # torch.save(layer_scores, save_path)

        # # ==================== 可选：同时写一份可读的 txt（方便快速查看）================
        # txt_path = os.path.join(save_dir, f"gen_{self.current_generate}_readable.txt")
        # with open(txt_path, "a", encoding="utf-8") as f:
        #     topk_val, topk_idx = neuron_score.topk(10)
        #     bottomk_val, _ = neuron_score.topk(10, largest=False)
        #     f.write(f"\n[GEN {self.current_generate}] LAYER {self.current_layer:03d} | shape {t.shape}\n")
        #     f.write(f"  Mean activation magnitude: {neuron_score.mean():.6f}\n")
        #     f.write(f"  Top-10 most active neurons: " + 
        #             " ".join([f"d{i.item():4d}({v:.6f})" for v,i in zip(topk_val, topk_idx)]) + "\n")
        #     f.write(f"  Bottom-10 weakest neurons: " + 
        #             " ".join([f"d{i.item():4d}({v:.6f})" for v,i in zip(bottomk_val, torch.argsort(neuron_score)[:10])]) + "\n")
        # # ===========================================================================
        
        # # ==================== 修改后：仅在 generate=1 时保存 layer 4 的 batch 0 完整激活值 ====================
        # import os
        # save_dir = "neuron_activation_magnitude"
        # os.makedirs(save_dir, exist_ok=True)

        # # 只在 generate == 1 且当前是 layer 4 时保存
        # if self.current_generate == 1 and self.current_layer == 4:
        #     t = output.detach().float()  # [B, S, H] 或 [B, H]

        #     # 取出 batch 0 的完整激活（不去平均）
        #     if t.ndim == 3:  # [B, S, H]
        #         batch0_activation = t[0]  # → [S, H]
        #     elif t.ndim == 2:  # [B, H]
        #         batch0_activation = t[0]  # → [H]
        #     else:
        #         batch0_activation = None  # 不支持的维度直接跳过

        #     if batch0_activation is not None:
        #         # 保存路径：固定一个文件，只存 layer 4 的 batch 0 完整激活
        #         save_path = os.path.join(save_dir, "layer4_batch0_full_activation_gen1.pt")
                
        #         # 我们只存一个 tensor，直接覆盖保存（不需要多层 dict）
        #         torch.save(batch0_activation.cpu(), save_path)

        #         # 可选：同时写一份可读的统计信息（保留你原来的 txt 风格，便于查看）
        #         txt_path = os.path.join(save_dir, "layer4_batch0_full_activation_gen1_readable.txt")
        #         with open(txt_path, "w", encoding="utf-8") as f:  # 用 w 覆盖，避免重复写入
        #             neuron_abs = batch0_activation.abs()
        #             if neuron_abs.ndim == 2:
        #                 neuron_score = neuron_abs.mean(dim=0)  # 每个 neuron 在 seq 维上的平均绝对激活
        #             else:
        #                 neuron_score = neuron_abs  # 1D 情况直接就是

        #             topk_val, topk_idx = neuron_score.topk(10)
        #             bottomk_val, _ = neuron_score.topk(10, largest=False)

        #             f.write(f"[GEN 1] LAYER 004 | shape {t.shape} | batch 0 full activation saved\n")
        #             f.write(f"  Mean activation magnitude (per neuron, avg over seq): {neuron_score.mean():.6f}\n")
        #             f.write(f"  Top-10 most active neurons: " + 
        #                     " ".join([f"d{i.item():4d}({v:.6f})" for v,i in zip(topk_val, topk_idx)]) + "\n")
        #             f.write(f"  Bottom-10 weakest neurons: " + 
        #                     " ".join([f"d{i.item():4d}({v:.6f})" for v,i in zip(bottomk_val, torch.argsort(neuron_score)[:10])]) + "\n")

        # # =================================================================================
        
        
        # ==================== 关键：计算每个 batch 的激活向量的平均 L2 范数 ====================
        import os
        save_dir = "activation_l2_norms"
        os.makedirs(save_dir, exist_ok=True)
        
        # 文件名：一个 generate_step 一个文件，里面存所有层的 [B] 向量
        save_path = os.path.join(save_dir, f"gen_{self.current_generate}.pt")
        
        # 加载已有数据（支持多层追加）
        if os.path.exists(save_path):
            layer_norms = torch.load(save_path)
        else:
            layer_norms = {}

        t = output.detach().float()  # [B, S, H] 或 [B, H]

        if t.ndim == 3:
            # [B, S, H] → 先在 seq 上平均（压缩序列维度为1），再计算每个 batch 的 L2 范数 → [B]
            batch_norms = torch.norm(t.mean(dim=1), p=2, dim=-1)  # → [B]
        elif t.ndim == 2:
            # [B, H] → 直接计算每个 batch 的 L2 范数
            batch_norms = torch.norm(t, p=2, dim=-1)  # → [B]
        else:
            batch_norms = None

        if batch_norms is not None:
            # 存下来：layer_id → [B] 的 L2 范数向量
            layer_norms[self.current_layer] = batch_norms.cpu()

            # 保存（每层都覆盖写一次，没问题）
            torch.save(layer_norms, save_path)

        # ==================== 可选：同时写一份可读的 txt（方便快速查看）================
        txt_path = os.path.join(save_dir, f"gen_{self.current_generate}_readable.txt")
        with open(txt_path, "a", encoding="utf-8") as f:
            if batch_norms is not None:
                f.write(f"\n[GEN {self.current_generate}] LAYER {self.current_layer:03d} | shape {t.shape}\n")
                f.write(f"  Per-batch L2 norms: " + " ".join([f"batch{i}: {v:.6f}" for i, v in enumerate(batch_norms)]) + "\n")
                f.write(f"  Mean L2 norm across batches: {batch_norms.mean():.6f}\n")
        # ===========================================================================
        
        # 获取要注入的卷积层设置
        corrupt_conv_set = self.corrupt_layer
        
        # 这两个变量用于记录翻转前后的值，便于后续比较和调试
        # 注意：如果是多位置注入，这里需要根据实际情况调整
        prev_value = None
        new_value = None
        rand_bit = -1

        # 处理多位置注入情况
        if (type(corrupt_conv_set) is list) and len(corrupt_conv_set) > 1:
            # 筛选当前层需要注入的位置
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.current_layer,
                    range(len(corrupt_conv_set)),
                )
            )    
            for i in inj_list:
                if self.current_generate == self.target_generate[self.corrupt_batch[i]]: # 进行故障注入
                    self.check_inj_oob(i, output)
                    if self.current_generate == 1:
                        current_generate_seq_len = self.corrupt_dim[0][i]
                    else:
                        current_generate_seq_len = 0
                    if (output.shape[1] == 1) and (current_generate_seq_len != 1):
                        current_generate_seq_len = 0
                    if output.ndim == 4:  # CNN 层 [N, C, H, W]
                        prev_value = output[self.corrupt_batch[i],current_generate_seq_len,self.corrupt_dim[1][i],self.corrupt_dim[2][i]].clone()
                    elif output.ndim == 3:  # Transformer Linear 层 [N, seq_len, hidden_dim]
                        prev_value = output[self.corrupt_batch[i],current_generate_seq_len,self.corrupt_dim[1][i]].clone()
                    elif output.ndim == 2:  # 特殊情况 [N, hidden_dim]
                        prev_value = output[self.corrupt_batch[i],current_generate_seq_len].clone()
                    else:
                        raise ValueError(f"Unsupported output.ndim={output.ndim}")
                    # 随机选择要翻转的比特位置
                    rand_bit = random.randint(0, self.bits - 1)
                    logging.info(f"Random Bit: {rand_bit}")
                    # 执行比特翻转
                    new_value, flip_info = self._flip_bit_signed(prev_value, rand_bit)
                    
                    # 写回，新值代替原始值
                    if output.ndim == 4:
                        output[self.corrupt_batch[i],current_generate_seq_len,self.corrupt_dim[1][i],self.corrupt_dim[2][i]] = new_value
                    elif output.ndim == 3:
                        output[self.corrupt_batch[i],current_generate_seq_len,self.corrupt_dim[1][i]] = new_value
                    elif output.ndim == 2:
                        output[self.corrupt_batch[i],current_generate_seq_len] = new_value
                    self.last_faults.append({
                    "batch": self.corrupt_batch[i],
                    "forword": self.current_generate,
                    "layer_id": self.current_layer,
                    "neuron_id": current_generate_seq_len,
                    "h_id": self.corrupt_dim[1][i],
                    "w_id": self.corrupt_dim[2][i],
                    "is_tuple": self.output_isTuple[self.current_layer],
                    "id_tuple": id_tuple,
                    "inj_type": "neuron",
                    "bit_position": flip_info,
                    "prev_value": prev_value.item() if torch.is_tensor(prev_value) else prev_value,
                    "faulty_value": new_value.item() if torch.is_tensor(new_value) else new_value,
                    "tensor_shape": tuple(output.shape),
                    "module_type": module.__class__.__name__,
                    })

        # 处理单位置注入情况
        else:
            if (type(corrupt_conv_set) is list):
                corrupt_conv_set = corrupt_conv_set[0]
            if (self.current_generate == self.target_generate[self.corrupt_batch[0]]) and (self.current_layer == corrupt_conv_set):  # 进行故障注入
                self.check_inj_oob(0, output)
                if self.current_generate == 1:
                    current_generate_seq_len = self.corrupt_dim[0][0]
                else:
                    current_generate_seq_len = 0
                if (output.shape[1] == 1) and (current_generate_seq_len != 1):
                        current_generate_seq_len = 0
                if output.ndim == 4:  # CNN 层 [N, C, H, W]
                    prev_value = output[self.corrupt_batch[0],current_generate_seq_len,self.corrupt_dim[1][0],self.corrupt_dim[2][0]].clone()
                elif output.ndim == 3:  # Transformer Linear 层 [N, seq_len, hidden_dim]
                    prev_value = output[self.corrupt_batch[0],current_generate_seq_len,self.corrupt_dim[1][0]].clone()
                elif output.ndim == 2:  # 特殊情况 [N, hidden_dim]
                    prev_value = output[self.corrupt_batch[0],current_generate_seq_len].clone()
                else:
                    raise ValueError(f"Unsupported output.ndim={output.ndim}")

                rand_bit = random.randint(0, self.bits - 1)
                logging.info(f"Random Bit: {rand_bit}")
                new_value, flip_info = self._flip_bit_signed(prev_value, rand_bit)
                # 写回
                if output.ndim == 4:
                    output[self.corrupt_batch[0],current_generate_seq_len,self.corrupt_dim[1][0],self.corrupt_dim[2][0]] = new_value
                elif output.ndim == 3:
                    output[self.corrupt_batch[0],current_generate_seq_len,self.corrupt_dim[1][0]] = new_value
                elif output.ndim == 2:
                    output[self.corrupt_batch[0],current_generate_seq_len] = new_value

                # 记录 fault 信息，方便外部访问
                self.last_faults.append({
                "batch": self.corrupt_batch[0],
                "forword": self.current_generate,
                "layer_id": self.current_layer,
                "neuron_id": current_generate_seq_len,
                "h_id": self.corrupt_dim[1][0],
                "w_id": self.corrupt_dim[2][0],
                "is_tuple": self.output_isTuple[self.current_layer],
                "id_tuple": id_tuple,
                "inj_type": "neuron",
                "bit_position": flip_info,
                "prev_value": prev_value.item() if torch.is_tensor(prev_value) else prev_value,
                "faulty_value": new_value.item() if torch.is_tensor(new_value) else new_value,
                "tensor_shape": tuple(output.shape),
                "module_type": module.__class__.__name__,
                })
        # 更新当前层索引
        self.update_layer()
        # 如果已处理完所有层则重置
        if self.current_layer >= len(self.output_size):
            # 计算所有 transformer block (层) 的平均 L2 范数（per batch）
            if os.path.exists(save_path):
                layer_norms = torch.load(save_path)
                if layer_norms:
                    all_norms = torch.stack(list(layer_norms.values()))  # [num_layers, B]
                    avg_norms = all_norms.mean(dim=0)  # [B]
                    # 追加到 txt：平均统计
                    with open(txt_path, "a", encoding="utf-8") as f:
                        f.write(f"\n[GEN {self.current_generate}] AVERAGE ACROSS ALL LAYERS\n")
                        f.write(f"  Per-batch average L2 norms: " + " ".join([f"batch{i}: {v:.6f}" for i, v in enumerate(avg_norms)]) + "\n")
                        f.write(f"  Mean average L2 norm across batches: {avg_norms.mean():.6f}\n")
            self.reset_current_layer()
            self.current_generate += 1
    
    def single_bit_flip_signed_across_batch(self, module, input_val, output):
        """
        在批量数据中执行单比特翻转操作
        
        参数:
        module: 当前处理的模块
        input_val: 模块输入值
        output: 模块输出值（将在此张量上执行注入）
        """
        id_tuple = random.randint(0, 1)
        if isinstance(output, tuple):
            # 随机0和1
            output = output[id_tuple]
        
        # 获取要注入的卷积层设置
        corrupt_conv_set = self.corrupt_layer
        
        # 这两个变量用于记录翻转前后的值，便于后续比较和调试
        # 注意：如果是多位置注入，这里需要根据实际情况调整
        prev_value = None
        new_value = None
        rand_bit = -1

        # 处理多位置注入情况
        if (type(corrupt_conv_set) is list) and len(corrupt_conv_set) > 1:
            # 筛选当前层需要注入的位置
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.current_layer,
                    range(len(corrupt_conv_set)),
                )
            )    
            for i in inj_list:
                if self.current_generate == self.target_generate[self.corrupt_batch[i]]: # 进行故障注入
                    self.check_inj_oob(i, output)
                    if self.current_generate == 1:
                        current_generate_seq_len = self.corrupt_dim[0][i]
                    else:
                        current_generate_seq_len = 0
                    if (output.shape[1] == 1) and (current_generate_seq_len != 1):
                        current_generate_seq_len = 0
                    if output.ndim == 4:  # CNN 层 [N, C, H, W]
                        prev_value = output[self.corrupt_batch[i],current_generate_seq_len,self.corrupt_dim[1][i],self.corrupt_dim[2][i]].clone()
                    elif output.ndim == 3:  # Transformer Linear 层 [N, seq_len, hidden_dim]
                        prev_value = output[self.corrupt_batch[i],current_generate_seq_len,self.corrupt_dim[1][i]].clone()
                    elif output.ndim == 2:  # 特殊情况 [N, hidden_dim]
                        prev_value = output[self.corrupt_batch[i],current_generate_seq_len].clone()
                    else:
                        raise ValueError(f"Unsupported output.ndim={output.ndim}")
                    # 随机选择要翻转的比特位置
                    rand_bit = random.randint(0, self.bits - 1)
                    logging.info(f"Random Bit: {rand_bit}")
                    # 执行比特翻转
                    new_value, flip_info = self._flip_bit_signed(prev_value, rand_bit)
                    
                    # 写回，新值代替原始值
                    if output.ndim == 4:
                        output[self.corrupt_batch[i],current_generate_seq_len,self.corrupt_dim[1][i],self.corrupt_dim[2][i]] = new_value
                    elif output.ndim == 3:
                        output[self.corrupt_batch[i],current_generate_seq_len,self.corrupt_dim[1][i]] = new_value
                    elif output.ndim == 2:
                        output[self.corrupt_batch[i],current_generate_seq_len] = new_value
                    self.last_faults.append({
                    "batch": self.corrupt_batch[i],
                    "forword": self.current_generate,
                    "layer_id": self.current_layer,
                    "neuron_id": current_generate_seq_len,
                    "h_id": self.corrupt_dim[1][i],
                    "w_id": self.corrupt_dim[2][i],
                    "is_tuple": self.output_isTuple[self.current_layer],
                    "id_tuple": id_tuple,
                    "inj_type": "neuron",
                    "bit_position": flip_info,
                    "prev_value": prev_value.item() if torch.is_tensor(prev_value) else prev_value,
                    "faulty_value": new_value.item() if torch.is_tensor(new_value) else new_value,
                    "tensor_shape": tuple(output.shape),
                    "module_type": module.__class__.__name__,
                    })

        # 处理单位置注入情况
        else:
            if (type(corrupt_conv_set) is list):
                corrupt_conv_set = corrupt_conv_set[0]
            if (self.current_generate == self.target_generate[self.corrupt_batch[0]]) and (self.current_layer == corrupt_conv_set):  # 进行故障注入
                self.check_inj_oob(0, output)
                if self.current_generate == 1:
                    current_generate_seq_len = self.corrupt_dim[0][0]
                else:
                    current_generate_seq_len = 0
                if (output.shape[1] == 1) and (current_generate_seq_len != 1):
                        current_generate_seq_len = 0
                if output.ndim == 4:  # CNN 层 [N, C, H, W]
                    prev_value = output[self.corrupt_batch[0],current_generate_seq_len,self.corrupt_dim[1][0],self.corrupt_dim[2][0]].clone()
                elif output.ndim == 3:  # Transformer Linear 层 [N, seq_len, hidden_dim]
                    prev_value = output[self.corrupt_batch[0],current_generate_seq_len,self.corrupt_dim[1][0]].clone()
                elif output.ndim == 2:  # 特殊情况 [N, hidden_dim]
                    prev_value = output[self.corrupt_batch[0],current_generate_seq_len].clone()
                else:
                    raise ValueError(f"Unsupported output.ndim={output.ndim}")

                rand_bit = random.randint(0, self.bits - 1)
                logging.info(f"Random Bit: {rand_bit}")
                new_value, flip_info = self._flip_bit_signed(prev_value, rand_bit)
                # 写回
                if output.ndim == 4:
                    output[self.corrupt_batch[0],current_generate_seq_len,self.corrupt_dim[1][0],self.corrupt_dim[2][0]] = new_value
                elif output.ndim == 3:
                    output[self.corrupt_batch[0],current_generate_seq_len,self.corrupt_dim[1][0]] = new_value
                elif output.ndim == 2:
                    output[self.corrupt_batch[0],current_generate_seq_len] = new_value

                # 记录 fault 信息，方便外部访问
                self.last_faults.append({
                "batch": self.corrupt_batch[0],
                "forword": self.current_generate,
                "layer_id": self.current_layer,
                "neuron_id": current_generate_seq_len,
                "h_id": self.corrupt_dim[1][0],
                "w_id": self.corrupt_dim[2][0],
                "is_tuple": self.output_isTuple[self.current_layer],
                "id_tuple": id_tuple,
                "inj_type": "neuron",
                "bit_position": flip_info,
                "prev_value": prev_value.item() if torch.is_tensor(prev_value) else prev_value,
                "faulty_value": new_value.item() if torch.is_tensor(new_value) else new_value,
                "tensor_shape": tuple(output.shape),
                "module_type": module.__class__.__name__,
                })
        # 更新当前层索引
        self.update_layer()
        # 如果已处理完所有层则重置
        if self.current_layer >= len(self.output_size):
            self.reset_current_layer()
            self.current_generate += 1
        
    def two_bit_flip_signed_across_batch(self, module, input_val, output):
        """
        在批量数据中执行单比特翻转操作
        
        参数:
        module: 当前处理的模块
        input_val: 模块输入值
        output: 模块输出值（将在此张量上执行注入）
        """
        id_tuple = random.randint(0, 1)
        if isinstance(output, tuple):
            # 随机0和1
            output = output[id_tuple]
        # 获取要注入的卷积层设置
        corrupt_conv_set = self.corrupt_layer
        
        # 不需要设置范围
        # 获取当前层的激活值最大范围
        # range_max = self.get_conv_max(self.current_layer)
        # logging.info(f"Current layer: {self.current_layer}")
        # logging.info(f"Range_max: {range_max}")
        
        # 这两个变量用于记录翻转前后的值，便于后续比较和调试
        # 注意：如果是多位置注入，这里需要根据实际情况调整
        prev_value = None
        new_value = None
        rand_bit = -1
        rand_bit2 = -1
        
        # 处理多位置注入情况
        if (type(corrupt_conv_set) is list) and len(corrupt_conv_set) > 1:
            # 筛选当前层需要注入的位置
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.current_layer,
                    range(len(corrupt_conv_set)),
                )
            )
            for i in inj_list:
                if self.current_generate == self.target_generate[self.corrupt_batch[i]]: # 进行故障注入
                    self.check_inj_oob(i, output)
                    if self.current_generate == 1:
                        current_generate_seq_len = self.corrupt_dim[0][i]
                    else:
                        current_generate_seq_len = 0
                    if (output.shape[1] == 1) and (current_generate_seq_len != 1):
                        current_generate_seq_len = 0
                    if output.ndim == 4:  # CNN 层 [N, C, H, W]
                        prev_value = output[self.corrupt_batch[i],current_generate_seq_len,self.corrupt_dim[1][i],self.corrupt_dim[2][i]].clone()
                    elif output.ndim == 3:  # Transformer Linear 层 [N, seq_len, hidden_dim]
                        prev_value = output[self.corrupt_batch[i],current_generate_seq_len,self.corrupt_dim[1][i]].clone()
                    elif output.ndim == 2:  # 特殊情况 [N, hidden_dim]
                        prev_value = output[self.corrupt_batch[i],current_generate_seq_len].clone()
                    else:
                        raise ValueError(f"Unsupported output.ndim={output.ndim}")
                    # 随机选择要翻转的比特位置
                    rand_bit = random.randint(0, self.bits - 1)
                    possible_bits = [i for i in range(self.bits) if i != rand_bit]
                    rand_bit2 = random.choice(possible_bits)
                    logging.info(f"Random Bit: {rand_bit}, {rand_bit2}")
                    # 执行比特翻转
                    new_value, flip_info = self._flip_two_bits_signed(prev_value, rand_bit, rand_bit2)     
                    # 写回，新值代替原始值
                    if output.ndim == 4:
                        output[self.corrupt_batch[i],current_generate_seq_len,self.corrupt_dim[1][i],self.corrupt_dim[2][i]] = new_value
                    elif output.ndim == 3:
                        output[self.corrupt_batch[i],current_generate_seq_len,self.corrupt_dim[1][i]] = new_value
                    elif output.ndim == 2:
                        output[self.corrupt_batch[i],current_generate_seq_len] = new_value
                    # 记录 fault 信息，方便外部访问
                    self.last_faults.append({
                    "batch": self.corrupt_batch[i],
                    "forword": self.current_generate,
                    "layer_id": self.current_layer,
                    "neuron_id": current_generate_seq_len,
                    "h_id": self.corrupt_dim[1][i],
                    "w_id": self.corrupt_dim[2][i],
                    "is_tuple": self.output_isTuple[self.current_layer],
                    "id_tuple": id_tuple,
                    "inj_type": "neuron",
                    "bit_position": flip_info,
                    "prev_value": prev_value.item() if torch.is_tensor(prev_value) else prev_value,
                    "faulty_value": new_value.item() if torch.is_tensor(new_value) else new_value,
                    "tensor_shape": tuple(output.shape),
                    "module_type": module.__class__.__name__,
                    })

        # 处理单位置注入情况
        else:
            if (type(corrupt_conv_set) is list):
                corrupt_conv_set = corrupt_conv_set[0]
            if (self.current_generate == self.target_generate[self.corrupt_batch[0]]) and (self.current_layer == corrupt_conv_set):  # 进行故障注入
                self.check_inj_oob(0, output)
                if self.current_generate == 1:
                    current_generate_seq_len = self.corrupt_dim[0][0]
                else:
                    current_generate_seq_len = 0
                if (output.shape[1] == 1) and (current_generate_seq_len != 1):
                        current_generate_seq_len = 0
                if output.ndim == 4:  # CNN 层 [N, C, H, W]
                    prev_value = output[self.corrupt_batch[0],current_generate_seq_len,self.corrupt_dim[1][0],self.corrupt_dim[2][0]].clone()
                elif output.ndim == 3:  # Transformer Linear 层 [N, seq_len, hidden_dim]
                    prev_value = output[self.corrupt_batch[0],current_generate_seq_len,self.corrupt_dim[1][0]].clone()
                elif output.ndim == 2:  # 特殊情况 [N, hidden_dim]
                    prev_value = output[self.corrupt_batch[0],current_generate_seq_len].clone()
                else:
                    raise ValueError(f"Unsupported output.ndim={output.ndim}")
                rand_bit = random.randint(0, self.bits - 1)
                possible_bits = [i for i in range(self.bits) if i != rand_bit]
                rand_bit2 = random.choice(possible_bits)
                logging.info(f"Random Bit: {rand_bit}, {rand_bit2}")
                new_value, flip_info = self._flip_two_bits_signed(prev_value, rand_bit, rand_bit2)
                # 写回
                if output.ndim == 4:
                    output[self.corrupt_batch[0],current_generate_seq_len,self.corrupt_dim[1][0],self.corrupt_dim[2][0]] = new_value
                elif output.ndim == 3:
                    output[self.corrupt_batch[0],current_generate_seq_len,self.corrupt_dim[1][0]] = new_value
                elif output.ndim == 2:
                    output[self.corrupt_batch[0],current_generate_seq_len] = new_value
                # 记录 fault 信息，方便外部访问
                self.last_faults.append({
                "batch": self.corrupt_batch[0],
                "forword": self.current_generate,
                "layer_id": self.current_layer,
                "neuron_id": current_generate_seq_len,
                "h_id": self.corrupt_dim[1][0],
                "w_id": self.corrupt_dim[2][0],
                "is_tuple": self.output_isTuple[self.current_layer],
                "id_tuple": id_tuple,
                "inj_type": "neuron",
                "bit_position": flip_info,
                "prev_value": prev_value.item() if torch.is_tensor(prev_value) else prev_value,
                "faulty_value": new_value.item() if torch.is_tensor(new_value) else new_value,
                "tensor_shape": tuple(output.shape),
                "module_type": module.__class__.__name__,
                })
        # 更新当前层索引
        self.update_layer()
        # 如果已处理完所有层则重置
        if self.current_layer >= len(self.output_size):
            self.reset_current_layer()
            self.current_generate += 1

def random_neuron_single_bit_inj_batched(
    pfi: core.FaultInjection, batch_random=True
):
    """
    为批量中的每个样本随机选择神经元位置进行单比特翻转
    
    参数:
    pfi: 故障注入器实例
    batch_random: 是否每个批次使用不同的随机位置（默认True）
    
    返回:
    配置好的故障注入声明
    """
    # # 设置各层激活值范围
    # pfi.set_conv_max(layer_ranges)

    # 根据batch_random参数生成位置列表
    locations = (
        [random_neuron_location(pfi) for _ in range(pfi.batch_size)]
        if batch_random
        else [random_neuron_location(pfi)] * pfi.batch_size
    )
    # 将位置元组列表转换为维度列表
    random_layers, random_c, random_h, random_w = map(list, zip(*locations))
    # random_layers[0] = 86
    # random_c[0] = 6
    # random_h[0] = 50

    # 声明神经元故障注入
    return pfi.declare_neuron_fault_injection(
        batch=range(pfi.batch_size),
        layer_num=random_layers,
        dim1=random_c,
        dim2=random_h,
        dim3=random_w,
        function=pfi.single_bit_flip_signed_across_batch,
    )


def random_neuron_two_bit_inj_batched(pfi, batch_random=True):
    locations = (
        [random_neuron_location(pfi) for _ in range(pfi.batch_size)]
        if batch_random
        else [random_neuron_location(pfi)] * pfi.batch_size
    )
    random_layers, random_c, random_h, random_w = map(list, zip(*locations))

    return pfi.declare_neuron_fault_injection(
        batch=list(range(pfi.batch_size)),
        layer_num=random_layers,
        dim1=random_c,
        dim2=random_h,
        dim3=random_w,
        function=pfi.two_bit_flip_signed_across_batch,
    )


def random_neuron_single_bit_inj(pfi: core.FaultInjection):
    """
    随机选择单个神经元位置进行单比特翻转
    
    参数:
    pfi: 故障注入器实例
    
    返回:
    配置好的故障注入声明
    """
    # # TODO 支持通过列表实现多种错误模型
    # pfi.set_conv_max(layer_ranges)

    # 随机选择批次索引和神经元位置
    batch = random_batch_element(pfi)
    (layer, C, H, W) = random_neuron_location(pfi)
    
    # 把随机选中的位置存起来
    pfi._last_rand_loc = (batch, layer, C, H, W)

    # 声明单个位置的故障注入
    return pfi.declare_neuron_fault_injection(
        batch=[batch],
        layer_num=[layer],
        dim1=[C],
        dim2=[H],
        dim3=[W],
        function=pfi.single_bit_flip_signed_across_batch,
    )

def random_neuron_two_bit_inj(pfi: core.FaultInjection):
    """
    随机选择单个神经元位置进行双比特翻转
    
    参数:
    pfi: 故障注入器实例
    
    返回:
    配置好的故障注入声明
    """
    # # TODO 支持通过列表实现多种错误模型
    # pfi.set_conv_max(layer_ranges)

    # 随机选择批次索引和神经元位置
    batch = random_batch_element(pfi)
    (layer, C, H, W) = random_neuron_location(pfi)
    
    # 把随机选中的位置存起来
    pfi._last_rand_loc = (batch, layer, C, H, W)

    # 声明单个位置的故障注入
    return pfi.declare_neuron_fault_injection(
        batch=[batch],
        layer_num=[layer],
        dim1=[C],
        dim2=[H],
        dim3=[W],
        function=pfi.two_bit_flip_signed_across_batch,
    )
