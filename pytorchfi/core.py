"""pytorchfi.core contains the core functionality for fault injections"""
# pytorchfi.core 包含故障注入的核心功能

import copy
import logging
import warnings
from typing import List

import torch
import torch.nn as nn


class FaultInjection:
    def __init__(
        self,
        model,
        batch_size: int,
        input_shape: List[int] = None,
        layer_types=None,
        **kwargs,
    ):
        # 初始化函数，接收模型、批大小、输入形状、层类型等参数
        if not input_shape:
            input_shape = [3, 224, 224]  # 默认输入形状为3通道224x224图像
        if not layer_types:
            layer_types = [nn.Conv2d]  # 默认注入层类型为卷积层

        # 配置日志格式，包含时间、客户端IP、用户和消息
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")

        self.original_model = model  # 保存原始模型
        self.output_isTuple = []  # 标记输出是否为元组
        self.output_size = []  # 存储每层输出的尺寸
        self.layers_type = []  # 存储每层的类型
        self.layers_dim = []  # 存储每层输出的维度数
        self.weights_size = []  # 存储每层权重的尺寸
        self.batch_size = batch_size  # 批大小

        self._input_shape = input_shape  # 输入形状
        self._inj_layer_types = layer_types  # 注入的层类型列表

        self.corrupted_model = None  # 注入故障后的模型
        self.current_layer = 0  # 当前处理的层索引
        self.handles = []  # 钩子句柄列表
        self.corrupt_batch = []  # 需要注入的批次索引列表
        self.corrupt_layer = []  # 需要注入的层索引列表
        self.corrupt_dim = [[], [], []]  # 需要注入的维度索引列表，分别对应C, H, W
        self.corrupt_value = []  # 注入的值列表

        # 是否使用CUDA，默认为模型参数所在设备是否为CUDA
        self.use_cuda = kwargs.get("use_cuda", next(model.parameters()).is_cuda)

        # 参数检查，确保输入形状为列表，批大小为大于等于1的整数，层类型列表非空
        if not isinstance(input_shape, list):
            raise AssertionError("Error: Input shape must be provided as a list.")
        if not (isinstance(batch_size, int) and batch_size >= 1):
            raise AssertionError("Error: Batch size must be an integer greater than 1.")
        if len(layer_types) < 0:
            raise AssertionError("Error: At least one layer type must be selected.")

        # 遍历模型，设置钩子，获取权重尺寸等信息
        handles = self._traverse_model_set_hooks(
            self.original_model, self._inj_layer_types
        )

        # 构造一个dummy输入张量，用于前向传播以获取层输出尺寸
        dummy_shape = (1, *self._input_shape)  # 只用一个batch元素进行profile
        model_dtype = next(model.parameters()).dtype  # 获取模型参数的数据类型
        device = "cuda" if self.use_cuda else None  # 设备选择
        # _dummy_tensor = torch.randint(low=0, high=model.config.vocab_size, size=dummy_shape, dtype=model_dtype, device=device)
        _dummy_tensor = torch.randint(low=0, high=model.config.vocab_size, size=dummy_shape, dtype=torch.long, device=device)
        # 运行一次前向传播，触发钩子，保存输出尺寸信息
        self.original_model(_dummy_tensor)

        # 移除之前注册的钩子，避免影响后续操作
        for index, _handle in enumerate(handles):
            handles[index].remove()
        handles = []

        # 记录输入形状和模型层尺寸信息
        logging.info("Input shape:")
        logging.info(dummy_shape[1:])

        logging.info("Model layer sizes:")
        logging.info(
            "\n".join(
                [
                    "".join(["{:4}".format(item) for item in row])
                    for row in self.output_size
                ]
            )
        )

    def reset_fault_injection(self):
        # 重置故障注入状态，清空注入相关信息
        self._reset_fault_injection_state()
        self.corrupted_model = None
        logging.info("Fault injector reset.")

    def _reset_fault_injection_state(self):
        # 内部函数，重置注入状态变量，移除所有钩子
        (
            self.current_layer,
            self.corrupt_batch,
            self.corrupt_layer,
            self.corrupt_dim,
            self.corrupt_value,
        ) = (0, [], [], [[], [], []], [])

        for index, _handle in enumerate(self.handles):
            self.handles[index].remove()
        self.handles = []

    def _traverse_model_set_hooks(self, model, layer_types):
        # 遍历模型所有子模块，针对指定层类型注册钩子，收集输出尺寸和权重尺寸信息
        handles = [] 
        for layer in model.children():
            # 叶子节点（无子模块）
            registered = False
            if list(layer.children()) == []:
                if "all" in layer_types:
                    # 如果指定注入所有层，注册钩子
                    handles.append(layer.register_forward_hook(self._save_output_size))
                    registered = True
                else:
                    for i in layer_types:
                        if isinstance(layer, i):
                            # 只对指定类型层注册钩子
                            handles.append(layer.register_forward_hook(self._save_output_size))
                            registered = True
                            break # 如果层匹配多个类型，避免多次注册。
            # 非叶子节点，递归遍历子模块
            else:
                subhandles = self._traverse_model_set_hooks(layer, layer_types)
                handles.extend(subhandles)
        return handles

    def _traverse_model_set_hooks_neurons(self, model, layer_types, customInj, injFunc):
        # 遍历模型，针对指定层类型注册神经元注入钩子，支持自定义注入函数
        handles = []
        for layer in model.children():
            # 叶子节点
            if list(layer.children()) == []:
                if "all" in layer_types:
                    # 注册自定义函数或默认_set_value钩子
                    hook = injFunc if customInj else self._set_value
                    handles.append(layer.register_forward_hook(hook))
                else:
                    for i in layer_types:
                        if isinstance(layer, i):
                            hook = injFunc if customInj else self._set_value
                            handles.append(layer.register_forward_hook(hook))
                            break
            # 非叶子节点，递归遍历
            else:
                subHandles = self._traverse_model_set_hooks_neurons(
                    layer, layer_types, customInj, injFunc
                )
                handles.extend(subHandles)
        return handles

    def declare_weight_fault_injection(self, **kwargs):
        # 声明权重故障注入，支持自定义注入函数或指定注入位置和值
        self._reset_fault_injection_state()
        custom_injection = False
        custom_function = False

        if kwargs:
            if "function" in kwargs:
                # 使用自定义注入函数
                custom_injection, custom_function = True, kwargs.get("function")
                corrupt_layer = kwargs.get("layer_num", [])
                corrupt_k = kwargs.get("k", [])
                corrupt_c = kwargs.get("dim1", [])
                corrupt_kH = kwargs.get("dim2", [])
                corrupt_kW = kwargs.get("dim3", [])
            else:
                # 指定注入位置和值
                corrupt_layer = kwargs.get(
                    "layer_num",
                )
                corrupt_k = kwargs.get("k", [])
                corrupt_c = kwargs.get("dim1", [])
                corrupt_kH = kwargs.get("dim2", [])
                corrupt_kW = kwargs.get("dim3", [])
                corrupt_value = kwargs.get("value", [])
        else:
            raise ValueError("Please specify an injection or injection function")

        # TODO: 这里可以添加边界检查

        # 复制原始模型，避免破坏原模型
        self.corrupted_model = copy.deepcopy(self.original_model)

        current_weight_layer = 0
        for layer in self.corrupted_model.modules():
            # 只对指定层类型进行注入
            if isinstance(layer, tuple(self._inj_layer_types)):
                # 找出所有需要注入当前层的索引
                inj_list = list(
                    filter(
                        lambda x: corrupt_layer[x] == current_weight_layer,
                        range(len(corrupt_layer)),
                    )
                )

                for inj in inj_list:
                    # 构造权重索引元组
                    corrupt_idx = tuple(
                        [
                            corrupt_k[inj],
                            corrupt_c[inj],
                            corrupt_kH[inj],
                            corrupt_kW[inj],
                        ]
                    )
                    orig_value = layer.weight[corrupt_idx].item()  # 保存原始权重值

                    with torch.no_grad():
                        if custom_injection:
                            # 使用自定义函数计算注入值
                            corrupt_value = custom_function(layer.weight, corrupt_idx)
                            layer.weight[corrupt_idx] = corrupt_value
                        else:
                            # 直接赋值
                            layer.weight[corrupt_idx] = corrupt_value[inj]

                    # 记录注入信息
                    # logging.info("Weight Injection")
                    # logging.info(f"Layer index: {corrupt_layer}")
                    # logging.info(f"Module: {layer}")
                    # logging.info(f"Original value: {orig_value}")
                    # logging.info(f"Injected value: {layer.weight[corrupt_idx]}")
                current_weight_layer += 1
        return self.corrupted_model

    def declare_neuron_fault_injection(self, **kwargs):
        # 声明神经元故障注入，支持自定义注入函数或指定注入位置和值
        self._reset_fault_injection_state()
        custom_injection = False
        injection_function = False

        if kwargs:
            if "function" in kwargs:
                # logging.info("Declaring Custom Function")
                custom_injection, injection_function = True, kwargs.get("function")
            else:
                # logging.info("Declaring Specified Fault Injector")
                self.corrupt_value = kwargs.get("value", [])

            # 读取注入位置参数
            self.corrupt_layer = kwargs.get("layer_num", [])
            self.corrupt_batch = kwargs.get("batch", [])
            self.corrupt_dim[0] = kwargs.get("dim1", [])
            self.corrupt_dim[1] = kwargs.get("dim2", [])
            self.corrupt_dim[2] = kwargs.get("dim3", [])

            # logging.info(f"Convolution: {self.corrupt_layer}")
            # logging.info("Batch, x, y, z:")
            # logging.info(
            #     f"{self.corrupt_batch}, {self.corrupt_dim[0]}, {self.corrupt_dim[1]}, {self.corrupt_dim[2]}"
            # )
        else:
            raise ValueError("Please specify an injection or injection function")

        # 检查注入位置是否越界
        self.check_bounds(
            self.corrupt_batch,
            self.corrupt_layer,
            self.corrupt_dim,
        )

        # 复制原始模型
        self.corrupted_model = copy.deepcopy(self.original_model)
        # 注册神经元注入钩子
        handles_neurons = self._traverse_model_set_hooks_neurons(
            self.corrupted_model,
            self._inj_layer_types,
            custom_injection,
            injection_function,
        )

        # 保存钩子句柄，方便后续移除
        for i in handles_neurons:
            self.handles.append(i)

        return self.corrupted_model

    def check_bounds(self, batch, layer, dim):
        # 检查注入位置参数长度是否匹配
        if (
            len(batch) != len(layer)
            or len(batch) != len(dim[0])
            or len(batch) != len(dim[1])
            or len(batch) != len(dim[2])
        ):
            raise AssertionError("Injection location missing values.")

        # logging.info("Checking bounds before runtime")
        # 逐个检查每个注入索引是否越界
        for i in range(len(batch)):
            self.assert_injection_bounds(i)

    def assert_injection_bounds(self, index: int):
        # 检查指定注入索引的合法性
        if index < 0:
            raise AssertionError(f"Invalid injection index: {index}")
        if self.corrupt_batch[index] >= self.batch_size:
            raise AssertionError(
                f"{self.corrupt_batch[index]} < {self.batch_size()}: Invalid batch element!"
            )
        if self.corrupt_layer[index] >= len(self.output_size):
            raise AssertionError(
                f"{self.corrupt_layer[index]} < {len(self.output_size)}: Invalid layer!"
            )

        corrupt_layer_num = self.corrupt_layer[index]
        layer_type = self.layers_type[corrupt_layer_num]
        layer_dim = self.layers_dim[corrupt_layer_num]
        layer_shape = self.output_size[corrupt_layer_num]

        # 检查各维度是否越界，维度从1开始对应C,H,W
        for d in range(1, 4):
            if layer_dim > d and self.corrupt_dim[d - 1][index] >= layer_shape[d]:
                raise AssertionError(
                    f"{self.corrupt_dim[d - 1][index]} < {layer_shape[d]}: Out of bounds error in Dimension {d}!"
                )

        # 如果层维度较低，警告忽略多余维度的注入值
        if layer_dim <= 2 and (
            self.corrupt_dim[1][index] is not None
            or self.corrupt_dim[2][index] is not None
        ):
            warnings.warn(
                f"Values in Dim2 and Dim3 ignored, since layer is {layer_type}"
            )

        if layer_dim <= 3 and self.corrupt_dim[2][index] is not None:
            warnings.warn(f"Values Dim3 ignored, since layer is {layer_type}")

        # logging.info(f"Finished checking bounds on inj '{index}'")

    def _set_value(self, module, input_val, output):
        # 默认的神经元注入钩子函数，根据注入列表修改输出张量对应位置的值
        # logging.info(
        #     f"Processing hook of Layer {self.current_layer}: {self.layers_type[self.current_layer]}"
        # )
        # 找出当前层需要注入的索引
        inj_list = list(
            filter(
                lambda x: self.corrupt_layer[x] == self.current_layer,
                range(len(self.corrupt_layer)),
            )
        )

        layer_dim = self.layers_dim[self.current_layer]

        # logging.info(f"Layer {self.current_layer} injection list size: {len(inj_list)}")
        if layer_dim == 2:
            # 对二维输出（如全连接层）注入
            for i in inj_list:
                self.assert_injection_bounds(index=i)
                # logging.info(
                #     f"Original value at [{self.corrupt_batch[i]}][{self.corrupt_dim[0][i]}]: {output[self.corrupt_batch[i]][self.corrupt_dim[0][i]]}"
                # )
                # logging.info(f"Changing value to {self.corrupt_value[i]}")
                output[self.corrupt_batch[i]][
                    self.corrupt_dim[0][i]
                ] = self.corrupt_value[i]
        elif layer_dim == 3:
            # 对三维输出（如某些卷积层输出）注入
            for i in inj_list:
                self.assert_injection_bounds(index=i)
                # logging.info(
                #     f"Original value at [{self.corrupt_batch[i]}][{self.corrupt_dim[0][i]}][{self.corrupt_dim[1][i]}]: {output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][self.corrupt_dim[1][i]]}"
                # )
                # logging.info(f"Changing value to {self.corrupt_value[i]}")
                output[self.corrupt_batch[i]][
                    self.corrupt_dim[0][i], self.corrupt_dim[1][i]
                ] = self.corrupt_value[i]
        elif layer_dim == 4:
            # 对四维输出（如标准卷积层输出）注入
            for i in inj_list:
                self.assert_injection_bounds(index=i)
                # logging.info(
                #     f"Original value at [{self.corrupt_batch[i]}][{self.corrupt_dim[0][i]}][{self.corrupt_dim[1][i]}][{self.corrupt_dim[2][i]}]: {output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][self.corrupt_dim[1][i]][self.corrupt_dim[2][i]]}"
                # )
                # logging.info(f"Changing value to {self.corrupt_value[i]}")
                output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                    self.corrupt_dim[1][i]
                ][self.corrupt_dim[2][i]] = self.corrupt_value[i]

        self.update_layer()

    def _save_output_size(self, module, input_val, output):
        # 钩子函数，用于保存每层输出的尺寸和维度信息
        self.output_isTuple.append(isinstance(output, tuple))  # 标记输出是否为元组
        if isinstance(output, tuple):
            # 如果输出是元组，取第一个元素
            output = output[0]
        shape = list(output.size())
        dim = len(shape)

        self.layers_type.append(type(module))  # 保存层类型
        self.layers_dim.append(dim)  # 保存输出维度数
        self.output_size.append(shape)  # 保存输出尺寸
        if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
            self.weights_size.append(module.weight.shape)
        else:
            self.weights_size.append(None)

    def update_layer(self, value=1):
        # 更新当前层索引，默认加1
        self.current_layer += value

    def reset_current_layer(self):
        # 重置当前层索引为0
        self.current_layer = 0

    def get_weights_size(self, layer_num):
        # 获取指定层权重尺寸
        return self.weights_size[layer_num]

    def get_weights_dim(self, layer_num):
        # 获取指定层权重维度数
        return len(self.weights_size[layer_num])

    def get_layer_type(self, layer_num):
        # 获取指定层类型
        return self.layers_type[layer_num]

    def get_layer_dim(self, layer_num):
        # 获取指定层输出维度数
        return self.layers_dim[layer_num]

    def get_layer_shape(self, layer_num):
        # 获取指定层输出尺寸
        return self.output_size[layer_num]

    def get_total_layers(self):
        # 获取模型总层数
        return len(self.output_size)

    def get_tensor_dim(self, layer, dim):
        # 获取指定层指定维度的尺寸，dim从0开始
        if dim > len(self.layers_dim):
            raise AssertionError(f"Dimension {dim} is out of bounds for layer {layer}")
        return self.output_size[layer][dim]

    def print_pytorchfi_layer_summary(self):
        # 打印模型层信息摘要，包括允许注入的层类型、输入形状、批大小、CUDA状态、每层信息等
        summary_str = (
            "============================ PYTORCHFI INIT SUMMARY =============================="
            + "\n\n"
        )

        summary_str += "Layer types allowing injections:\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )
        for l_type in self._inj_layer_types:
            summary_str += "{:>5}".format("- ")
            substring = str(l_type).split(".")[-1].split("'")[0]
            summary_str += substring + "\n"
        summary_str += "\n"

        summary_str += "Model Info:\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )

        summary_str += "   - Shape of input into the model: ("
        for dim in self._input_shape:
            summary_str += str(dim) + " "
        summary_str += ")\n"

        summary_str += "   - Batch Size: " + str(self.batch_size) + "\n"
        summary_str += "   - CUDA Enabled: " + str(self.use_cuda) + "\n\n"

        summary_str += "Layer Info:\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )
        line_new = "{:>5}  {:>15}  {:>10} {:>20} {:>20}".format(
            "Layer #", "Layer type", "Dimensions", "Weight Shape", "Output Shape"
        )
        summary_str += line_new + "\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )
        for layer, _dim in enumerate(self.output_size):
            weight_str = str(list(self.weights_size[layer])) if self.weights_size[layer] is not None else "N/A"
            line_new = "{:>5}  {:>15}  {:>10} {:>20} {:>20}".format(
                layer,
                str(self.layers_type[layer]).split(".")[-1].split("'")[0],
                str(self.layers_dim[layer]),
                weight_str,
                str(self.output_size[layer]),
            )
            summary_str += line_new + "\n"

        summary_str += (
            "=================================================================================="
            + "\n"
        )

        logging.info(summary_str)
        return summary_str
