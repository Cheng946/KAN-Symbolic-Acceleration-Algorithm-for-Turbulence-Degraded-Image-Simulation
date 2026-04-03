import torch
import numpy as np
import argparse
import os
from collections import defaultdict
import torch.nn as nn
import re

# ===================== 手动配置核心参数（你可以直接在这里修改）=====================
# 基础激活函数保留的最大权重项数（手动设置）
TOPK_BASE = 1
# 初等函数内部线性变换保留的最大权重项数（手动设置）
TOPK_EF_LINEAR = 1
# 强制仅使用权重最大的1个初等函数（固定为1，无需修改）
TOPK_EF_FUNC = 1

# 定义所有支持的初等函数及其计算逻辑
SUPPORTED_ELEMENTARY_FUNCTIONS = {
    'silu': lambda x: torch.nn.functional.silu(x),
    'relu': lambda x: torch.nn.functional.relu(x),
    'sin': lambda x: torch.sin(x),
    'cos': lambda x: torch.cos(x),
    'exp': lambda x: torch.exp(torch.clamp(x, -10, 10)),
    'log': lambda x: torch.log(torch.abs(x) + 1e-6),
    'tanh': lambda x: torch.tanh(x),
    'sigmoid': lambda x: torch.sigmoid(x),
    'sqrt': lambda x: torch.sqrt(torch.abs(x) + 1e-6),
    'square': lambda x: torch.square(x),
    'abs': lambda x: torch.abs(x),
    'identity': lambda x: x
}

# 默认初等函数集合
DEFAULT_ELEMENTARY_FUNCTIONS = ['silu', 'relu', 'tanh', 'sigmoid', 'abs', 'identity']


# ========== 2. SymbolicKAN 定义 ==========
class SymbolicKANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            scale_base=1.0,
            scale_mlp=1.0,
            base_activation=torch.nn.SiLU,
            elementary_functions=None,
    ):
        super(SymbolicKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_base = scale_base
        self.scale_mlp = scale_mlp
        self.base_activation = base_activation()

        if elementary_functions is None:
            self.elementary_functions = DEFAULT_ELEMENTARY_FUNCTIONS
        else:
            invalid_funcs = [f for f in elementary_functions if f not in SUPPORTED_ELEMENTARY_FUNCTIONS]
            if invalid_funcs:
                raise ValueError(
                    f"不支持的初等函数: {invalid_funcs}，支持的函数列表: {list(SUPPORTED_ELEMENTARY_FUNCTIONS.keys())}")
            self.elementary_functions = elementary_functions

        self.num_ef = len(self.elementary_functions)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.ef_mlp_linears = nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(out_features, in_features))
            for _ in range(self.num_ef)
        ])
        self.ef_mlp_biases = nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(out_features))
            for _ in range(self.num_ef)
        ])

        self.ef_weights = torch.nn.Parameter(torch.ones(self.num_ef))

        self.reset_parameters()

    def reset_parameters(self):
        import math
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        for i in range(self.num_ef):
            torch.nn.init.kaiming_uniform_(self.ef_mlp_linears[i], a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.ef_mlp_linears[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.ef_mlp_biases[i], -bound, bound)

        torch.nn.init.constant_(self.ef_weights, 1.0 / self.num_ef)

    def apply_elementary_function(self, x, func_idx):
        import torch.nn.functional as F
        func_name = self.elementary_functions[func_idx]
        mlp_output = F.linear(x, self.ef_mlp_linears[func_idx], self.ef_mlp_biases[func_idx])
        ef_output = SUPPORTED_ELEMENTARY_FUNCTIONS[func_name](mlp_output)
        return ef_output * self.ef_weights[func_idx]

    def forward(self, x: torch.Tensor):
        import torch.nn.functional as F
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)

        mlp_outputs = []
        for i in range(self.num_ef):
            ef_mlp_output = self.apply_elementary_function(x, i)
            mlp_outputs.append(ef_mlp_output)

        mlp_output = torch.stack(mlp_outputs, dim=-1).sum(dim=-1) * self.scale_mlp
        output = base_output + mlp_output
        return output.reshape(*original_shape[:-1], self.out_features)

    def get_l1_regularization(self):
        return self.ef_weights.abs().sum()


class SymbolicKAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            scale_base=1.0,
            scale_mlp=1.0,
            base_activation=torch.nn.SiLU,
            elementary_functions=None,
    ):
        super(SymbolicKAN, self).__init__()

        self.layers = torch.nn.ModuleList()
        if elementary_functions is None:
            ef_list = DEFAULT_ELEMENTARY_FUNCTIONS
        else:
            invalid_funcs = [f for f in elementary_functions if f not in SUPPORTED_ELEMENTARY_FUNCTIONS]
            if invalid_funcs:
                raise ValueError(
                    f"不支持的初等函数: {invalid_funcs}，支持的函数列表: {list(SUPPORTED_ELEMENTARY_FUNCTIONS.keys())}")
            ef_list = elementary_functions

        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                SymbolicKANLinear(
                    in_features=in_features,
                    out_features=out_features,
                    scale_base=scale_base,
                    scale_mlp=scale_mlp,
                    base_activation=base_activation,
                    elementary_functions=ef_list,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_l1_regularization(self):
        total_l1 = 0.0
        for layer in self.layers:
            total_l1 += layer.get_l1_regularization()
        return total_l1


# ===================== 公式提取核心类 =====================
class SymbolicFormulaExtractor:
    def __init__(
            self,
            model: SymbolicKAN,
            device: torch.device,
            weight_threshold: float = 0.01,
            input_names: list = None,
            output_names: list = None
    ):
        self.model = model.eval()
        self.device = device
        self.weight_threshold = weight_threshold
        self.func_map = SUPPORTED_ELEMENTARY_FUNCTIONS

        # 原始网络输入
        self.original_input_names = input_names if input_names else [f"x_{i}" for i in range(model.layers[0].in_features)]
        self.final_output_names = output_names if output_names else [f"y_{i}" for i in range(model.layers[-1].out_features)]

        for param in self.model.parameters():
            param.requires_grad = False

    def simplify_weights(self, weights, bias=None):
        """权重剪枝+简化，过滤小权重"""
        weights = torch.where(
            torch.abs(weights) < self.weight_threshold,
            torch.zeros_like(weights),
            weights
        )
        weights = torch.round(weights * 1000) / 1000

        if bias is not None:
            bias = torch.where(
                torch.abs(bias) < self.weight_threshold,
                torch.zeros_like(bias),
                bias
            )
            bias = torch.round(bias * 1000) / 1000
            return weights, bias
        return weights

    def get_top_k_weights(self, weight_vector, k):
        """获取权重绝对值最大的前k个索引和值"""
        abs_weights = torch.abs(weight_vector)
        top_vals, top_indices = torch.topk(abs_weights, k=min(k, len(weight_vector)))
        mask = torch.zeros_like(weight_vector)
        mask[top_indices] = 1.0
        return weight_vector * mask, top_indices

    def layer_to_formula(self, layer: SymbolicKANLinear, input_names: list):
        """单层公式生成：严格按照需求保留最大权重项"""
        in_dim = layer.in_features
        out_dim = layer.out_features
        ef_names = layer.elementary_functions
        ef_weights = self.simplify_weights(layer.ef_weights)

        output_formulas = []

        for out_idx in range(out_dim):
            terms = []

            # 1. Base分支：仅保留权重最大的 TOPK_BASE 项
            base_w = self.simplify_weights(layer.base_weight[out_idx])
            base_w, _ = self.get_top_k_weights(base_w, k=TOPK_BASE)
            base_func = "silu"

            for in_idx in range(in_dim):
                w = base_w[in_idx].item()
                if abs(w) < 1e-6:
                    continue
                terms.append(f"{w:.3f}*{base_func}({input_names[in_idx]})")

            # 2. 初等函数分支：仅保留权重最大的 1 个函数
            if TOPK_EF_FUNC > 0:
                best_ef_idx = torch.argmax(torch.abs(ef_weights)).item()
                ef_w = ef_weights[best_ef_idx].item()
                func_name = ef_names[best_ef_idx]

                if abs(ef_w) >= 1e-6:
                    # 线性变换：仅保留权重最大的 TOPK_EF_LINEAR 项
                    lin_w = self.simplify_weights(layer.ef_mlp_linears[best_ef_idx][out_idx])
                    lin_w, _ = self.get_top_k_weights(lin_w, k=TOPK_EF_LINEAR)
                    lin_b = layer.ef_mlp_biases[best_ef_idx][out_idx].item()
                    lin_b = round(lin_b * 1000) / 1000

                    linear_terms = []
                    for in_idx in range(in_dim):
                        w = lin_w[in_idx].item()
                        if abs(w) < 1e-6:
                            continue
                        linear_terms.append(f"{w:.3f}*{input_names[in_idx]}")

                    if abs(lin_b) >= 1e-6:
                        linear_terms.append(f"{lin_b:.3f}")

                    if linear_terms:
                        linear_str = " + ".join(linear_terms)
                        func_str = f"{ef_w:.3f}*{func_name}({linear_str})"
                        terms.append(func_str)

            # 拼接公式
            if not terms:
                formula = "0"
            else:
                formula = " + ".join(terms)
                formula = formula.replace("+ -", "- ")
            output_formulas.append(formula)

        return output_formulas

    def simplify_full_formula(self, formula):
        """简化完整公式"""
        formula = re.sub(r'\- \+', '-', formula)
        formula = re.sub(r'\+ \-', '-', formula)
        formula = re.sub(r' +', ' ', formula)
        return formula

    def extract_full_formulas(self):
        print("=" * 70)
        print(f"🚀 开始提取【端到端】符号公式")
        print(f"✅ 核心配置：")
        print(f"   基础激活保留最大前 {TOPK_BASE} 项")
        print(f"   仅保留权重最大的 {TOPK_EF_FUNC} 个初等函数")
        print(f"   初等函数内部线性变换保留最大前 {TOPK_EF_LINEAR} 项")
        print(f"📥 网络输入: {len(self.original_input_names)}个变量 | 📤 网络输出: {len(self.final_output_names)}个变量")
        print("=" * 70)

        current_formulas = self.original_input_names.copy()

        for layer_idx, layer in enumerate(self.model.layers):
            print(f"\n📦 处理第 {layer_idx + 1}/{len(self.model.layers)} 层")
            current_formulas = self.layer_to_formula(layer, current_formulas)
            print(f"✅ 第 {layer_idx + 1} 层公式代入完成")

        final_formulas = [self.simplify_full_formula(f) for f in current_formulas]
        return final_formulas

    def save_formulas(self, save_path="symbolic_formulas_full.txt"):
        """保存端到端完整公式"""
        final_formulas = self.extract_full_formulas()

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"SymbolicKAN 端到端符号公式\n")
            f.write(f"✅ 核心配置：\n")
            f.write(f"   基础激活保留最大前 {TOPK_BASE} 项\n")
            f.write(f"   仅保留权重最大的 {TOPK_EF_FUNC} 个初等函数\n")
            f.write(f"   初等函数内部线性变换保留最大前 {TOPK_EF_LINEAR} 项\n")
            f.write(f"权重剪枝阈值: {self.weight_threshold}\n")
            f.write(f"网络输入变量: {', '.join(self.original_input_names)}\n")
            f.write(f"网络输出变量: {', '.join(self.final_output_names)}\n")
            f.write("=" * 60 + "\n\n")

            f.write("🎯 最终端到端公式：\n\n")
            for out_idx, (out_name, formula) in enumerate(zip(self.final_output_names, final_formulas)):
                f.write(f"{out_name} = {formula}\n\n")

        print(f"\n✅ 【端到端公式】已保存到: {save_path}")
        return final_formulas


# ===================== 运行脚本 =====================
def parse_extract_args():
    parser = argparse.ArgumentParser(description="SymbolicKAN 端到端公式提取工具")
    parser.add_argument('--model_path', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/Last_SymbolicKAN_Para_1_fold_4L_-524-524-524_15_2.pt",
                        help='训练好的模型权重路径')
    parser.add_argument('--input_nc', type=int, default=33, help='输入特征维度')
    parser.add_argument('--output_nc', type=int, default=70, help='输出特征维度')
    parser.add_argument('--elementary_functions', type=str, nargs='+',
                        default=DEFAULT_ELEMENTARY_FUNCTIONS)
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='权重剪枝阈值')
    parser.add_argument('--save_path', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_Output_Input/symbolic_formulas_1_1_1.txt",
                        help='公式保存路径')
    return parser.parse_args()


def main_extract():
    args = parse_extract_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 网络结构
    layers_hidden = [args.input_nc, 524, 524, 524, args.output_nc]
    model = SymbolicKAN(
        layers_hidden=layers_hidden,
        elementary_functions=args.elementary_functions
    ).to(device)

    print(f"🔍 加载模型权重: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    if 'parameter' in state_dict:
        model.load_state_dict(state_dict['parameter'])
    else:
        model.load_state_dict(state_dict)

    # ===================== 关键修改：输入输出符号定义 =====================
    # 输入：a_4 到 a_36 （共33个）
    input_names = [f"a_{i}" for i in range(4, 37)]
    # 输出：α_0 到 α_69 （共70个）
    output_names = [f"α_{i}" for i in range(70)]

    # 初始化提取器
    extractor = SymbolicFormulaExtractor(
        model=model,
        device=device,
        weight_threshold=args.threshold,
        input_names=input_names,
        output_names=output_names
    )

    # 提取并保存公式
    extractor.save_formulas(save_path=args.save_path)


if __name__ == "__main__":
    main_extract()
