import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super(GroupedQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_groups = num_groups
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
        self.depth = d_model // num_heads

        # 线性层用于 Q
        self.WQ = nn.Linear(d_model, d_model)
        # 线性层用于 K 和 V，每个组共享一组 KV
        self.WKV = nn.Linear(d_model, 2 * self.depth * num_groups)

        # 最终的线性层
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """将最后一个维度拆分为 (num_heads, depth)。"""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性变换
        Q = self.WQ(Q)
        KV = self.WKV(K)
        KV = KV.view(batch_size, -1, self.num_groups, 2 * self.depth)
        K, V = torch.split(KV, self.depth, dim=-1)

        # 扩展 K 和 V 以匹配头的数量
        K = K.unsqueeze(2).expand(-1, -1, self.num_heads // self.num_groups, -1, -1).reshape(
            batch_size, -1, self.num_heads, self.depth)
        V = V.unsqueeze(2).expand(-1, -1, self.num_heads // self.num_groups, -1, -1).reshape(
            batch_size, -1, self.num_heads, self.depth)

        # 将 Q 拆分为多个头
        Q = self.split_heads(Q, batch_size)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # 缩放点积注意力
        scaled_attention, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 拼接头
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        original_size_attention = scaled_attention.view(batch_size, -1, self.d_model)

        # 最终的线性层
        output = self.dense(original_size_attention)

        return output, attention_weights

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """计算注意力权重和输出。"""
        matmul_qk = torch.matmul(Q, K.permute(0, 1, 3, 2))
        dk = K.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class GQAPredictor(nn.Module):
    def __init__(self, d_model, num_heads, num_groups, output_size):
        super(GQAPredictor, self).__init__()
        self.gqa = GroupedQueryAttention(d_model, num_heads, num_groups)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, Q, K, V, mask=None):
        output, attention_weights = self.gqa(Q, K, V, mask)
        # 这里简单地取序列最后一个时间步的特征进行预测
        last_step_output = output[:, -1, :]
        prediction = self.fc(last_step_output)

        print("Output shape:", output.shape)
        print("Attention weights shape:", attention_weights.shape)
        print("Attention weights:", attention_weights)

        return prediction


# 参数设置
d_model = 64
num_heads = 8
num_groups = 2  # 组数
seq_len = 10
batch_size = 2
output_size = 1  # 假设预测一个标量值

# 随机生成输入矩阵 (batch_size, seq_len, d_model)
Q = torch.rand(batch_size, seq_len, d_model)
K = torch.rand(batch_size, seq_len, d_model)
V = torch.rand(batch_size, seq_len, d_model)

# 初始化 GQA 预测器
gqa_predictor = GQAPredictor(d_model, num_heads, num_groups, output_size)

# 进行预测
prediction = gqa_predictor(Q, K, V)

# 打印预测结果
print("Prediction shape:", prediction.shape)
print("Prediction values:", prediction)