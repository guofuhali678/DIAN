#   Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_{K}}})V
#简单多头注意力机制
#KV Cache,MLA,GQA,MQA机制

#以下代码来自CSDN，仅对部分数字进行调整，实在无法复刻。
# 但一直在阅读相关有关transformer和注意力机制的博客。对原理进行了一部分理解
#multitransformer
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads  # Depth of each head

        # Linear layers for Q, K, V
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        # Final linear layer
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear transformations
        Q = self.WQ(Q)  # (batch_size, seq_len, d_model)
        K = self.WK(K)  # (batch_size, seq_len, d_model)
        V = self.WV(V)  # (batch_size, seq_len, d_model)

        # Split into multiple heads
        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, depth)
        original_size_attention = scaled_attention.view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)

        # Final linear layer
        output = self.dense(original_size_attention)  # (batch_size, seq_len, d_model)

        return output, attention_weights

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Calculate the attention weights and output."""
        matmul_qk = torch.matmul(Q, K.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        dk = K.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  # Apply mask

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, depth)

        return output, attention_weights
#MQA 让所有的头之间 共享 同一份 Key 和 Value 矩阵，每个头只单独保留了一份 Query 参数，从而大大减少 Key 和 Value 矩阵的参数量
# 参数设置
d_model = 64  # 输入维度
num_heads = 8  # 注意力头数
seq_len = 10   # 序列长度
batch_size = 2 # 批量大小

# 随机生成输入矩阵 (batch_size, seq_len, d_model)
#三个参数
Q = torch.rand(batch_size, seq_len, d_model)
K = torch.rand(batch_size, seq_len, d_model)
V = torch.rand(batch_size, seq_len, d_model)

# 初始化多头注意力机制
multi_head_attention = MultiHeadAttention(d_model, num_heads)

# 计算注意力权重和输出
output, attention_weights = multi_head_attention(Q, K, V)

# 打印结果
print("Output shape:", output.shape)  # 输出形状
print("Attention weights shape:", attention_weights.shape)  # 注意力权重形状
print("Attention weights:", attention_weights)  # 注意力权重值
