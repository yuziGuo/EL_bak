# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
import dgl.ops as ops

class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(3)
            ])
        
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, mask):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, seq_length, hidden_size)


        query, key, value = [l(x). \
                             view(batch_size, -1, heads_num, per_head_size). \
                             transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                            ] # [b, 6, l, 128] --> [b, l, 6, 128]

        scores = torch.matmul(query, key.transpose(-2, -1))  # [b, 6, l, l]
        scores = scores / math.sqrt(float(per_head_size)) 
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))    # [b, 6, l, 128]
        output = self.final_linear(output)
        
        return output


class MultiHeadedAttentionOnGraph(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttentionOnGraph, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(3)
        ])

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, g):
        """
        Args:
            h: [seq_length x hidden_size]

        Returns:
            output: [seq_length x hidden_size]
        """
        seq_length, hidden_size = h.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                contiguous(). \
                view(batch_size, seq_length, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, hidden_size)

        # query, key, value = [l(x). \
        #                          view(batch_size, -1, heads_num, per_head_size). \
        #                          transpose(1, 2) \
        #                      for l, x in zip(self.linear_layers, (query, key, value))
        #                      ]  # [b, 6, l, 128] --> [b, l, 6, 128]
        q = self.linear_layers[0](h).view(-1, heads_num, per_head_size)  # (N, h, d_qkv)
        k =   self.linear_layers[1](h).view(-1, heads_num, per_head_size)  # (N, h, d_qkv)
        v = self.linear_layers[2](h).view(-1, heads_num, per_head_size)  # (N, h, d_qkv)

        # scores = torch.matmul(query, key.transpose(-2, -1))  # [b, 6, l, l]
        # scores = scores / math.sqrt(float(per_head_size))
        # scores = scores + mask
        # probs = nn.Softmax(dim=-1)(scores)
        # probs = self.dropout(probs)

        scores = ops.u_dot_v(g, k, q)  # (E, h, 1): dot product of query and key
        scores = scores / math.sqrt(float(per_head_size))
        probs = ops.edge_softmax(g, scores)  # (E, h, 1)
        probs = self.dropout(probs)

        # output = unshape(torch.matmul(probs, value))  # [b, 6, l, 128]
        # output = self.final_linear(output)
        # return output
        output = ops.u_mul_e_sum(g, v, probs)  # (N, h, d_qkv): weighted sum of value by attention score
        output = output.view(-1, heads_num*per_head_size)
        output = self.final_linear(output)
        return output


class RelationAwareMultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, dropout):
        super(RelationAwareMultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(3)
        ])

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, mask, rel_embs_k, rel_embs_v):  # mask values indicate relationships
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                contiguous(). \
                view(batch_size, seq_length, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, hidden_size)

        query, key, value = [l(x). \
                                 view(batch_size, -1, heads_num, per_head_size). \
                                 transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                             ]  # [b, 6, l, 128]

        rel_k = rel_embs_k(mask.type(torch.LongTensor).to(key.device))  # [b, 1, l, l, 64]
        rel_v = rel_embs_v(mask.type(torch.LongTensor).to(key.device))  # [b, 1, l, l, 64]
        scores = torch.matmul(query, key.transpose(-2, -1))  # [b, 12, l, l]
        scores_rel = torch.matmul(query.unsqueeze(-2), rel_k.transpose(-1, -2)).squeeze(-2)  # [b, 12, l, l]
        scores += scores_rel / 4.0
        scores = scores / math.sqrt(float(per_head_size))

        mask = (mask > 0).float()
        mask = (1.0 - mask) * -10000.0
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)

        # import ipdb; ipdb.set_trace()
        output = unshape(torch.matmul(probs, value))
        output_rel = unshape(torch.matmul(probs.unsqueeze(-2), rel_v).squeeze(-2))
        output = self.final_linear(output+output_rel*0.25)

        return output
