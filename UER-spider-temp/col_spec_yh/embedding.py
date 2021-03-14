# -*- encoding:utf-8 -*-
# author : Guo Yuhe
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm


class WordEmbedding(nn.Module):
    def __init__(self, args, vocab_size):
        super(WordEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg):
        word_emb = self.word_embedding(src)
        word_emb = self.dropout(self.layer_norm(word_emb))
        return word_emb



class TabEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(TabEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg):
        word_emb = self.word_embedding(src)
        # pos_emb = self.position_embedding(seg // 10000)
        # pos_emb_token = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, \
        #                                                dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        pos_token = torch.arange(0, word_emb.size(1)).unsqueeze(0).repeat(word_emb.size(0),1).long().to(word_emb.device)
        pos_emb_token = self.position_embedding(pos_token)
        pos_emb = pos_emb_token

        # seg token
        ones = torch.ones_like(seg)
        twos = ones * 2
        seg_token = torch.where(seg % 10000 // 100 > 1, twos, seg)
        seg_token = torch.where(seg % 10000 // 100 == 1, ones, seg_token)
        seg_emb = self.segment_embedding(seg_token)


        # in_cell_pos_embs = self.position_embedding(seg // 10000)
        # col_cell_pos = self.position_embedding(seg % 10000 // 100)
        # row_cell_pos = self.position_embedding(seg % 10000 % 100)
        # pos_emb = (col_cell_pos + row_cell_pos + in_cell_pos_embs + pos_emb_token) / 2

        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb


class MinimalEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(MinimalEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        # self.segment_embedding = nn.Embedding(3, args.emb_size)
        # self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg):
        word_emb = self.word_embedding(src)
        emb = word_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb
