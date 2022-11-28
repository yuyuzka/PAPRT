import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import math


def clones(module, num_sub_layer):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_sub_layer)])


class Embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=False):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = nn.Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        outputs = F.embedding(inputs, self.lookup_table, self.padding_idx, None, 2, False, False)
        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class SubLayerConnect(nn.Module):
    def __init__(self, features):
        super(SubLayerConnect, self).__init__()
        self.norm = nn.LayerNorm(features)

    def forward(self, x, sublayer):
        # (*, d)
        return x + sublayer(self.norm(x))

# 简单的前馈网络
class FFN(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(features, exp_factor * features)
        self.act = nn.ReLU()
        self.w_2 = nn.Linear(exp_factor * features, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x



class SA(nn.Module):
    def __init__(self, dropout):
        super(SA, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, pad_mask):
        scale_term = math.sqrt(x.size(-1))
        scores = torch.matmul(x, x.transpose(-2, -1)) / scale_term
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask == 0.0, -1e9)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            scores = scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        return torch.matmul(prob, x)


class GeoEncoderLayer(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(GeoEncoderLayer, self).__init__()
        self.sa_layer = SA(dropout)
        self.ffn_layer = FFN(features, exp_factor, dropout)
        self.sublayer = clones(SubLayerConnect(features), 2)

    def forward(self, x):
        # (b ,n, l, d)
        x = self.sublayer[0](x, lambda x:self.sa_layer(x, None, None))
        x = self.sublayer[1](x, self.ffn_layer)
        return x


class GeoEncoder(nn.Module):
    def __init__(self, features, layer, depth):
        super(GeoEncoder, self).__init__()
        self.layers = clones(layer, depth)
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=-2)
        return self.norm(x)


# 自注意力机制Q，K，V？？？
class InrAwaSA(nn.Module):
    def __init__(self, features,output_size,dropout):
        super(InrAwaSA, self).__init__()
        self.query=nn.Linear(features,output_size,bias=False)
        self.key=nn.Linear(features,output_size,bias=False)
        self.value=nn.Linear(features,output_size,bias=False)
        self.dropout = nn.Dropout(dropout)

    # def forward(self, x, r_mat, attn_mask, pad_mask):
    def forward(self, x, attn_mask, pad_mask):
        scale_term = math.sqrt(x.size(-1))
        # scores = torch.matmul(x, x.transpose(-2, -1)) / scale_term
        scores = torch.matmul(self.query(x), self.key(x).transpose(-2, -1)) / scale_term
        # mask = pad_mask
        # r_mat = r_mat.masked_fill(mask == 0.0, -1e9)
        # mask = attn_mask.unsqueeze(0)
        # r_mat = r_mat.masked_fill(mask == 0.0, -1e9)
        # r_mat = F.softmax(r_mat, dim=-1)
        # scores += r_mat
        if pad_mask is not None:
            scores.masked_fill(pad_mask == 0.0, -1e9)
        if attn_mask is not None:
            attn_mask.unsqueeze(0)
            scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        # return torch.matmul(prob, x)
        return torch.matmul(prob, self.value(x))



class InrEncoderLayer(nn.Module):
    # def __init__(self, features, exp_factor, dropout):
    def __init__(self, features, exp_factor, dropout,n_heads):
        super(InrEncoderLayer, self).__init__()
        # self.inr_sa_layer = InrAwaSA(dropout)
        # self.inr_sa_layer = InrAwaSA(features, features, dropout)
        self.inr_sa_layer = MultiHeadAttentionBlock(features, int(features/n_heads), n_heads,dropout)
        self.ffn_layer = FFN(features, exp_factor, dropout)
        self.sublayer = clones(SubLayerConnect(features), 2)

    def forward(self, x, r_mat, attn_mask, pad_mask):
    # def forward(self, x, attn_mask, pad_mask):
        x = self.sublayer[0](x, lambda x:self.inr_sa_layer(x, r_mat, attn_mask, pad_mask))
        # x = self.sublayer[0](x, lambda x:self.inr_sa_layer(x, attn_mask, pad_mask))
        x = self.sublayer[1](x, self.ffn_layer)
        return x


class InrEncoder(nn.Module):
    def __init__(self, features, layer, depth):
        super(InrEncoder, self).__init__()
        self.layers = clones(layer, depth)
        self.norm = nn.LayerNorm(features)

    def forward(self, x, r_mat, attn_mask, pad_mask):
    # def forward(self, x, attn_mask, pad_mask):
        for layer in self.layers:
            x = layer(x, r_mat, attn_mask, pad_mask)
            # x = layer(x, attn_mask, pad_mask)
        return self.norm(x)

    #注意力机制Q，K，V ？？？
class TrgAwaDecoder(nn.Module):
    # def __init__(self, features, dropout):
    def __init__(self, features, output_size,dropout):
        super(TrgAwaDecoder, self).__init__()
        self.query=nn.Linear(features,output_size,bias=False)
        self.key=nn.Linear(features,output_size,bias=False)
        self.value=nn.Linear(features,output_size,bias=False)
        self.norm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src, trg, key_pad_mask, mem_mask):
        res = src
        src = self.norm(src)
        trg = self.norm(trg)
        scale_term = src.size(-1)
        # scores = torch.matmul(trg ,src.transpose(-2, -1)) / scale_term
        scores = torch.matmul(self.query(trg) ,self.key(src).transpose(-2, -1)) / scale_term
        if key_pad_mask is not None:
            scores = scores.masked_fill(key_pad_mask == 0.0, -1e9)
        if mem_mask is not None:
            mem_mask.unsqueeze(0)
            scores = scores.masked_fill(mem_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        # x = res + torch.matmul(prob, src)
        x = res + torch.matmul(prob, self.value(src))
        return self.norm(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, features,output_size,dropout):
        super(SelfAttentionBlock, self).__init__()
        self.query=nn.Linear(features,output_size,bias=False)
        self.key=nn.Linear(features,output_size,bias=False)
        self.value=nn.Linear(features,output_size,bias=False)
        self.dropout = nn.Dropout(dropout)

    # def forward(self, x, attn_mask, pad_mask):
    def forward(self, x, r_mat, attn_mask, pad_mask):
        scale_term = math.sqrt(x.size(-1))
        scores = torch.matmul(self.query(x), self.key(x).transpose(-2, -1)) / scale_term
        mask = pad_mask
        r_mat = r_mat.masked_fill(mask == 0.0, -1e9)
        mask = attn_mask.unsqueeze(0)
        r_mat = r_mat.masked_fill(mask == 0.0, -1e9)
        r_mat = F.softmax(r_mat, dim=-1)
        scores += r_mat
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask == 0.0, -1e9)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            scores = scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        return torch.matmul(prob, self.value(x))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads,dropout):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(SelfAttentionBlock(dim_val, dim_attn,dropout))
        
        self.heads = nn.ModuleList(self.heads)
        
        self.fc = nn.Linear(n_heads * dim_attn, dim_val, bias = False)
                      
        
    def forward(self, x,r_mat, attn_mask, pad_mask):
        a = []
        for attn in self.heads:
            a.append(attn(x,r_mat, attn_mask, pad_mask))
            
        a = torch.stack(a, dim = -1) #combine heads
        a = a.flatten(start_dim = 2) #flatten all head outputs
        
        x = self.fc(a)
        
        return x
    
class Attn(nn.Module):
    def __init__(self, emb_loc,emb_quadkey, loc_max, quadkey_max,dropout=0.1):
        super(Attn, self).__init__()
        self.value = nn.Linear(100, 1, bias=False)
        self.emb_loc = emb_loc
        self.emb_quadkey=emb_quadkey
        self.loc_max = loc_max
        self.quadkey_max=quadkey_max
        self.sigmoid_scores=nn.Sigmoid()

    def forward(self, src,ds):
        [N,L,M]=src.shape
        candidates_locs = torch.linspace(1, int(self.loc_max), int(self.loc_max)).long()  # (L)
        candidates_locs = candidates_locs.unsqueeze(0).expand(N, -1).to('cuda:0')  # (N, L)
        # candidates_locs = candidates_locs.unsqueeze(0).expand(N, -1).to('cpu')
        emb_candidates_locs = self.emb_loc(candidates_locs)  # (N, L, emb)
        src=torch.split(src,512,-1)[0]
        # candidates_quadkeys=torch.linspace(1, int(self.quadkey_max), int(self.quadkey_max)).long()
        # candidates_quadkeys = candidates_quadkeys.unsqueeze(0).expand(N, -1)
        # emb_candidates_quadkeys = self.emb_quadkey(candidates_quadkeys)
        # emb_candidates=torch.cat([emb_candidates_locs, emb_candidates_quadkeys], dim=-1)
        # 注意力解码
        attn = torch.matmul(src, emb_candidates_locs.transpose(-1, -2))
        # 源代码乘法方式
        # src = src[torch.arange(len(ds)), torch.tensor(ds) - 1, :]
        # # (b, L,  d)
        # src = src.unsqueeze(1).repeat(1, self.loc_max, 1)
        # # src = src.repeat(1, self.loc_max//src.size(1), 1)
        # attn = torch.sum(src * emb_candidates_locs, dim=-1)

        # mean=torch.mean(attn,dim=-1,keepdim=True)
        # std=torch.std(attn,dim=-1,keepdim=True)
        # attn =attn.sub(mean).div(std)
        # norm=nn.BatchNorm1d(self.loc_max)
        # attn_scores = norm(attn)
        # attn_scores = F.softmax(attn,dim=-1)
        # min=torch.min(attn,dim=-1)[0]
        # max=torch.max(attn,dim=-1)
        # attn_scores =attn.sub(min).div(max.sub(min))
        
        attn_scores=self.sigmoid_scores(attn)
        # print(attn_scores[0,:512])
      
        # attn_out = self.value(attn) # (N, L)
        
        return attn_scores  # (N, L)