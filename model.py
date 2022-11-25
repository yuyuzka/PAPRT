import torch

from time_aware_pe import TAPE
from attn_modules import *


class STiSAN(nn.Module):
    def __init__(self, n_loc, n_quadkey, features, exp_factor, k_t, k_g, depth, dropout):
        super(STiSAN, self).__init__()
        self.emb_loc = Embedding(n_loc, features, True, True)
        self.emb_quadkey = Embedding(n_quadkey, features, True, True)
        # self.emb_period =  Embedding(169, features, True, True)

        self.geo_encoder_layer = GeoEncoderLayer(features, exp_factor, dropout)
        self.geo_encoder = GeoEncoder(features, self.geo_encoder_layer, depth=2)

        self.tape = TAPE(dropout)

        self.k_t = torch.tensor(k_t)
        self.k_g = torch.tensor(k_g)

        self.inr_awa_attn_layer = InrEncoderLayer(features * 2, exp_factor, dropout,n_heads=8)
        self.inr_awa_attn_block = InrEncoder(features * 2, self.inr_awa_attn_layer, depth)

        self.attn=Attn(self.emb_loc,self.emb_quadkey,n_loc-1,n_quadkey-1)

        self.trg_awa_attn_decoder = TrgAwaDecoder(features * 2,features*2,dropout)

        # self.fc_loc=nn.Linear(features*2,1)

    # def forward(self, src_loc, src_quadkey, src_time, t_mat, g_mat, pad_mask, attn_mask,
    #             trg_loc, trg_quadkey, key_pad_mask, mem_mask, ds,src_period):
    def forward(self, src_loc, src_quadkey, src_time,  pad_mask, attn_mask,
                trg_loc, trg_quadkey, key_pad_mask, mem_mask, ds,
                src_period, w_mat,c_mat):
        # (b, n, d)
        src_loc_emb = self.emb_loc(src_loc)
        # (b, n * (1 + k), d)
        trg_loc_emb = self.emb_loc(trg_loc)

        # src_period_emb = self.emb_period(src_period)

        # (b, n, l, d)
        src_quadkey_emb = self.emb_quadkey(src_quadkey)
        # (b, n, d)
        src_quadkey_emb = self.geo_encoder(src_quadkey_emb)
        # (b, n * (1 + k), d)
        trg_quadkey_emb = self.emb_quadkey(trg_quadkey)
        # (b, n * (1 + k), d)
        trg_quadkey_emb = self.geo_encoder(trg_quadkey_emb)

        # src_loc_emb=src_loc_emb+src_period_emb

        # (b, n, 2 * d)
        src = torch.cat([src_loc_emb, src_quadkey_emb], dim=-1)
        # (b, n * (1 + k), 2 * d)
        trg = torch.cat([trg_loc_emb, trg_quadkey_emb], dim=-1)
        # (b, n, 2 * d)
        src = self.tape(src, src_time, ds)
        # (b, n, n)
        # for i in range(src.size(0)):
        #     mask = torch.gt(t_mat[i], self.k_t)
        #     t_mat[i] = t_mat[i].masked_fill(mask == True, self.k_t)
        #     t_mat[i] = t_mat[i].max() - t_mat[i]
        #     mask = torch.gt(g_mat[i], self.k_g)
        #     g_mat[i] = g_mat[i].masked_fill(mask == True, self.k_g)
        #     g_mat[i] = g_mat[i].max() - g_mat[i]

        for i in range(src.size(0)):
            
            w_mat[i] = 7 - w_mat[i]
            c_mat[i] = 24 - c_mat[i]
            # mask = torch.gt(s_mat[i], self.k_g)
            # s_mat[i] = s_mat[i].masked_fill(mask == True, self.k_g)
            # s_mat[i] = s_mat[i].max() - s_mat[i]
        # # (b, n, n)
        # r_mat = t_mat + g_mat
        r_mat = w_mat + c_mat
        # (b, n, 2 * d)
        src = self.inr_awa_attn_block(src, r_mat, attn_mask, pad_mask)
        # src = self.inr_awa_attn_block(src, attn_mask, pad_mask)
        srcout=self.attn(src,ds)

        if self.training:
            # (b, n * (1 + k), 2 * d)
            src = src.repeat(1, trg.size(1)//src.size(1), 1)
            # (b, n * (1 + k), 2 * d)
            src = self.trg_awa_attn_decoder(src, trg, key_pad_mask, mem_mask)
        else:
            # (b, 2 * d)
            src = src[torch.arange(len(ds)), torch.tensor(ds) - 1, :]
            # (b, 1 + k, 2 * d)
            src = src.unsqueeze(1).repeat(1, trg.size(1), 1)
            src = self.trg_awa_attn_decoder(src, trg, key_pad_mask, mem_mask)
        # (b, 1 + k)
        output = torch.sum(src * trg, dim=-1)
        # srcout=self.fc_loc(src)
        return output,srcout
        # return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))