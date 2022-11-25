from torch import tensor
import torch
from utils import haversine
import torch.nn as nn
from einops import reduce, rearrange

# class distance_loss(nn.Module):
#     def __init__(self):
#         super(distance_loss, self).__init__()
        
#     def forward(self, predict_loc, trg_loc,id2gps):
#         [b,n,l]=predict_loc.shape
#         predict_label_test = rearrange(predict_loc, 'b n l -> ( b n )l')
#         predict_label =predict_label_test.expand(-1,2*l)
#         trg_label = rearrange(trg_loc, 'b n l -> ( b n )l').expand(-1,2*l)
#         # trg_label =trg_label.expand(-1,2*l)
#         value=tensor(list(id2gps.values()))
#         predict_gps_test = value.gather(0,predict_label)
#         trg_gps_test = value.gather(0,trg_label)
#         d_gps=trg_gps_test-predict_gps_test
        
#         a = torch.sin(d_gps[:,0]/2).pow(2)
#         +torch.cos(predict_gps_test[:,0])*torch.cos(trg_gps_test[:,0])*(torch.sin(d_gps[:,1]/2).pow(2))

#         c=2*torch.asin(a.sqrt())
#         r=6371
#         dis_loss= torch.mean(torch.floor(c*r)) 
        
        
#         return dis_loss

