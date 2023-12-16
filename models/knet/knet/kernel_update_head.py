import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiheadattention import (MultiheadAtten, Ffn)
from .kernel_updator import KernelUpdator
from timm.models.layers import trunc_normal_

from thop import profile
class DySepConvAtten(nn.Module):
    def __init__(self):
        super(DySepConvAtten, self).__init__()
        self.hidden_dim = 256#cfg.MODEL.YOSO.HIDDEN_DIM
        self.num_proposals = 64 #cfg.MODEL.YOSO.NUM_PROPOSALS
        self.kernel_size = 3    #cfg.MODEL.YOSO.CONV_KERNEL_SIZE_1D

        # self.depth_weight_linear = nn.Linear(hidden_dim, kernel_size)
        # self.point_weigth_linear = nn.Linear(hidden_dim, num_proposals)
        self.weight_linear = nn.Linear(self.hidden_dim, self.num_proposals + self.kernel_size)
        self.norm = nn.LayerNorm(self.hidden_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # print("init weights")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, query, value):
#        assert query.shape == value.shape
        B, N, C = query.shape
        value=value.reshape(B,N,C)
        # dynamic depth-wise conv
        # dy_depth_conv_weight = self.depth_weight_linear(query).view(B, self.num_proposals, 1,self.kernel_size) # B, N, 1, K
        # dy_point_conv_weight = self.point_weigth_linear(query).view(B, self.num_proposals, self.num_proposals, 1)

        dy_conv_weight = self.weight_linear(query)
        dy_depth_conv_weight = dy_conv_weight[:, :, :self.kernel_size].view(B,self.num_proposals,1,self.kernel_size)
        dy_point_conv_weight = dy_conv_weight[:, :, self.kernel_size:].view(B,self.num_proposals,self.num_proposals,1)

        res = []
        value = value.unsqueeze(1)
        for i in range(B):
            # input: [1, N, C]
            # weight: [N, 1, K]
            # output: [1, N, C]
            out = F.relu(F.conv1d(input=value[i], weight=dy_depth_conv_weight[i], groups=N, padding="same"))
            # input: [1, N, C]
            # weight: [N, N, 1]
            # output: [1, N, C]
            out = F.conv1d(input=out, weight=dy_point_conv_weight[i], padding='same')

            res.append(out)
        point_out = torch.cat(res, dim=0)
        point_out = self.norm(point_out)
        return point_out

class KernelUpdateHead(nn.Module):

    def __init__(self,
                 num_classes=80,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_mask_fcs=3,
                 in_channels=256,
                 out_channels=256,
                 dropout=0.0,
                 mask_thr=0.5,
                 conv_kernel_size=1,
                 hard_mask_thr=0.5,
                 kernel_init=False,
                 with_ffn=True,
                 mask_out_stride=4,
                 relative_coors=False,
                 relative_coors_off=False,
                 feat_gather_stride=1,
                 mask_transform_stride=2,
                 num_points=100
                 ):
        super(KernelUpdateHead, self).__init__()
        self.num_classes = num_classes


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_thr = mask_thr
        self.fp16_enabled = False
        self.dropout = dropout

        self.num_heads = num_heads
        self.hard_mask_thr = hard_mask_thr
        self.kernel_init = kernel_init
        self.with_ffn = with_ffn
        self.mask_out_stride = mask_out_stride
        self.relative_coors = relative_coors
        self.relative_coors_off = relative_coors_off
        self.conv_kernel_size = conv_kernel_size
        self.feat_gather_stride = feat_gather_stride
        self.mask_transform_stride = mask_transform_stride
        self.num_points = num_points

        self.loc_convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.GroupNorm(32, 256, eps=1e-05, affine=True),
            nn.ReLU(inplace=True)
        )

        self.topk_attn = MultiheadAtten(in_channels, num_heads, dropout)
        self.topk_norm = nn.LayerNorm(in_channels*(conv_kernel_size**2), \
                eps=1e-05, elementwise_affine=True)
       

        self.attention = MultiheadAtten(in_channels * (conv_kernel_size**2),
                                            num_heads, dropout)
        self.attention_norm = nn.LayerNorm(in_channels*(conv_kernel_size**2), eps=1e-05, elementwise_affine=True)
        self.kernel_update_conv = KernelUpdator(in_channels=256,
                                                feat_channels=256,
                                                out_channels=256,
                                                input_feat_shape=3,
                                                gate_sigmoid=True,
                                                gate_norm_act=False,
                                                activate_out=False,
                                                act_cfg=dict(type='ReLU', inplace=True),
                                                norm_cfg=dict(type='LN'))

        

        if self.with_ffn:
            self.ffn = Ffn()
            self.ffn_norm = nn.LayerNorm(in_channels, eps=1e-05, elementwise_affine=True)
            self.ffn_pre = Ffn()
            self.ffn_norm_pre = nn.LayerNorm(in_channels, eps=1e-05, elementwise_affine=True)
           

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                nn.LayerNorm((256,), eps=1e-05, elementwise_affine=True))
            self.cls_fcs.append(nn.ReLU(inplace=True))

        self.fc_cls = nn.Linear(in_channels, self.num_classes)

        self.mask_fcs = nn.ModuleList()
        for _ in range(num_mask_fcs):
            self.mask_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.mask_fcs.append(
                nn.LayerNorm((256,), eps=1e-05, elementwise_affine=True))
            self.mask_fcs.append(nn.ReLU(inplace=True))

        self.fc_mask = nn.Linear(in_channels, out_channels)
        #___
        self.f_atten = DySepConvAtten()
        self.k_atten = DySepConvAtten()
        self.f_dropout = nn.Dropout(0.0)
        self.f_atten_norm = nn.LayerNorm(256)
        self.k_dropout = nn.Dropout(0.0)
        self.k_atten_norm = nn.LayerNorm(256)
        
        self.s_atten = nn.MultiheadAttention(embed_dim=256,
                                             num_heads=8,
                                             dropout=0.0)
        self.s_dropout = nn.Dropout(0.0)
        self.s_atten_norm = nn.LayerNorm(256)
         
        #___
    #图片 文本 掩码
    def forward(self, x, proposal_feat, mask_preds, mask_shape=None):
        K = self.num_points
        x = self.loc_convs(x)
        # proposal_feat: [B, 230, 256]
        B, N = proposal_feat.shape[:2]
        # x: [B, 256, H//8, W//8] <--> Features $F$
        C, H, W = x.shape[-3:]
        # mask_preds: [B, 230, H//4, W//4] <--> $M$
        mask_h, mask_w = mask_preds.shape[-2:]
        if mask_h != H or mask_w != W:
            gather_mask = F.interpolate(
                mask_preds, (H, W), align_corners=False, mode='bilinear')
            # gather_mask: [B, 230, H//8, W//8]
        else:
            gather_mask = mask_preds

       # _, topk_inds = torch.topk(gather_mask.flatten(-2), K)
        # [B, N, K]
        v_feat = x.flatten(-2).transpose(1, 2)
        
        #__gtyupdated
     #  topk_feats=torch.
        gather_mask=gather_mask.sigmoid()
        gather_mask=(gather_mask>0.5).float()
        topk_feats= torch.einsum('bchw,bnhw->bnc', x, gather_mask)
        #
        # [B, HW, C]
        # topk_feats = []
        # for i in range(B):
        #     topk_inds_tmp = topk_inds[i]
        #     # [N, K]
        #     v_feat_tmp = v_feat[i]
        #     # [HW, C]
        #     topk_feats.append(v_feat_tmp[topk_inds_tmp])
        # topk_feats = torch.stack(topk_feats)
        # [B, N, K, C]
        obj_feat = proposal_feat.unsqueeze(2)
        # [B, N, 1, C]
       # topk_feats = topk_feats.reshape(B*N, K, C)
        obj_feat = obj_feat.reshape(B*N, 1, C)
       # topk_feats = topk_feats.transpose(0, 1)
        # [B*N, K, C]
        obj_feat = obj_feat.transpose(0, 1)
        # [B*N, 1, C]
      #  topk_feats=topk_feats.reshape(-1,B,N,C)
        obj_feat = obj_feat.reshape(B,N,C)
        
     
        # [B, N, K]
        #____
        #____
        # c=summary( self.f_atten ,obj_feat,topk_feats)
        # print(c)
       # topk_feats=topk_feats.mean(dim=0)
       # topk_feats=topk_feats.unsqueeze(dim=0)
        f_tmp = self.f_atten(obj_feat,topk_feats)
        topk_feats= topk_feats + self.f_dropout(f_tmp)
        topk_feats = self.f_atten_norm(topk_feats)
        f_tmp = self.k_atten( obj_feat,topk_feats)
        topk_feats =  topk_feats+ self.k_dropout(f_tmp)
        topk_feats = self.k_atten_norm( topk_feats)
        topk_feats = topk_feats.permute(1, 0, 2)

        k_tmp = self.s_atten(query = topk_feats, key = topk_feats, value = topk_feats )
        topk_feats = topk_feats + self.s_dropout(k_tmp)
        topk_feats = self.s_atten_norm(topk_feats.permute(1, 0, 2))
        # flops, params = profile(self.f_atten, inputs=(obj_feat,topk_feats)) 
        
        # print(flops)
        #____
       # obj_feat = self.topk_attn(obj_feat, topk_feats)
   #     obj_feat = obj_feat + self.f_dropout(f_tmp)
        
    #    obj_feat = self.f_atten_norm( obj_feat)


     #   obj_feat = self.topk_norm(obj_feat)
        
        #____
        # f_tmp = self.k_atten(obj_feat,topk_feats)
        # flops, params = profile(self.k_atten, inputs=(obj_feat,topk_feats)) 
        
        # print(flops)
        
        # obj_feat =  obj_feat + self.k_dropout(f_tmp)
 
        
        #______
        
       # topk_feats = topk_feats.transpose(0, 1)
        topk_feats = topk_feats.reshape(B, N, 1, C).squeeze(2)
        topk_feats = self.ffn_norm_pre(self.ffn_pre(topk_feats))
        # [B, N, C]

        mask_feat = topk_feats

        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)
        mask_feat = self.fc_mask(mask_feat)
        # [B, N, C, K*K] -> [B*N, C, K, K]

    
        mask_x = x
        # new_mask_preds: [B, C, H//8, W//8]
        new_mask_preds = torch.einsum('bchw,bnc->bnhw', mask_x, mask_feat)

        if self.mask_transform_stride == 2:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)


        return new_mask_preds,mask_feat
