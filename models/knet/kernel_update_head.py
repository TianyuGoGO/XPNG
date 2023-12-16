import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiheadattention import (MultiheadAtten, Ffn)
from .kernel_updator import KernelUpdator
from timm.models.layers import trunc_normal_
from .attention import SelfAttention,CrossAttention,AoaAttention
#from memory_profiler import profile
import math
import copy
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding= True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image

    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j

    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    #returns a relational embedding for each pair of bboxes, with dimension = dim_g
    #follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size = f_g.size(0)

    cx, cy,w, h = torch.chunk(f_g, 4, dim=-1)
    
    

    w =w + 1.
    h =h+ 1.

    #cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat
        position_mat=position_mat.cuda()

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return(embedding)
class BoxMultiHeadedAttention(nn.Module):
    '''
    Self-attention layer with relative position weights.
    Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
    '''
    
    def __init__(self, h, d_model, trignometric_embedding=True, legacy_extra_skip=False, dropout=0.1):
        "Take in model size and number of heads."
        super(BoxMultiHeadedAttention, self).__init__()

        assert d_model % h == 0
        self.trignometric_embedding=trignometric_embedding
        self.legacy_extra_skip = legacy_extra_skip

        # We assume d_v always equals d_k
        self.h = h
        self.d_k = d_model // h
        if self.trignometric_embedding:
            self.dim_g = 64
        else:
            self.dim_g = 4
        geo_feature_dim = self.dim_g

        #matrices W_q, W_k, W_v, and one last projection layer
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.WGs = clones(nn.Linear(geo_feature_dim, 1, bias=True),8)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, input_box, mask=None):
        "Implements Figure 2 of Relation Network for Object Detection"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = input_box.size(0)

        #tensor with entries R_mn given by a hardcoded embedding of the relative position between bbox_m and bbox_n
        relative_geometry_embeddings = BoxRelationalEmbedding(input_box, trignometric_embedding= self.trignometric_embedding)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1,self.dim_g)

        # 1) Do all the linear projections in batch from d_model => h x d_k
       
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head),1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        # 2) Apply attention on all the projected vectors in batch.
        # x, self.box_attn = box_attention(query, key, value, relative_geometry_weights, mask=mask,
        #                          dropout=self.dropout)

        # # 3) "Concat" using a view and apply a final linear.
        # x = x.transpose(1, 2).contiguous() \
        #      .view(nbatches, -1, self.h * self.d_k)

        # # An extra internal skip connection is added. This is only
        # # kept here for compatibility with some legacy models. In
        # # general, there is no advantage in using it, as there is
        # # already an outer skip connection surrounding this layer.
        # if self.legacy_extra_skip:
        #     x = input_value + x

        return relative_geometry_weights

def calcualHW(mask):
    #mask h,w
    pos=torch.nonzero(mask)
    
    hh=pos[:,0]
    ww=pos[:,1]
   # hh=torch.sort(input, dim=-1)
    hh= torch.unique(hh,sorted=False)
    ww= torch.unique(ww,sorted=False)
    
    return len(hh),len(ww)

def calcualHW_directsub(mask):
    #mask h,w
    pos=torch.nonzero(mask)
    
    
    
    center_h=0
    center_w=0
    
    
    hh=pos[:,0]
    
    ww=pos[:,1]
    
    if hh.shape[0]!=0:
        hh=torch.max(hh)-torch.min(hh)
        center_h=(torch.max(hh)+torch.min(hh))/2
    else:
        hh=0
        
    if ww.shape[0]!=0:
        ww=torch.max(ww)-torch.min(ww)
        center_w=(torch.max(ww)+torch.min(ww))/2
    else:
        ww=0

    
    
#     hh_max=hh_max.item()
#     hh_min=hh_min.item()
    
    
#     ww_max=torch.max(ww,dim=0)[0].item()
#     ww_min=torch.min(ww,dim=0)[0].item()
    
#     # ww_max=ww_max.item()
#     # ww_min=ww_min.item()
    
#     hh=hh_max-hh_min
    
#     ww=ww_max-ww_min
#     # ww,_=torch.sort(ww, dim=-1)
#     # k=ww.shape[0]-1
#     # ww=ww[k].item()-ww[0].item()
    
    
#    # hh=hh[]
#     # hh= torch.unique(hh,sorted=True)
#     # ww= torch.unique(ww,sorted=True)

  #  print("LKKKKKKKKKKKK")
    return hh,ww,center_h,center_w
    

    
def calcualcentroid(attn_map,attn_mask):
    #attn_mask b n
    attn_maps = attn_map.clone().detach()
    attn_maps[attn_maps >= 0.5] = 1
    attn_maps[attn_maps < 0.5] = 0
    
    b=attn_maps.shape[0]
    

    h=attn_maps.shape[2]
    
    w=attn_maps.shape[3]
    
    
    attn_sum = torch.sum(torch.sum(attn_maps, dim=-1), dim=-1) + 1e-5
    i, j = torch.meshgrid(torch.arange(h * 1.0), torch.arange(w * 1.0))
    i, j = i.cuda(), j.cuda() # h, w


            # b, 4, h, w
    h_i = torch.einsum('bnhw, hw->bn', attn_maps, i) / attn_sum
    w_j = torch.einsum('bnhw, hw->bn', attn_maps, j) / attn_sum
    
  
    information=torch.zeros((b,64,4),dtype=torch.float32).cuda()
    

    for g in range(b):

        for m in range(64):
            
            if attn_mask[g][m]==0:
                break
            else:
                hh,ww=calcualHW(attn_maps[g,m])
             #   hh,ww,xx,yy=calcualHW_directsub(attn_maps[g,m])
                
                
                information[g,m,0]=h_i[g][m]
                information[g,m,1]=w_j[g][m]
                # information[g,m,0]=xx
                # information[g,m,1]=yy
                information[g,m,2]=hh
                information[g,m,3]=ww
    
    return information
        
        

# class AoaAttentionModel(nn.Module):
    def __init__(self, h=8, d_model=256, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_aoa=1, norm_q=0, dropout_aoa=0.3):
        super(AoaAttentionModel, self).__init__()
        assert d_model * scale % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model * scale // h 
        self.h = h
        self.attention=AoaAttention(256,256,256,8)
        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v

        # normalize the query?
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x:x
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)

        # output linear layer after the multi-head attention?
        self.output_layer = nn.Linear(d_model * scale, d_model)

        # apply aoa after attention?
        self.use_aoa = do_aoa
        if self.use_aoa:
            self.aoa_layer =  nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())
            # dropout to the input of AoA layer
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x:x

        if self.use_aoa or not use_output_layer:
            # AoA doesn't need the output linear layer
            del self.output_layer
            self.output_layer = lambda x:x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, value, key,attn_mask,attn_mask1,mask=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)

        query = self.norm(query)

        # Do all the linear projections in batch from d_model => h x d_k 
        if self.project_k_v == 0:
            query_ =  self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_, key_, value_ = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch. 
        x = self.attention(query_, key_, value_, attn_mask,attn_mask1, 
                            )

        # "Concat" using a view
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        if self.use_aoa:
            # Apply AoA 
            x = self.aoa_layer(self.dropout_aoa(torch.cat([x, query], -1)))
        
        x = self.output_layer(x)

        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x
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

    #@profile
    def forward(self, query, value):
#        assert query.shape == value.shape
        B, N, C = query.shape
      #  print(query.device)
        value=value.reshape(B,N,C)
        # dynamic depth-wise conv
        # dy_depth_conv_weight = self.depth_weight_linear(query).view(B, self.num_proposals, 1,self.kernel_size) # B, N, 1, K
        # dy_point_conv_weight = self.point_weigth_linear(query).view(B, self.num_proposals, self.num_proposals, 1)
      #  print(self.weight_linear.device)
        dy_conv_weight = self.weight_linear(query)
        dy_depth_conv_weight = dy_conv_weight[:, :, :self.kernel_size].view(B,self.num_proposals,1,self.kernel_size)
        dy_point_conv_weight = dy_conv_weight[:, :, self.kernel_size:].view(B,self.num_proposals,self.num_proposals,1)

        res = []
        
        value = value.unsqueeze(1)
     #   value = F.pad(value,(1,1),'constant',0)
        for i in range(B):
            # input: [1, N, C]
            # weight: [N, 1, K]
            # output: [1, N, C]
        
        
            
            out = F.relu(F.conv1d(input=value[i], weight=dy_depth_conv_weight[i], groups=N,padding='same'))
            # input: [1, N, C]
            # weight: [N, N, 1]
            # output: [1, N, C]
            out = F.conv1d(input=out, weight=dy_point_conv_weight[i],padding='same')

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
        # self.k_atten = DySepConvAtten()
        self.f_dropout = nn.Dropout(0.0)
        self.f_atten_norm = nn.LayerNorm(256)
        # self.k_dropout = nn.Dropout(0.0)
        # self.k_atten_norm = nn.LayerNorm(256)
        
        #_________
        self.zero_conv=nn.Conv1d(256,256,1,bias=True)
        
    
        
        
        w1=torch.zeros(self.zero_conv.weight.shape)
        w2=torch.zeros(self.zero_conv.bias.shape)
        self.zero_conv.weight= nn.Parameter(w1)
        self.zero_conv.bias=nn.Parameter(w2)
        
        
        self.one_conv=nn.Conv1d(256,256,1,bias=True)
        
        w1=torch.ones(self.one_conv.weight.shape)
        w1=torch.eye(256)
        w1=w1.unsqueeze(dim=-1)
        w2=torch.zeros(self.one_conv.bias.shape)
        self.one_conv.weight= nn.Parameter(w1)
        self.one_conv.bias=nn.Parameter(w2)
        
        
        #___________
        self.s_atten = SelfAttention(256,256,256,8)
        self.s_dropout = nn.Dropout(0.0)
        self.s_atten_norm = nn.LayerNorm(256)
        
        
        # self.c_atten =  CrossAttention(256,256,256,8)
        # self.c_dropout = nn.Dropout(0.0)
        # self.c_atten_norm = nn.LayerNorm(256)
        
        
        
        self.position = BoxMultiHeadedAttention(8,64)
        
        
         
        #___
    #图片 文本 掩码
  
    def forward(self, x, proposal_feat, mask_preds, attn_masks,ori_text_feat,mask_shape=None):
        
        
        attn_mask,attn_mask1, aattn_mask=attn_masks

       # K = self.num_points
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
       # v_feat = x.flatten(-2).transpose(1, 2)
        
        #__gtyupdated
     #  topk_feats=torch.
        gather_mask=gather_mask.sigmoid()
        gather_mask=(gather_mask>0.5).float()
        
        
    
        
        position_information=calcualcentroid(gather_mask,aattn_mask)
     
        position_information=self.position(position_information)
       
       
        #temp_mask= gather_mask
       # gather_mask[gather_mask<0.5]=0
        #temp_mask=(temp_mask>0.5).float()
        
        #gather_mask=gather_mask*temp_mask
      
        #temp_mask= gather_mask.detach()
     #   temp_mask[temp_mask<0.5]=0
       # gather_mask=gather_mask.float()
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
        
        # f_tmp = self.k_atten( obj_feat,topk_feats)
        # topk_feats =  topk_feats+ self.k_dropout(f_tmp)
        # topk_feats = self.k_atten_norm( topk_feats)
        
    #   topk_feats = topk_feats.permute(1, 0, 2)

        k_tmp = self.s_atten(topk_feats,topk_feats,topk_feats,attn_mask,attn_mask1,position_information)
        
        topk_feats = topk_feats + self.s_dropout(k_tmp)
        topk_feats = self.s_atten_norm(topk_feats)
        
        
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
        
        # topk_feats = self.one_conv(topk_feats.transpose(2,1))
        # topk_feats = topk_feats.transpose(2,1)+self.zero_conv(proposal_feat.transpose(2,1)).transpose(2,1)
        
        topk_feats = self.zero_conv(topk_feats.transpose(2,1))
        topk_feats = topk_feats.transpose(2,1)+self.one_conv(proposal_feat.transpose(2,1)).transpose(2,1)
        
        
      #  topk_feats=topk_feats+self.text_conv(ori_text_feat.transpose(2,1)).transpose(2,1)
        # [B, N, C]

        mask_feat = topk_feats
        #mask_feat =mask_feat

        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)
        mask_feat = self.fc_mask(mask_feat)
        mask_x = x
     #   b=mask_x.shape[0]
      #  c=mask_x.shape[1]
      #  h=mask_x.shape[2]
      #  w=mask_x.shape[3]
      #  mask_x=mask_x.reshape(b,c,-1).transpose(2,1).contiguous()
      #  mask_x_tmp = self.c_atten(mask_x,mask_feat,mask_feat,attn_mask2)
        
        
      #  mask_x = mask_x + self.s_dropout( mask_x_tmp)
     #   mask_x = self.s_atten_norm(mask_x)
        
      #  mask_x=mask_x.transpose(2,1).contiguous().reshape(b,c,h,w)
        # [B, N, C, K*K] -> [B*N, C, K, K]

    
      
        # new_mask_preds: [B, C, H//8, W//8]
        new_mask_preds = torch.einsum('bchw,bnc->bnhw', mask_x, mask_feat)

        if self.mask_transform_stride == 2:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)


        return new_mask_preds,mask_feat
