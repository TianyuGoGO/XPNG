import torch.nn as nn
import torch.nn.functional as F
from .kernel_update_head_no_stage5_stage3_one1_zhixin import KernelUpdateHead

from thop import profile
from memory_profiler import profile
class KernelIterHead(nn.Module):

    def __init__(self, 
        num_stages=5,
        num_points=100,
    ):
        super(KernelIterHead, self).__init__()
        self.num_stages = num_stages
        self.mask_head = nn.ModuleList()
        for i in range(num_stages):
            self.mask_head.append(KernelUpdateHead(num_points=num_points))


    def _mask_forward(self, stage, x, kernels, mask_preds,attn_masks,ori_text_feat):
        mask_head = self.mask_head[stage]
        # flops, params = profile(mask_head, inputs=(x, kernels, mask_preds)) 
        
        # print(flops)
        mask_preds, kernels = mask_head(
            x, kernels, mask_preds,attn_masks,ori_text_feat)

        return mask_preds, kernels

    #图片 文本 掩码
    #@profile
    def forward_train(self, x, proposal_feats, mask_preds,attn_masks,):
        
        # object_feats = proposal_feats
        kernels = proposal_feats
        all_stage_mask_results = []
        
        ori_text_feat=proposal_feats
        for stage in range(self.num_stages):
            mask_preds, kernels = self._mask_forward(stage, x, kernels,
                                              mask_preds,attn_masks,ori_text_feat)
            all_stage_mask_results.append(mask_preds)

        return all_stage_mask_results

    