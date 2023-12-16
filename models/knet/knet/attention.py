import math
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

from scipy.spatial import distance
def get_relative_pos(H, W):
       
        col, row = np.meshgrid(np.arange(0,W),np.arange(0,H))
        coord = np.stack((row,col), axis=2)
        coord=coord.reshape(-1,2)
        coord=coord.tolist()
        dis_matrix=distance.cdist(coord, coord, 'euclidean')
        dis_matrix=dis_matrix.astype(np.float32)
        dis_matrix = torch.tensor(dis_matrix)
        dis_matrix =torch.clamp(dis_matrix,0,2)
        return dis_matrix
class SelfAttention(nn.Module):
    def __init__(self, input_dim, dim_q, dim_v, n_head):
        super(SelfAttention, self).__init__()
        # dim_q = dim_k
        self.dim_q, self.dim_k, self.dim_v, self.n_head = dim_q, dim_q, dim_v, n_head
         
        if self.dim_k % n_head != 0:
            print("dim_k can't divide n_head")
        if self.dim_v % n_head != 0:
            print("dim_v  can't divide n_head")
        
   
        self.n_head=n_head
        self.wq = nn.Linear(input_dim, dim_q)
        self.wk = nn.Linear(input_dim, dim_q)
        self.wv = nn.Linear(input_dim, dim_v)
        self.wo = nn.Linear(dim_v, dim_v)
        self._norm_fact = 1 / math.sqrt(dim_q/n_head)
     #   self.init_weights()
        self.w=nn.Parameter(torch.ones(n_head, 1, 1))
        self.dropout = nn.Dropout(p=0.1)
 #   def init_weights(self):
 #       nn.init.xavier_uniform_(self.wq.weight)
 #       nn.init.xavier_uniform_(self.wk.weight)
  #      nn.init.xavier_uniform_(self.wv.weight)
  #      nn.init.xavier_uniform_(self.wo.weight)
   #     nn.init.constant_(self.wq.bias, 0)
    #    nn.init.constant_(self.wk.bias, 0)
    #    nn.init.constant_(self.wv.bias, 0)
   #     nn.init.constant_(self.wo.bias, 0)

    def forward(self, q, k, v,attn_mask,attn_mask1):
     
        
        # q: B Nq input_dim
        self.b=q.shape[0]
        Q = self.wq(q)  # B Nq input_dim ->B Nq dim_q
        K = self.wk(k)  # B Nk dim_q
        V = self.wv(v)  # B NV dim_q
        
        
   
        Q = Q.reshape(Q.shape[0], self.n_head, Q.shape[1], self.dim_k // self.n_head)

        # B Nq dim_q->  B head  Nq dim_q/head = [batch_size,n_head,seq_len,dim_q]
        K = K.reshape(K.shape[0], self.n_head, K.shape[1], self.dim_k // self.n_head)
        V = V.reshape(V.shape[0], self.n_head, V.shape[1], self.dim_v // self.n_head)

               
        # Q: [batch_size,n_head,seq_len,dim_q]
        # K: [batch_size,n_head,seq_len,dim_k]
        # V: [batch_size,n_head,seq_len,dim_v]
        # print(f' Q.shape:{Q.shape} , K.shape: {K.shape} , V.shape:{V.shape}')

        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self._norm_fact         # batch_size ,n_head, seq_len , seq_len
        
       # attn_mask=attn_mask.reshape(Q.shape[0],Q.shape[1],attn_mask.shape[1],attn_mask.shape[2])
        attention =attention +attn_mask  # attn_mask   b n seq_len,seq_len
        
        attention=torch.nn.functional.softmax(attention, dim=-1) 
        
        attention=attention*attn_mask1
        
        value = torch.matmul(attention, V).reshape(Q.shape[0], Q.shape[2], -1)         # [ batch_size ,n_head, seq_len , seq_len ]  ,[b,n,seq_len,dim_v]
   
       
        
        value=self.wo(value)
        
        return value
class CrossAttention(nn.Module):
    def __init__(self, input_dim, dim_q, dim_v, n_head):
        super(CrossAttention, self).__init__()
        # dim_q = dim_k
        self.dim_q, self.dim_k, self.dim_v, self.n_head = dim_q, dim_q, dim_v, n_head
         
        if self.dim_k % n_head != 0:
            print("dim_k can't divide n_head")
        if self.dim_v % n_head != 0:
            print("dim_v  can't divide n_head")

        self.n_head=n_head
        
        
        
        self.wq = nn.Linear(input_dim, dim_q)
        self.wk = nn.Linear(input_dim, dim_q)
        self.wv = nn.Linear(input_dim, dim_v)
        self.wo = nn.Linear(dim_v, dim_v)
        self._norm_fact = 1 / math.sqrt(dim_q)
      #  self.init_weights()
        self.dropout = nn.Dropout(p=0.1)
    def init_weights(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.wo.weight)
        nn.init.constant_(self.wq.bias, 0)
        nn.init.constant_(self.wk.bias, 0)
        nn.init.constant_(self.wv.bias, 0)
        nn.init.constant_(self.wo.bias, 0)

    def forward(self, q, k, v,attn_mask):

        
        # q: B Nq input_dim
        
        self.b=q.shape[0]
        Q = self.wq(q)  # B Nq input_dim ->B Nq dim_q
        K = self.wk(k)  # B Nk dim_q
        V = self.wv(v)  # B NV dim_q
        

        Q = Q.reshape(Q.shape[0], self.n_head, Q.shape[1], self.dim_k // self.n_head)

        # B Nq dim_q->  B head  Nq dim_q/head = [batch_size,n_head,n_q,dim_q/n_head]
        
        K = K.reshape(K.shape[0], self.n_head, K.shape[1], self.dim_k // self.n_head)
        V = V.reshape(V.shape[0], self.n_head, V.shape[1], self.dim_v // self.n_head)
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self._norm_fact         # batch_size ,n_head, seq_len , seq_len
     #   attn_mask=attn_mask.reshape(Q.shape[0],Q.shape[1],attn_mask.shape[1],attn_mask.shape[2])
      
        attention =attention +attn_mask  # attn_mask   b n seq_len,seq_len
       
       
        attention=torch.nn.functional.softmax(attention, dim=-1) 
        
        #attention=attention*attn_mask1
      #  print("attention is ",attention[0][0][0])
        
        attention=self.dropout(attention)

         #B head Nq Nk  *  B head Nk 768/head     -> B head Nq 768/head  -> B Nq dv
        value = torch.matmul(attention, V).reshape(Q.shape[0], Q.shape[2], -1)
        
        value=self.wo(value)
        return value

