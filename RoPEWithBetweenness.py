
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BetweennessBase(nn.Module):

    def __init__(self, dim, adjustment_scale=0.1):
        super().__init__()
        self.dim = dim
        self.adjustment_scale = adjustment_scale
        self.content_proj = nn.Linear(dim, dim, bias=False)
        self.betweenness_gate = nn.Parameter(torch.ones(1) * 0.5)
        self.cp_dim = None  # To track the last dimension we initialized for

    def get_position_adjustments(self, x):

        betweenness = self.compute_betweenness(x)
        adj = self.betweenness_gate * (betweenness - 0.5) * self.adjustment_scale
        return adj

class TripletBetweenness(BetweennessBase):

    def compute_betweenness(self, x):
        orig_dim = x.dim()
        if orig_dim == 4:
            batch, seq_len, heads, dim = x.shape
            x_flat = x.transpose(1, 2).reshape(batch * heads, seq_len, dim)
            has_heads = True
            
        else:
            batch, seq_len, dim = x.shape
            x_flat = x
            has_heads = False
            heads = 1
        
        if not hasattr(self, 'cp_dim') or self.cp_dim != dim:
            self.content_proj = nn.Linear(dim, dim, bias=False).to(x.device, x.dtype)
            self.cp_dim = dim
            
        content = self.content_proj(x_flat)
        token_i = content.unsqueeze(2)
        token_j = content.unsqueeze(1)
        distances = torch.norm(token_i - token_j, dim=-1)
        betweenness = torch.zeros(x_flat.shape[0], seq_len, device=x.device)
        
        if seq_len <= 2:
            if has_heads:
                return betweenness.view(batch, heads, seq_len).transpose(1, 2)
            elif orig_dim <= 3:
                return betweenness.view(batch, seq_len)
            return betweenness

        for i in range(seq_len - 2):
            direct = distances[:, i, i+2]
            path = distances[:, i, i+1] + distances[:, i+1, i+2]
            between_score = torch.relu(1.0 - (path - direct) / torch.clamp(direct, min=1e-6))
            betweenness[:, i+1] += between_score

        betweenness = betweenness / (seq_len - 2)
      
        if has_heads:
            betweenness = betweenness.view(batch, heads, seq_len).transpose(1, 2)
        elif orig_dim <= 3:
            betweenness = betweenness.view(batch, seq_len)
            
        return betweenness

class MultiGapBetweenness(BetweennessBase):

    def __init__(self, dim, adjustment_scale=0.1, max_gap=8):
        super().__init__(dim, adjustment_scale)
        self.max_gap = max_gap

    def compute_betweenness(self, x):

        orig_dim = x.dim()
        if orig_dim == 4:
            batch, seq_len, heads, dim = x.shape
            x_flat = x.transpose(1, 2).reshape(batch * heads, seq_len, dim)
            has_heads = True
       
        else:
            batch, seq_len, dim = x.shape
            x_flat = x
            has_heads = False
            heads = 1

        if not hasattr(self, 'cp_dim') or self.cp_dim != dim:
            self.content_proj = nn.Linear(dim, dim, bias=False).to(x.device, x.dtype)
            self.cp_dim = dim
  
        c = self.content_proj(x_flat)
        dist = torch.cdist(c, c, p=2)
        btw = torch.zeros(x_flat.shape[0], seq_len, device=x.device)
        
        if seq_len <= 2:
            if has_heads:
                return btw.view(batch, heads, seq_len).transpose(1, 2)
            elif orig_dim <= 3:
                return btw.view(batch, seq_len)
            return btw
        
        max_gap = min(seq_len, self.max_gap)
        total_paths = 0
        
        for gap in range(2, max_gap + 1):
            pairs_count = seq_len - gap
            if pairs_count <= 0:
                continue
                

            src_indices = torch.arange(pairs_count, device=x.device)
            tgt_indices = src_indices + gap

            for b in range(x_flat.shape[0]):
                direct_distances = dist[b, src_indices, tgt_indices]
                for mid_offset in range(1, gap):
                    mid_indices = src_indices + mid_offset
                    paths = dist[b, src_indices, mid_indices] + dist[b, mid_indices, tgt_indices]
                    scores = torch.relu(1.0 - (paths - direct_distances) / 
                                       torch.clamp(direct_distances, min=1e-6))
                    btw[b].index_add_(0, mid_indices, scores)
     
            total_paths += pairs_count * (gap - 1)
        
        if total_paths > 0:
            btw = btw / total_paths

        if has_heads:
            btw = btw.view(batch, heads, seq_len).transpose(1, 2)
        elif orig_dim <= 3:
            btw = btw.view(batch, seq_len)
            
        return btw

class BetweennessRoPE(nn.Module):

    def __init__(self, dim, max_seq_len=2048, adjustment_scale=0.1):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.adjustment_scale = adjustment_scale
        self.content_proj = nn.Linear(dim, dim)
        self.betweenness_gate = nn.Parameter(torch.ones(1) * 0.5)
        
        self.register_buffer(
            "base_freqs", 
            1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        )

    def precompute_freqs_cis(self, seq_len, device):
        t = torch.arange(seq_len, device=device)
        freqs = torch.outer(t, self.base_freqs.to(device))
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return freqs_cos.unsqueeze(0).unsqueeze(2), freqs_sin.unsqueeze(0).unsqueeze(2)

    def compute_betweenness(self, x):

        shape = x.shape
        orig_dim = x.dim()
        
        if orig_dim == 4:
            batch, seq_len, heads, dim = x.shape
            x = x.transpose(1, 2).reshape(batch * heads, seq_len, dim)
        else:
            batch, seq_len = x.shape[:2]
        
        content = self.content_proj(x)
        
        token_i = content.unsqueeze(2)
        token_j = content.unsqueeze(1)
        distances = torch.norm(token_i - token_j, dim=-1)
        
        betweenness = torch.zeros(x.shape[0], seq_len, device=x.device)
        
        for i in range(seq_len-2):
            direct = distances[:, i, i+2]
            path = distances[:, i, i+1] + distances[:, i+1, i+2]
            between_score = torch.relu(1.0 - (path - direct)/torch.clamp(direct, min=1e-6))
            betweenness[:, i+1] += between_score
        
        if seq_len > 2:
            betweenness = betweenness / (seq_len - 2)
        
        if orig_dim == 4:
            betweenness = betweenness.view(batch, heads, seq_len).transpose(1, 2)
        
        return betweenness

    def apply_rope(self, x, freqs_cos, freqs_sin, betweenness=None):

        seq_len = x.shape[1]
        if betweenness is None:
            x_rope = x.float()
            
            x_out = torch.zeros_like(x_rope)
            for i in range(seq_len):
                x_out[:, i, :, 0::2] = x_rope[:, i, :, 0::2] * freqs_cos[i] - x_rope[:, i, :, 1::2] * freqs_sin[i]
                x_out[:, i, :, 1::2] = x_rope[:, i, :, 1::2] * freqs_cos[i] + x_rope[:, i, :, 0::2] * freqs_sin[i]
            
            return x_out.type_as(x)
        
        pos_adjustments = (betweenness - 0.5) * self.adjustment_scale
        
        x_out = torch.zeros_like(x, dtype=torch.float)
        for i in range(seq_len):
            adj_pos = torch.clamp(i + pos_adjustments[:, i:i+1], 0, seq_len-1)
            
            idx_lo = torch.floor(adj_pos).long()
            idx_hi = torch.ceil(adj_pos).long()
            frac = (adj_pos - idx_lo).unsqueeze(-1)
            cos_interp = (1 - frac) * freqs_cos[idx_lo] + frac * freqs_cos[idx_hi]
            sin_interp = (1 - frac) * freqs_sin[idx_lo] + frac * freqs_sin[idx_hi]
            
            cos_interp = cos_interp.squeeze(1)
            sin_interp = sin_interp.squeeze(1)
            x_out[:, i, :, 0::2] = x[:, i, :, 0::2] * cos_interp - x[:, i, :, 1::2] * sin_interp
            x_out[:, i, :, 1::2] = x[:, i, :, 1::2] * cos_interp + x[:, i, :, 0::2] * sin_interp
        
        return x_out.type_as(x)
        
    def forward(self, x):
        batch, seq_len = x.shape[:2]
        betweenness = self.compute_betweenness(x)
        freqs_cos, freqs_sin = self.precompute_freqs_cis(seq_len, device = x.device)
        return self.apply_rope(x, freqs_cos, freqs_sin, betweenness)

class BetweennessAttention(nn.Module):

    def __init__(self, dim, heads, dim_head=64):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        
        self.q_proj = nn.Linear(dim, inner_dim)
        self.k_proj = nn.Linear(dim, inner_dim)
        self.v_proj = nn.Linear(dim, inner_dim)
        self.output = nn.Linear(inner_dim, dim)
        
        self.rope = BetweennessRoPE(dim_head, adjustment_scale=0.1)
        
    def forward(self, x, mask=None):
        batch, seq_len = x.shape[:2]
        
        q = self.q_proj(x).reshape(batch, seq_len, self.heads, self.dim_head)
        k = self.k_proj(x).reshape(batch, seq_len, self.heads, self.dim_head)
        v = self.v_proj(x).reshape(batch, seq_len, self.heads, self.dim_head)
        
        q_rope = self.rope(q)
        k_rope = self.rope(k)
        
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q_rope, k_rope.transpose(-1, -2)) / math.sqrt(self.dim_head)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.output(output)


def apply_to_rope(rope_func, x, positions, betweenness_module):
    adjustments = betweenness_module.get_position_adjustments(x)
    adjusted_positions = positions + adjustments
    return rope_func(x, adjusted_positions)
