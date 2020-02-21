from onmt.modules import GlobalAttention
from torch import nn
from torch.nn import Parameter
import torch


class SOTWModel(nn.Module):
    def forward(self, current_sotw, new_obs):
        raise NotImplementedError


class SymmetricAttentionSOTW(SOTWModel):
    def __init__(self, attn_dim: int):
        super(SOTWModel, self).__init__()
        self.sotw_attention = GlobalAttention(dim=attn_dim, coverage=False, attn_type="mlp",
                                              attn_func="softmax")
        self.mixing_proportion = Parameter(torch.rand(1).cuda())

    def forward(self,
                current_sotw: torch.Tensor,
                new_obs: torch.Tensor) -> torch.Tensor:
        """
        Assumees current_sotw and new_obs have the same shapes: [batch_size x num_world_dimensions x hidden_dim]
        """
        assert current_sotw.size(0) == new_obs.size(0)
        assert current_sotw.size(1) == new_obs.size(1)
        assert current_sotw.size(2) == new_obs.size(2)

        new_sotw = (1 - self.mixing_proportion) * \
                   self.sotw_attention(new_obs.contiguous(), current_sotw.contiguous())[
                       0].transpose(0, 1) + \
                   self.mixing_proportion * \
                   self.sotw_attention(current_sotw.contiguous(), new_obs.contiguous())[
                       0].transpose(0, 1)
        return new_sotw


class AsymmetricAttentionSOTW(SOTWModel):
    def __init__(self, attn_dim: int):
        super(SOTWModel, self).__init__()
        self.sotw_attention = GlobalAttention(dim=attn_dim, coverage=False, attn_type="mlp",
                                              attn_func="softmax")
        self.mixing_proportion = Parameter(torch.rand(1).cuda())

    def forward(self,
                current_sotw: torch.Tensor,
                new_obs: torch.Tensor) -> torch.Tensor:
        """
        Assumees current_sotw and new_obs have the same shapes: [batch_size x num_world_dimensions x hidden_dim]
        """
        assert current_sotw.size(0) == new_obs.size(0)
        assert current_sotw.size(1) == new_obs.size(1)
        assert current_sotw.size(2) == new_obs.size(2)

        new_sotw = (1 - self.mixing_proportion) * current_sotw + \
                   self.mixing_proportion * \
                   self.sotw_attention(new_obs.contiguous(), current_sotw.contiguous())[
                       0].transpose(0, 1)
        return new_sotw


class WightedSumSOTW(SOTWModel):
    def __init__(self, attn_dim: int):
        super(SOTWModel, self).__init__()
        self.mixing_proportion = Parameter(torch.rand(1).cuda())

    def forward(self,
                current_sotw: torch.Tensor,
                new_obs: torch.Tensor) -> torch.Tensor:
        """
        Assumees current_sotw and new_obs have the same shapes: [batch_size x num_world_dimensions x hidden_dim]
        """
        assert current_sotw.size(0) == new_obs.size(0)
        assert current_sotw.size(1) == new_obs.size(1)
        assert current_sotw.size(2) == new_obs.size(2)

        new_sotw = (1 - self.mixing_proportion) * current_sotw + self.mixing_proportion * new_obs
        return new_sotw
