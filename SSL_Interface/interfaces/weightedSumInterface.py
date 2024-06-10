# ref: https://github.com/s3prl/s3prl
import torch
import torch.nn
from typing import List, Tuple

from SSL_Interface.interfaces.baseInterface import BaseInterface
from SSL_Interface.configs import WeightedSumInterfaceConfig
import logging

import torch.nn.functional as F


logger = logging.getLogger(__name__)

__all__ = ["WeightSumInterface"]


class WeightSumInterface(BaseInterface):
    def __init__(self, config: WeightedSumInterfaceConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.config: WeightedSumInterfaceConfig

        self.weighte_sum_weights = torch.nn.Parameter(
            torch.zeros(self.config.upstream_layer_num)
        )

        self.log_params()

    def interface_forward(
        self,
        stacked_hs: torch.FloatTensor,
    ) -> torch.FloatTensor:
        _, *origin_shape = stacked_hs.shape
        stacked_hs = stacked_hs.view(self.config.upstream_layer_num, -1)
        norm_weights = F.softmax(self.weighte_sum_weights, dim=-1)
        weighted_hs = (norm_weights.unsqueeze(-1) * stacked_hs).sum(dim=0)
        weighted_hs = weighted_hs.view(*origin_shape)

        return weighted_hs
