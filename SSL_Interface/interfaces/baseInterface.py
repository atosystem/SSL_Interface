import torch
import torch.nn
from typing import List, Tuple
import numpy as np
from SSL_Interface.configs import BaseInterfaceConfig
import logging
import torch.nn.functional as F
import math

logger = logging.getLogger(__name__)

__all__ = ["BaseInterface"]


class BaseInterface(torch.nn.Module):
    def __init__(self, config: BaseInterfaceConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

        self._output_dim = self.config.upstream_feat_dim

    def log_params(self, _params=None):
        if _params is None:
            _params = self.parameters()
        model_parameters = filter(lambda p: p.requires_grad, _params)
        params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info(f"{self.__class__.__name__} has {params:,} trainable parameters")

    @property
    def output_dim(self) -> int:
        """
        The hidden size of the final weighted-sum output
        """
        return self._output_dim

    def forward(self, stacked_hs: torch.FloatTensor):

        assert stacked_hs.shape[0] == self.config.upstream_layer_num
        assert stacked_hs.shape[-1] == self.config.upstream_feat_dim

        if self.config.normalize:
            stacked_hs = F.layer_norm(stacked_hs, (stacked_hs.shape[-1],))

        # stacked_hs (layer, batch_size, seq_len, hidden_size)

        stacked_hs = self.interface_forward(stacked_hs)
        return stacked_hs

    def interface_forward(self, stacked_hs: torch.FloatTensor) -> torch.FloatTensor:
        """Main function for interface impomentation

        Args:
            stacked_hs (torch.FloatTensor): (layer, batch_size, seq_len, hidden_size)

        """
        raise NotImplementedError()
