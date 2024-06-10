import torch
import torch.nn
from typing import List, Tuple
from SSL_Interface.interfaces.baseInterface import BaseInterface
from SSL_Interface.configs import HierarchicalConvInterfaceConfig
import logging
import torch.nn.functional as F
import math
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["HierarchicalConvInterface"]


class HierarchicalConvInterface(BaseInterface):
    def __init__(
        self, config: HierarchicalConvInterfaceConfig, *args, **kwargs
    ) -> None:
        super().__init__(config, *args, **kwargs)
        self.config: HierarchicalConvInterfaceConfig

        # self.config.output_dim = self.config.get("output_dim", None)
        if self.config.output_dim is None:
            logger.error(f"[HConv] - Please specify output_dim")
            exit(1)
        self._output_dim = self.config.output_dim

        L = self.config.upstream_layer_num
        if L > 1:
            conv_kernel_stride = self.config.conv_kernel_stride
            conv_kernel_size = self.config.conv_kernel_size

            estimated_num_convs = math.floor(np.log(L) / np.log(conv_kernel_stride))

            self.transforms = torch.nn.Sequential()

            for conv_i in range(estimated_num_convs):
                _padding = math.floor(conv_kernel_size / 2)
                _dilation = 1
                L = int(
                    np.floor(
                        (L + 2 * _padding - _dilation * (conv_kernel_size - 1) - 1)
                        / conv_kernel_stride
                        + 1
                    )
                )

                if conv_i == estimated_num_convs - 1:
                    _in_channels = self.config.upstream_feat_dim
                    _out_channels = math.ceil(self.config.upstream_feat_dim // L)
                else:
                    _in_channels = self.config.upstream_feat_dim
                    _out_channels = self.config.upstream_feat_dim

                self.transforms.append(
                    torch.nn.Conv1d(
                        in_channels=_in_channels,
                        out_channels=_out_channels,
                        kernel_size=conv_kernel_size,
                        stride=conv_kernel_stride,
                        dilation=_dilation,
                        padding=_padding,
                    )
                )
                self.transforms.append(torch.nn.ReLU())

            logger.info(
                f"[ConvOnLayerAutoLogFeaturizer] - Conv Transform ({sum(p.numel() for p in self.transforms.parameters())/1000000:.2f}M) params)"
            )
            logger.info(
                f"[ConvOnLayerAutoLogFeaturizer] - Concat last conv output: {math.ceil( self.config.upstream_feat_dim // L)} * {L} = {math.ceil( self.config.upstream_feat_dim // L) * L}"
            )
            if (
                not self.config.output_dim
                == math.ceil(self.config.upstream_feat_dim // L) * L
            ):
                logger.error(
                    f"[ConvOnLayerFeaturizer] - Specfied output dim({math.ceil(self.config.upstream_feat_dim // L) * L}) != {output_dim}"
                )
                exit(1)
            self._output_size = self.config.output_dim

            logger.info(self.transforms)
            logger.info(
                f"[ConvOnLayerAutoLogFeaturizer] - Normalize upstream=({self.config.normalize})"
            )
        else:
            logger.error("Upstream model must has more layer than 1")
            exit(1)

        self.log_params(self.parameters())

    def interface_forward(self, stacked_hs: torch.FloatTensor) -> torch.FloatTensor:
        _nLayers, _bsz, max_len, _hiddim = stacked_hs.shape

        # -> _bsz, max_len, _hiddim, _nLayers -> _bsz * max_len, _hiddim, _nLayers
        stacked_hs = stacked_hs.permute(1, 2, 3, 0).view(-1, _hiddim, _nLayers)

        # conv transforms ->  _bsz * max_len, dim', layer'
        stacked_hs = self.transforms(stacked_hs)

        stacked_hs = stacked_hs.view(_bsz * max_len, -1)
        stacked_hs = stacked_hs.view(_bsz, max_len, -1)

        return stacked_hs
