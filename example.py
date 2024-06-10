import torch
import SSL_Interface
import SSL_Interface.configs
import SSL_Interface.interfaces


WS_Interface = SSL_Interface.interfaces.WeightSumInterface(
    SSL_Interface.configs.WeightedSumInterfaceConfig(
        upstream_feat_dim=768,
        upstream_layer_num=13,
        normalize=False,
    )
)

HConv_Interface = SSL_Interface.interfaces.HierarchicalConvInterface(
    SSL_Interface.configs.HierarchicalConvInterfaceConfig(
        upstream_feat_dim=768,
        upstream_layer_num=13,
        normalize=False,
        conv_kernel_size=5,
        conv_kernel_stride=3,
        output_dim=768
    )
)



layer, batch_size, seq_len, hidden_size = 13,8,100,768

feats = torch.randn(layer, batch_size, seq_len, hidden_size)

feats = feats.cuda()
WS_Interface = WS_Interface.cuda()
HConv_Interface = HConv_Interface.cuda()

out = WS_Interface(feats)
print(out.shape)
# torch.Size([8, 100, 768])

out = HConv_Interface(feats)
print(out.shape)
# torch.Size([8, 100, 768])