import torch
from einops import rearrange, repeat
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
# from xception import xception
from network.xception import xception
from network.vit_pytorch import Transformer

"""  
    function: 提取特征图
"""
class CNNStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = xception(pretrained=False)
        # self.load_weight()  # 预训练模型
        self.entry_flow = nn.Sequential(*list(self.stem.children())[:8])  # xception的 Entry flow, 参数同预训练模型中
        # self.entry_flow = IntermediateLayerGetter(model=stem, return_layers={'block1':'shallow'})

    def load_weight(self):
        self.stem.fc = self.stem.last_linear
        del self.stem.last_linear
        state_dict = torch.load('/home/Users/laizhenqiang/ckpt/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        self.stem.load_state_dict(state_dict)

    def forward(self, input):
        shallow_maps = self.entry_flow(input)  #
        return shallow_maps

"""  
    Middle flow
"""
class LANet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=out_features, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


"""  
    Transformer
"""
class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, channels):
        super().__init__()
        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, channels+1, dim))
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2)
        )
    def forward(self, x):
        ## input shape: [-1, num_patchs, dim]
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]

        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])  # 选择cls_token
        x = self.mlp_head(x)
        return x

class CGTransformer(nn.Module):
    def __init__(self, att_nums, use_RDA=False):
        super().__init__()
        self.att_nums = att_nums
        self.use_RDA = use_RDA
        self.Stem = CNNStem()
        self.layers = nn.ModuleList([])
        # number of LANet
        for _ in range(self.att_nums):
            self.layers.append(
                LANet(728, 128)
            )
        # transformer and classfication
        self.c_transformer = CrossTransformer(dim=722, depth=12, heads=19, mlp_dim=1444, dropout=0.1, channels=728)
    # 随机选择一张图置0, 一个batch中同一LANet输出的图 or 随机LANet
    def RDA(self, feature_maps, possibility=0.8):
            ### 一定概率，使得1个batch的相同drop_dim的注意力图置0
            p = torch.rand((1))  # 生成一个随机概率p
            if p.item() < possibility:
                drop_dim = torch.randint(0, self.att_nums, (1,))  # 置0的注意力图dim
                feature_maps[:, drop_dim.item(), :, :] = 0
            return feature_maps

    def forward(self, x):
        att_maps = torch.tensor([], dtype=torch.float, device=x.device)
        feature_maps = self.Stem(x)  # 从CNN中提取的特征图 [-1, C, H, W]  N: 特征图个数
        # attention maps
        for id, layer in enumerate(self.layers):
            att_map = layer(feature_maps)
            att_maps = torch.cat((att_maps, att_map), dim= 1)  # [-1, att_nums, H, W]

        if self.use_RDA:
            # 使用随机置 attention map 0
            att_maps = self.RDA(att_maps)
        ## 取 所有注意力图中 最大的值 到一张图中
        max_att_maps = torch.max(att_maps, 1, keepdim=True)  # att_maps[0]:values [-1, 1, H, W] att_maps[1]:indices
        ## element-wised product
        x_out = max_att_maps[0] * feature_maps # [-1, C, H, W]
        
        x_out = rearrange(x_out, 'b c h w -> b c (h w)')
        feature_maps = rearrange(feature_maps, 'b c h w -> b c (h w)')
        ct_input = torch.cat((x_out, feature_maps), dim=2)
        outputs = self.c_transformer(ct_input)
        return outputs
        

if __name__ == '__main__':
    from torchsummary import summary
    import random
    from torch.backends import cudnn
    SEED = 12345
    # 设置随机种子
    random.seed(SEED)
    torch.manual_seed(SEED)
    cudnn.deterministic = True
    cudnn.benchmark = False
    model = CGTransformer(att_nums=3, use_RDA=True)
    # model = CrossTransformer(dim=722, depth=12, heads=16, mlp_dim=1444, dropout=0.1)
    # model = CrossTransformer(dim=722, depth=1, heads=1, mlp_dim=1024, dropout=0.1)
    summary(model.to('cuda:0'), (3, 299, 299))
    ### 使用torchvision.models._utils.IntermediateLayerGetter获取中间层的输出
    # model = Xception(num_classes=2)
    # model.last_linear = model.fc
    # del model.fc
    # new_m = IntermediateLayerGetter(model=model, return_layers={'block11': 'dsffwe'})
    # out = new_m(torch.rand(2, 3, 299, 299))
    # for k, v in out.items():
    #     print(k, v.shape)
