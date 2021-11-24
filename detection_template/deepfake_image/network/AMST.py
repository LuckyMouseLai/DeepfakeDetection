import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torchsummary import summary
from sklearn.metrics.pairwise import pairwise_distances, paired_distances
from torch.nn.functional import pairwise_distance

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        # x.shape [?, 65, 1024]
        for attn, ff in self.layers:
            x = attn(x, mask = mask)  # attention
            x = ff(x)  # feed forward
        return x

class AdaptiveSelectModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=1)

    def forward(self, x):
        # x [batch_size, 64 + 1, dim]
        pass




class AMST(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim=[1024, 256, 64], depth, heads, mlp_dim=[2048, 512, 128], channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2  # 64
        
        patch_dim_l = channels * patch_size ** 2  # 3 × 32**2 = 3072
        patch_dim_m = channels * 16 ** 2  # 3 × 16**2 = 768
        patch_dim_s = channels * 8 ** 2  # 3 × 8**2 = 192
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_size_l = patch_size  # 32
        self.patch_size_m = 16
        self.patch_size_s = 8
        # print('image_size:{} patch_size:{} num_patches:{} patch_dim:{}'.format(image_size, patch_size, num_patches, patch_dim))
        self.pos_embedding_l = nn.Parameter(torch.randn(1, num_patches + 1, dim[0]))  # (1, 65, 1024)
        self.pos_embedding_m = nn.Parameter(torch.randn(1, num_patches + 1, dim[1]))  # (1, 65, 512)
        self.pos_embedding_s = nn.Parameter(torch.randn(1, num_patches + 1, dim[2]))  # (1, 65, 128)
        # print('position embedding', self.pos_embedding.shape)
        self.patch_to_embedding_l = nn.Linear(patch_dim_l, dim[0])
        self.patch_to_embedding_m = nn.Linear(patch_dim_m, dim[1])
        self.patch_to_embedding_s = nn.Linear(patch_dim_s, dim[2])

        self.cls_token_l = nn.Parameter(torch.randn(1, 1, dim[0]))  # (1, 1, 1024)
        self.cls_token_m = nn.Parameter(torch.randn(1, 1, dim[1]))  # (1, 1, 256)
        self.cls_token_s = nn.Parameter(torch.randn(1, 1, dim[2]))  # (1, 1, 64)
        # print('cls_token', self.cls_token.shape)
        self.dropout = nn.Dropout(emb_dropout)


        # large scale transformer -- patch_size = 32 

        self.transformer_l = Transformer(dim[0], depth, heads, mlp_dim[0], dropout)

        # medium scale transformer -- patch_size = 16
        self.transformer_m = Transformer(dim[1], depth, heads, mlp_dim[1], dropout)
        # small scale transformer -- patch_size = 8
        self.transformer_s = Transformer(dim[2], depth, heads, mlp_dim[2], dropout)
        # print('transformer dim:{} depth:{} heads:{} mlp_dim:{} dropout:{}'.format(dim, depth, heads, mlp_dim, dropout))

        self.to_cls_token = nn.Identity()

        self.mlp_head_l = nn.Sequential(
            nn.LayerNorm(dim[0]),
            nn.Linear(dim[0], num_classes)
        )
        self.mlp_head_m = nn.Sequential(
            nn.LayerNorm(dim[1]),
            nn.Linear(dim[1], num_classes)
        )
        self.mlp_head_s = nn.Sequential(
            nn.LayerNorm(dim[2]),
            nn.Linear(dim[2], num_classes)
        )    
    def compute_distance(self, x, y):
        # x:[m, k]  y:[n, k]
        # 1. 计算成对的距离x[0]y[0], x[1]y[1], m = n, 返回m个标量
        # dis = paired_distances(x, y, metric='euclidean')
        # 2. 计算所有距离x[0]y[:], x[1]y[:], 返回 [m, n]的矩阵
        """  
            计算cls_token与其他token的距离
                即：D(cls_token, patch_tokens)
            input: x:[batch_size, dim]  y:[batch_size, num_patchs, dim]
            return: distance [batch_size, num_patchs]

            distance[i, j]表示batch中第i张img的cls_token和第j个patch_token的距离(j从0开始编号)
            
        """
        x = torch.unsqueeze(x, 1)
        # print(x.shape, y.shape)
        distance = torch.cdist(x.float(), y.float(), p=2)  # euclidean distance
        # print(distance[0])  # 验证distance的归一化
        distance = torch.squeeze(distance, 1)
        return distance

    def choose_blocks(self, patch_size, distance, img):
        """  
            input: 
                img[batch_size, H, W] 
                patch_size: the size of patch, a scalar  
                distance: [batch_size, num_patchs]
                feature_map: [batch_size, num_patch, patch_size, patch_size]

            return : croped_img[batch_size, 4*p_size, 4*p_size]
        """
        b, n = distance.shape
        croped_img = torch.zeros((b, 3, 4*patch_size, 4*patch_size), dtype=torch.float, device=img.device)
        
        distance = distance.reshape((b, 8, 8))  # [?, 8, 8]
        avg_pool = nn.AvgPool2d(kernel_size=4, stride=1)  
        distance = avg_pool(distance)  # [?, 5, 5]
        _, h , w = distance.shape
        distance = distance.reshape(b, -1)

        min = torch.argmin(distance, dim=1)  # [Batch_size]

        i = torch.floor_divide(min, w)  # index i: [batch_size]
        j = torch.fmod(min, w)  # index j: [batch_size]
        ### choose from feature maps

        ### choose from img
        for k in range(b):
            croped_one_dim = torch.narrow(img[k], 1, i[k]*patch_size, 4*patch_size)
            croped_img[k] = torch.narrow(croped_one_dim, 2, j[k]*patch_size, 4*patch_size)
        
        return croped_img

    def forward(self, img, texture, mask = None):
        
        ### large scale transformer
        p_l = self.patch_size_l  # 32
        x_l = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p_l, p2 = p_l)
        x_l = self.patch_to_embedding_l(x_l)
        b_l, n_l, _ = x_l.shape  # [?, 64, 1024]
        cls_tokens_l = repeat(self.cls_token_l, '() n d -> b n d', b = b_l)  # [?, 1, 1024] cls tokens for every item in a batch
        x_l = torch.cat((cls_tokens_l, x_l), dim=1)  # [?, 65, 1024]  add cls_tokens
        x_l += self.pos_embedding_l[:, :(n_l + 1)]  # x: [?, 65, 1024] pos_embedding: [1, 65, 1024]
        x_l = self.dropout(x_l)
        x_l = self.transformer_l(x_l, mask)  # [?, 65, 1024]
        feature_map_l = x_l[:, 1:].reshape(b_l, n_l, p_l, p_l)  # [?, 64, 32, 32]
        distance_l = self.compute_distance(x_l[:, 0], x_l[:, 1: ])  # [batch_size, num_patchs]
        img = self.choose_blocks(p_l, distance_l, img)  # [?, 3, 128, 128]
        # print('large transformer out img', img.shape)

        ### medium scale transformer
        p_m = self.patch_size_m
        x_m = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p_m, p2 = p_m)
        x_m = self.patch_to_embedding_m(x_m)
        b_m, n_m, _ = x_m.shape
        cls_tokens_m = repeat(self.cls_token_m, '() n d -> b n d', b=b_m)
        x_m = torch.cat((cls_tokens_m, x_m), dim=1)
        x_m += self.pos_embedding_m[:, :(n_m + 1)]
        x_m = self.dropout(x_m)
        # print(x_m.shape)
        x_m = self.transformer_m(x_m, mask)
        # print(x_m.shape)
        feature_map_m = x_m[:, 1:].reshape(b_m, n_m, p_m, p_m)  
        distance_m = self.compute_distance(x_m[:, 0], x_m[:, 1: ])  # [batch_size, num_patchs]
        img = self.choose_blocks(p_m, distance_m, img)
        ### small scale transformer
        p_s = self.patch_size_s
        x_s = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p_s, p2 = p_s)
        x_s = self.patch_to_embedding_s(x_s)
        b_s, n_s, _ = x_s.shape
        cls_tokens_s = repeat(self.cls_token_s, '() n d -> b n d', b=b_s)
        x_s = torch.cat((cls_tokens_s, x_s), dim=1)
        x_s += self.pos_embedding_s[:, :(n_s + 1)]
        x_s = self.dropout(x_s)
        x_s = self.transformer_s(x_s, mask)
        feature_map_s = x_s[:, 1:].reshape(b_s, n_s, p_s, p_s)
        # distance_s = self.compute_distance(x_s[:, 0], x_s[:, 1: ])  # [batch_size, num_patchs]
        # img = self.choose_blocks(p_s, distance_s, img)

        # choose cls_token to classification
        x = self.to_cls_token(x_s[:, 0])
        return self.mlp_head_s(x)

if __name__ == '__main__':
    t = AMST(image_size=256, patch_size=32, num_classes=2, dim=[1024, 256, 64], depth=1, heads=16, mlp_dim=[2048, 512, 128], dropout=0.1, emb_dropout=0.1)
    print(t)
    # summary(t.to('cuda:3'), (3, 256, 256))