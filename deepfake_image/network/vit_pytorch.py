import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torchsummary import summary

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
            # nn.GLU,
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
        self.U = torch.tensor([1.0], dtype=torch.float)
        self.W = torch.tensor([1.0], dtype=torch.float)
        self.b = torch.tensor([0.0], dtype=torch.float)
        self.Mem = torch.zeros_like(self.W, dtype=torch.float)
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

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2  # 64
        
        patch_dim = channels * patch_size ** 2  # 3 × 32**2 = 3072
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_size = patch_size  # 32
        print('image_size:{} patch_size:{} num_patches:{} patch_dim:{}'.format(image_size, patch_size, num_patches, patch_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # (1, 65, 1024)
        print('position embedding', self.pos_embedding.shape)
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # (1, 1, 1024)
        print('cls_token', self.cls_token.shape)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        print('transformer dim:{} depth:{} heads:{} mlp_dim:{} dropout:{}'.format(dim, depth, heads, mlp_dim, dropout))

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape  # [?, 64, 1024]
        print('x.shape', x.shape)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # [?, 1, 1024] cls tokens for every item in a batch
        print('cls tokens', cls_tokens.shape, x.shape)
        x = torch.cat((cls_tokens, x), dim=1)  # [?, 65, 1024]  add cls_tokens
        print('torch cat', x.shape)
        x += self.pos_embedding[:, :(n + 1)]  # x: [?, 65, 1024] pos_embedding: [1, 65, 1024]
        x = self.dropout(x)
        print('ViT forward:', self.pos_embedding[:, :(n+1)].shape)
        x = self.transformer(x, mask)  # [?, 65, 1024]
        print('ViT forward:', x.shape)  
        print('ViT forward:', x[:, 0].shape) 
        x = self.to_cls_token(x[:, 0])  # x:[?, 1024] x[:, 0]: [?, 1024]
        return self.mlp_head(x)

if __name__ == '__main__':
    t = ViT(image_size=256, patch_size=32, num_classes=2, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)
    # print(t)
    summary(t.cuda(), (3, 256, 256))