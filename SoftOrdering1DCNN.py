import torch
from torch import nn
from torch.nn.utils import weight_norm
from einops import rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.2):
        super().__init__()
        self.feedfoward = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.feedfoward(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 16, dropout = 0.2):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SoftOrdering1DCNN(nn.Module):

    def __init__(self, input_dim, output_dim, sign_size=16, cha_input=8, cha_hidden=32, dropout_rate = 0.3):
        super().__init__()

        hidden_size = sign_size*cha_input
        attn_dim = cha_hidden
        mlp_dim = attn_dim * 2
        output_size = (sign_size//4) * cha_hidden
        self.sign_size = sign_size
        self.cha_input = cha_input

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Input
        self.BnInput = nn.BatchNorm1d(input_dim)
        InputDense = nn.Linear(input_dim, hidden_size, bias=False)
        self.InputDense = weight_norm(InputDense)

        # Pooling layer
        self.AvgPooling = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
        self.MaxPooling = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        # 1st conv layer
        self.layer1 = self.make_layer(cha_input, cha_input*2, 5, 1, 2, groups=cha_input)

        # 2nd conv layer
        self.layer2 = self.make_layer(cha_input*2, cha_hidden, 3, 1, 1)

        # 3rd conv layer
        self.layer3 = self.make_layer(cha_hidden, cha_hidden, 5, 1, 2, groups=cha_hidden)

        #Attention
        self.attention = PreNorm(attn_dim, Attention(attn_dim))
        self.mlp = PreNorm(attn_dim, MLP(attn_dim, mlp_dim))

        # Output
        self.flatten = nn.Flatten()
        self.BnOutput = nn.BatchNorm1d(output_size)
        OutputDense = nn.Linear(output_size, output_dim, bias=False)
        self.OutputDense = weight_norm(OutputDense)

    def make_layer(self, cha_input, cha_output, KernelSize, stride, padding, groups=1, dropout_rate=0.3):
        conv = nn.Conv1d(
            cha_input,
            cha_output,
            kernel_size = KernelSize,
            stride = stride,
            padding = padding,
            groups = groups,
            bias = False)

        Layer = nn.Sequential(
            weight_norm(conv, dim=None),
            nn.BatchNorm1d(cha_output),
            nn.CELU(),
            nn.Dropout(dropout_rate),
            )

        return Layer

    def forward(self, x):
        x = self.BnInput(x)
        x = nn.functional.celu(self.InputDense(x))
        x = x.reshape(x.shape[0], self.cha_input, self.sign_size)

        x = self.layer1(x)
        x = self.MaxPooling(x)

        x = self.layer2(x)
        
        x = x.permute(0,2,1)
        x = self.attention(x) + x
        x = self.mlp(x) + x
        x = x.permute(0,2,1)

        x = self.layer3(x)
        x = self.AvgPooling(x)

        x = self.flatten(x)
        x = self.BnOutput(x)
        x = self.OutputDense(x)
        x = self.dropout(x)

        return nn.functional.softmax(x, dim=-1)