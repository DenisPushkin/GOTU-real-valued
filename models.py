
import torch
from torch import nn
import torch.nn.functional as F
import math

from einops import rearrange, repeat

"""
    Transformer model

    This code was adapted from https://github.com/aryol/GOTU/blob/main/token_transformer.py
"""

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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
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

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    


class TokenTransformer(nn.Module):
    def __init__(self, *, seq_len, output_dim, dim, depth, heads, mlp_dim, input_domain, num_embeddings=0,
                 pool='cls', dim_head=64, dropout=0., emb_dropout=0., albert=False):
        super().__init__()
        
        if input_domain not in ['pos_neg_ones', 'bounded_integers']:
            raise ValueError("Illegal value of input_domain")
        
        if (input_domain == 'bounded_integers') and (num_embeddings % 2 != 1):
            raise ValueError('When input_domain is bounded_integers, num_embeddings must be odd')
        
        self.token_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.albert = albert
        self.window_size = round((num_embeddings - 1) / 2) if input_domain == 'bounded_integers' else None
        self.input_domain = input_domain
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_dim)
        )

    def forward(self, inputs):
        if self.input_domain == 'pos_neg_ones':
            inputs = torch.round((inputs + 1) / 2).int()
        else:
            # self.input_domain == 'bounded_integers'
            inputs = torch.round(inputs + self.window_size).int()
        x = self.token_embedding(inputs)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

"""
    Random Features model
"""

class RandomFeatures(nn.Module):

    def __init__(self, input_dimension, num_features, activation, small_features=False, epsilon=None):
        super(RandomFeatures, self).__init__()
        self.activation = activation
        self.features = nn.Linear(input_dimension, num_features)
        std = epsilon if small_features else 1/math.sqrt(input_dimension)
        print(f'small_features = {small_features}, std = {std}')
        torch.nn.init.normal_(self.features.bias, mean=0, std=std)
        torch.nn.init.normal_(self.features.weight, mean=0, std=std)
        for param in self.features.parameters():
            param.requires_grad = False

        self.weights = nn.Linear(num_features, 1, bias=False)
        self.weights.weight.data.fill_(0.0)

    def forward(self, x):
        return self.weights(self.activation(self.features(x)))

"""
    MLP
"""

class MLP(nn.Module):

    def __init__(self, input_dimension, n_layers, layer_width, activation, small_features=False, epsilon=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([])
        self.activation = activation
        self.layers.append(nn.Linear(input_dimension, layer_width))
        for _ in range(n_layers-2):
            self.layers.append(nn.Linear(layer_width, layer_width))
        self.layers.append(nn.Linear(layer_width, 1))

        if small_features:
            print('In small features regime')
            for i in range(len(self.layers) - 1):
                torch.nn.init.normal_(self.layers[i].weight, mean=0, std=epsilon)
                torch.nn.init.normal_(self.layers[i].bias, mean=0, std=epsilon)
            
            i = len(self.layers) - 1 # id of last layer
            self.layers[i].weight.data.fill_(0.0)
            self.layers[i].bias.data.fill_(0.0)
        else:
            print('Default regime')

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        
        return x

"""
    Helpers
"""

def get_activation_fn(activation_name, deg):
    if activation_name == 'relu': activation = F.relu
    elif activation_name == 'shifted_relu': activation = lambda x: F.relu(x) - 1
    elif activation_name == 'gelu': activation = F.gelu
    elif activation_name == 'tanh': activation = F.tanh
    elif activation_name == 'sigmoid': activation = F.sigmoid
    elif activation_name == 'elu': activation = F.elu
    elif activation_name == 'softplus': activation = F.softplus
    elif activation_name == 'poly':
       # non_central_moments[i] = E[x^i], X~N(1,1), 0 <= n <= 8
       non_central_moments = [1,1,2,4,10,26,76,232,764]
       activation = lambda x: torch.pow(1+x, deg) / (2*non_central_moments[deg])
    else: raise ValueError(f"Unexpected value of 'activation_name' parameter: received {activation_name} of type {type(activation_name)}")

    return activation

def get_transformer_input_domain(distribution):
    if distribution == 'boolean':
       input_domain = 'pos_neg_ones'
    elif distribution == 'unif_discrete':
       input_domain = 'bounded_integers'
    else:
       raise ValueError(f'Transformer can only be used with discrete distributions, but received {distribution}')
    
    return input_domain

def build_model(architecture, input_dimension, **params):
    if architecture == 'rf':
        activation_fn = get_activation_fn(params['activation'], params['deg'])
        model = RandomFeatures(input_dimension, params['num_features'], activation_fn, params['small_features'], params['epsilon'])
    elif architecture == 'mlp':
        activation_fn = get_activation_fn(params['activation'], params['deg'])
        model = MLP(input_dimension, params['n_layers'], params['layer_width'], activation_fn, params['small_features'], params['epsilon'])
    elif architecture == 'transformer':
        input_domain = get_transformer_input_domain(params['distr'])
        num_embeddings = 2 if params['distr'] == 'boolean' else params['support_size']
        model = TokenTransformer(
            seq_len=input_dimension, output_dim=1, dim=64, depth=12, heads=6, mlp_dim=64, 
            input_domain=input_domain, num_embeddings=num_embeddings)
    else:
        raise ValueError("illegal value of 'arch' parameter")
    
    return model



