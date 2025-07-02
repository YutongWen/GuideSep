import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from .operator_utils import prob_mask_like
from .utils import exists
import torch.nn.functional as F


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations. Same as timestep_embedding in unet2d_oai
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    in DiT: class_channels== time_embed_dim
    """
    def __init__(self, 
                 num_classes, 
                 class_embed_dim,
                 model_channels, 
                 class_channels,
                 ):
        super().__init__()

        assert num_classes is None or class_embed_dim is None, "Provide either num_classes or class_embed_dim, not both."
        self.num_classes = num_classes
        self.null_classes_emb = nn.Parameter(torch.randn(1, model_channels))

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, model_channels)

        elif class_embed_dim is not None:
            # only embedding is provided
            self.class_embed_norm = L2NormalizationLayer()
            self.label_emb = nn.Linear(class_embed_dim, model_channels)
            nn.init.normal_(self.null_classes_emb, 0, 1 / model_channels ** 0.5) # TODO: verify
        
        self.class_to_cond = nn.Sequential(
                nn.LayerNorm(model_channels),
                nn.Linear(model_channels, class_channels),
                nn.SiLU(),
                nn.Linear(class_channels, class_channels)
            )

    def forward(self, classes, cond_drop_prob):

        if self.num_classes is None:
            classes = self.class_embed_norm(classes)

        classes_emb = self.label_emb(classes)
        
        if cond_drop_prob > 0:

            label_keep_mask = prob_mask_like((classes.shape[0],), 1 - cond_drop_prob, device=classes.device)

            classes_emb = torch.where(
                rearrange(label_keep_mask, 'b -> b 1'),
                classes_emb,
                self.null_classes_emb
            )

        classes_emb = self.class_to_cond(classes_emb)

        return classes_emb

class TextEmbedder(nn.Module):
    """
    Embeds text into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, 
                 text_cond_dim, 
                 text_embed_dim, 
                 max_text_len):
        super().__init__()

        self.text_to_cond = nn.Linear(text_embed_dim, text_cond_dim)

        # for classifier free guidance
        self.max_text_len = max_text_len
        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, text_cond_dim))

        # normalizations
        self.norm_cond = nn.LayerNorm(text_cond_dim)

    def forward(self, text_embeds, text_mask, cond_drop_prob):

        batch_size, device = text_embeds.shape[0], text_embeds.device

        text_vectors = self.text_to_cond(text_embeds)
            
        text_vectors = text_vectors[:, :self.max_text_len]
        text_vectors_len = text_vectors.shape[1]

        remainder = self.max_text_len - text_vectors_len

        if remainder > 0:
            text_vectors = F.pad(text_vectors, (0, 0, 0, remainder))

        if exists(text_mask):
            text_mask = text_mask[:, :self.max_text_len]

            if remainder > 0:
                text_mask = F.pad(text_mask, (0, remainder), value = False)

            text_mask = rearrange(text_mask, 'b n -> b n 1')
        
        if cond_drop_prob > 0:
            text_keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
            text_keep_mask_embed = rearrange(text_keep_mask, 'b -> b 1 1')

            null_text_mask = torch.ones(batch_size, self.max_text_len).to(text_vectors.dtype).to(device)

            if exists(text_mask):
                text_keep_mask_embed = text_mask & text_keep_mask_embed
                text_mask = text_mask.squeeze(-1)
                text_mask = torch.where(
                    rearrange(text_keep_mask, 'b -> b 1'),
                    text_mask,
                    null_text_mask)

            null_text_embed = self.null_text_embed.to(text_vectors.dtype)
            text_vectors = torch.where(
                text_keep_mask_embed,
                text_vectors,
                null_text_embed)
        else:
            text_mask = text_mask.squeeze(-1) if exists(text_mask) else None

        context = self.norm_cond(text_vectors)

        return context, text_mask