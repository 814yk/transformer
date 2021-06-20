from torch import nn
import torch
from mask import masking_pad, masking_subsequent


class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sources, targets):
        batch_size, sources_len = sources.size()
        batch_size, targets_len = targets.size()
        sources_mask = masking_pad(sources, sources_len)
        memory_mask = masking_pad(sources, targets_len)
        targets_mask = masking_subsequent(targets) | masking_pad(targets, targets_len)

        memory = self.encoder(sources, sources_mask)
        outputs, state = self.decoder(targets, memory, memory_mask, targets_mask)
        return outputs


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, dim_model, num_heads, dim_ff, dropout_prob, embedding):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(dim_model, num_heads, dim_ff, dropout_prob) for _ in range(num_layers)]
        )
        self.embedding = embedding

    def forward(self, sources, mask):

        sources = self.embedding(sources)

        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(sources, mask)

        return sources


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads, dim_ff, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(dim_model=dim_model, num_heads=num_heads, dropout_prob=dropout_prob)
        self.attention_norm = nn.LayerNorm(normalized_shape=dim_model)

        self.feed_forward = PositionWiseFeedForwardNetwork(
            dim_model=dim_model, dim_ff=dim_ff, dropout_prob=dropout_prob
        )
        self.feedforward_norm = nn.LayerNorm(normalized_shape=dim_model)

    def forward(self, x, mask):
        attn = self.self_attention(query=x, key=x, value=x, mask=mask)
        out = self.attention_norm(attn + x)
        ff = self.feedforward_layer(out)
        out = self.feedforward_norm(ff + out)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        # attn : (batch_size,num_heads,query_len,key_len)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(attn)
            attn = attn.masked_fill(mask=mask_expanded, value=float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        # out : (batch_size,num_heads,query_len,dim_head)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_heads, dropout_prob):
        super(MultiHeadAttention, self).__init__()
        assert dim_model % num_heads == 0
        self.dim_head = dim_model // num_heads
        self.num_heads = num_heads
        self.query_projection = nn.Linear(dim_model, num_heads * self.dim_head)
        self.key_projection = nn.Linear(dim_model, num_heads * self.dim_head)
        self.value_projection = nn.Linear(dim_model, num_heads * self.dim_head)
        self.final_projection = nn.Linear(dim_model, num_heads * self.dim_head)
        self.get_attention_score = ScaledDotProductAttention(temperature=dim_model ** 0.5, dropout=dropout_prob)
        self.dropout_prob = dropout_prob

    def forward(self, query, key, value, mask):
        batch_size, query_len, dim_model = query.size()
        batch_size, key_len, dim_model = key.size()
        batch_size, value_len, dim_model = value.size()

        query_projected = self.query_projection(query)
        key_projected = self.key_projection(key)
        value_projected = self.value_projection(value)

        query_heads = query_projected.view(batch_size, query_len, self.num_heads, self.dim_head).transpose(1, 2)
        # query_heads : (batch_size,num_heads,query_len,dim_head)
        key_heads = key_projected.view(batch_size, key_len, self.num_heads, self.dim_head).transpose(1, 2)
        # key_heads : (batch_size,num_heads,key_len,dim_head)
        value_heads = value_projected.view(batch_size, value_len, self.num_heads, self.dim_head).transpose(1, 2)
        # value_heads : (batch_size,num_heads,value_len,dim_head)

        attn_score = self.get_attention_score(query_heads, key_heads, value_heads, mask)
        # context_score : (batch_size,num_heads,query_len,dim_head)
        attn_score = attn_score.transpose(1, 2).contiguous()
        # attn_score : (batch_size,query_len,num_heads,dim_head)
        score = attn_score.view(batch_size, query_len, self.num_heads * self.dim_head)
        # score : (batch_size,query_len,dim_model)
        out = self.final_projection(score)
        return out


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, dim_model, dim_ff, dropout_prob):
        super(PositionWiseFeedForwardNetwork, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_ff, True),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(dim_ff, dim_model, True),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        return self.feed_forward(x)
