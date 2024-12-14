import torch.nn as nn

class AttentionLayer(nn.Module):
    """
    Multi-head attention layer to fuse visual and textual features.
    """
    def __init__(self, embed_dim, num_heads):
        super(AttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, visual_feats, textual_feats):
        # Self-attention between visual and textual features
        attn_output, _ = self.multihead_attn(visual_feats, textual_feats, textual_feats)
        return self.norm(attn_output + visual_feats)

class ExtendedPhi(nn.Module):
    """
    Enhanced Textual Inversion Phi network with attention mechanism.
    Takes as input the visual features of an image and optional textual features,
    outputs the pseudo-word embedding.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_heads: int):
        super().__init__()
        self.visual_proj = nn.Linear(input_dim, hidden_dim)
        self.textual_proj = nn.Linear(input_dim, hidden_dim)

        self.attention = AttentionLayer(hidden_dim, num_heads)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, visual_feats, textual_feats):
        # Project visual and textual features to hidden space
        visual_feats = self.visual_proj(visual_feats)
        textual_feats = self.textual_proj(textual_feats)

        # Fuse using attention
        fused_feats = self.attention(visual_feats, textual_feats)

        # Generate the pseudo-word embedding
        return self.mlp(fused_feats)
