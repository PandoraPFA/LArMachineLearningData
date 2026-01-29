import math
import os
import numpy.random as random
import torch
import torch.nn as nn

class TrackPIDAttention(nn.Module):
    def __init__(self, model_depth, feed_forward_depth, n_heads, dropout):
        super().__init__()
        """
            Custom multi-head attention layer for cross-attention

            Args: 
                model_depth: the depth of the encoding (often called d_model)
                feed_forward_depth: the depth of the feed-forward layers
                n_heads: the number of attention heads
                dropout: the dropout fraction
        """
        self.attention = nn.MultiheadAttention(model_depth, n_heads, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(model_depth)
        self.layer_norm_2 = nn.LayerNorm(model_depth)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_depth, feed_forward_depth),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_depth, model_depth)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, kv_mask):
        """
            Forward pass

            Args:
                q: input query torch tensor
                kv: input key and values torch tensor
                kv_mask: padding mask for kv

            Returns:
                layer output torch tensor
        """
        # 1. Cross-attention (mask K/V only)
        kv_mask = kv_mask.to(dtype=torch.bool)
        kv_mask = kv_mask.contiguous()
        out, _ = self.attention(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=kv_mask,
            need_weights=False
        )

        # 2. Residual + norm
        out = self.dropout(out)
        out = self.layer_norm_1(out + q)

        # 3. Feed-forward
        out2 = self.feed_forward(out)
        out = out + self.dropout(out2)
        out = self.layer_norm_2(out)

        return out

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class TrackPIDCrossAttentionEncoder(nn.Module):
    def __init__(self, model_depth, feed_forward_depth, n_heads, n_layers, dropout):
        super().__init__()
        """
            Cross-attention encoder

            Args: 
                model_depth: the depth of the encoding (often called d_model)
                feed_forward_depth: the depth of the feed-forward layers
                n_heads: the number of attention heads
                n_layers: the number of encoder layers
                drouput: the dropout fraction
        """
        self.encoder_layers = nn.ModuleList()
        for l in range(n_layers):
            self.encoder_layers.append(TrackPIDAttention(model_depth, feed_forward_depth, n_heads, dropout))

    def forward(self, q, kv, kv_mask): 
        """
            Forward pass

            Args:
                q: input query torch tensor
                kv: input key and values torch tensor
                kv_mask: padding mask for kv

            Returns:
                output torch tensor from the encoder
        """
        out = q
        for encoder_layer in self.encoder_layers:
            out = encoder_layer(out, kv, kv_mask)
        return out

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class TrackPIDNetwork2d(nn.Module):
    def __init__(self, n_features, n_classes, sequence_length, model_depth, n_heads, feed_forward_depth, n_encoder_layers, n_cross_attention_layers, dropout, n_auxillary):
        super(TrackPIDNetwork2d, self).__init__()
        """
            Track PID model

            Args:
                n_features: number of features in the input hit sequences
                n_classes: number of classes to classify the tracks into
                sequence_length: length of the input hit sequences
                model_depth: depth of the encoded inputs (also called d_model)
                n_heads: number of attention heads to use
                feed_forward_depth: depth of the feed forward layers
                n_encoder_layers: number of layers used in the self-attention encoders
                n_cross_attention_layers: number of layers in the cross-attention encoders
                dropout: the dropout fraction
                n_auxillary: the number of auxillary variables
        """
        self.n_features = n_features
        self.sequence_length = sequence_length

        # Shared input mapping for each view
        self.input_mapping = nn.Linear(n_features, model_depth)
        self.position_encoding = nn.Embedding(sequence_length, model_depth)
        self.stream_embedding = nn.Embedding(3, model_depth)
        self.layer_norm_u = nn.LayerNorm(model_depth)
        self.layer_norm_v = nn.LayerNorm(model_depth)
        self.layer_norm_w = nn.LayerNorm(model_depth)
        self.layer_norm_aux = nn.LayerNorm(model_depth)

        self.skip_proj = nn.Sequential(nn.LayerNorm(2 * model_depth), nn.Linear(2 * model_depth, model_depth))

        # Shared encoder for the first processing of u, v and w sequences
        self.encoder_layer = nn.TransformerEncoderLayer(model_depth, n_heads, feed_forward_depth, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_encoder_layers)

        # For cross attention, we need to do things ourselves.
        self.cross_attention = TrackPIDCrossAttentionEncoder(model_depth, feed_forward_depth, n_heads, n_cross_attention_layers, dropout)

        # Use the CLS token approach for the cross-attention encoders
        self.cls_tokens = nn.Parameter(torch.randn(6, model_depth))
        self.cls_layer_norm = nn.LayerNorm(6 * model_depth)

        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        self.classifier_dropout = nn.Dropout(2.0 * dropout)
        # Classifier layer
        self.aux_expand = nn.Linear(n_auxillary, model_depth)
        self.classifier1 = nn.Linear(6 * model_depth, model_depth)
        self.classifier2 = nn.Linear(2 * model_depth, n_classes)

    def forward(self, u, v, w, auxillary):
        """
            Forward pass

            Args:
                u: input u view hit sequence
                v: input v view hit sequence
                w: input w view hit sequence
                auxillary: input auxillary variables

            Returns:
                track PID predictions (no softmax applied)
        """
        # Create input masks
        u_mask = (u[:, :, 2].abs() < 1e-8).to(dtype=torch.bool, device=u.device)
        v_mask = (v[:, :, 2].abs() < 1e-8).to(dtype=torch.bool, device=v.device)
        w_mask = (w[:, :, 2].abs() < 1e-8).to(dtype=torch.bool, device=w.device)

        # Expand the inputs to model_depth
        u = self.input_mapping(u)
        v = self.input_mapping(v)
        w = self.input_mapping(w)

        # Add simple position encoding
        positions = torch.arange(u.size(1), device=u.device).unsqueeze(0).repeat(u.size(0), 1)
        u = u + self.position_encoding(positions) + self.stream_embedding(torch.tensor(0, device=u.device))
        v = v + self.position_encoding(positions) + self.stream_embedding(torch.tensor(1, device=v.device))
        w = w + self.position_encoding(positions) + self.stream_embedding(torch.tensor(2, device=w.device))

        u = self.layer_norm_u(u)
        v = self.layer_norm_v(v)
        w = self.layer_norm_w(w)

        u = self.dropout(u)
        v = self.dropout(v)
        w = self.dropout(w)

        # Run the self attention encoders with masking
        u = self.encoder(u, src_key_padding_mask=u_mask)
        v = self.encoder(v, src_key_padding_mask=v_mask)
        w = self.encoder(w, src_key_padding_mask=w_mask)

        # Pooled versions for skip connections
        u_pool = u.mean(dim=1)   # (B, D)
        v_pool = v.mean(dim=1)   # (B, D)
        w_pool = w.mean(dim=1)   # (B, D)
        skip_uv = self.skip_proj(torch.cat([u_pool, v_pool], dim=-1))
        skip_uw = self.skip_proj(torch.cat([u_pool, w_pool], dim=-1))
        skip_vw = self.skip_proj(torch.cat([v_pool, w_pool], dim=-1))

        # Add the required cls tokens
        cls_tokens = self.cls_tokens.unsqueeze(0).expand(u.size(0), 6, -1)
        u_for_uv = torch.cat([cls_tokens[:,0,:].unsqueeze(1), u], dim=1)
        u_for_uw = torch.cat([cls_tokens[:,1,:].unsqueeze(1), u], dim=1)
        v_for_vu = torch.cat([cls_tokens[:,2,:].unsqueeze(1), v], dim=1)
        v_for_vw = torch.cat([cls_tokens[:,3,:].unsqueeze(1), v], dim=1)
        w_for_wu = torch.cat([cls_tokens[:,4,:].unsqueeze(1), w], dim=1)
        w_for_wv = torch.cat([cls_tokens[:,5,:].unsqueeze(1), w], dim=1)

        # Run the cross attention encoders with masking in a bidirectional fashion
        cls_outputs = []
        cls_outputs.append(skip_uv + self.cross_attention(u_for_uv, v, v_mask)[:,0,:])
        cls_outputs.append(skip_uv + self.cross_attention(v_for_vu, u, u_mask)[:,0,:])
        cls_outputs.append(skip_uw + self.cross_attention(u_for_uw, w, w_mask)[:,0,:])
        cls_outputs.append(skip_uw + self.cross_attention(w_for_wu, u, u_mask)[:,0,:])
        cls_outputs.append(skip_vw + self.cross_attention(v_for_vw, w, w_mask)[:,0,:])
        cls_outputs.append(skip_vw + self.cross_attention(w_for_wv, v, v_mask)[:,0,:])

        # Now we need to combine out six outputs and perform the classification
        all_cls_outputs = torch.cat(cls_outputs, dim=1)
        all_cls_outputs = self.classifier_dropout(all_cls_outputs)
        all_cls_outputs = self.cls_layer_norm(all_cls_outputs)

        # Finally, run the classifier
        output = self.classifier1(all_cls_outputs)
        output = self.relu(output)
        output = self.classifier_dropout(output)
        auxillary = self.aux_expand(auxillary)
        auxillary = self.layer_norm_aux(auxillary)
        output = self.classifier2(torch.cat([output, auxillary], dim=1))
        return output
