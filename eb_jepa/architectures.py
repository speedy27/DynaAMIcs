from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from eb_jepa.nn_utils import TemporalBatchMixin, init_module_weights


class conv3d2(nn.Sequential):
    """Simple 3D convnet with 2 layers."""

    def __init__(self, in_d, h_d, out_d, tk, ts, sk, ss, pad):
        super(conv3d2, self).__init__(
            nn.Conv3d(
                in_d, h_d, kernel_size=(tk, sk, sk), stride=(1, 1, 1), padding=pad
            ),
            nn.ReLU(),
            nn.Conv3d(
                h_d, out_d, kernel_size=(tk, sk, sk), stride=(ts, ss, ss), padding=pad
            ),
        )
        self.apply(init_module_weights)
        self.input_dim = in_d
        self.hidden_dim = h_d
        self.output_dim = out_d
        # t_shift is the index (in the time dimension) of the first output
        # cannot see its coresponding input
        if pad == "valid":
            self.t_shift = 2 * tk - 1
        elif pad == "same":
            self.t_shift = 2 * (tk - 1)
        else:
            raise NameError("invalid padding for con3d2. Must be 'valid' or 'same'")


class ResidualBlock(nn.Module):
    """Standard residual block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet5(TemporalBatchMixin, nn.Module):
    """
    A lightweight ResNet with 5 layers (2 blocks).
    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """

    def __init__(self, in_d, h_d, out_d, s1=1, s2=1, s3=1, avg_pool=False):
        super().__init__()
        self.avg_pool = avg_pool
        self.conv1 = nn.Conv2d(
            in_d, h_d, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(h_d, h_d, stride=s1)
        self.layer2 = ResidualBlock(h_d, h_d * 2, stride=s2)
        self.layer3 = ResidualBlock(h_d * 2, out_d, stride=s3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if avg_pool else torch.nn.Identity()

    def _forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        if self.avg_pool:
            out = out.flatten(1)
        return out


class SimplePredictor(nn.Module):
    """Wrapper that concatenates states and actions channel-wise before prediction."""

    def __init__(self, predictor, context_length):
        super().__init__()
        self.predictor = predictor
        self.is_rnn = predictor.is_rnn
        self.context_length = context_length

    def forward(self, x, a):
        return self.predictor(torch.cat([x, a], dim=1))


class StateOnlyPredictor(SimplePredictor):
    """Wrapper for a simple predictor which concatenates states and actions channel wise."""

    def forward(self, x, a):
        # action not used on purpose
        prev_state = x[:, :, :-1]  # [B, C, T-1, H, W]
        next_state = x[:, :, 1:]  # [B, C, T-1, H, W]
        combined_xa = torch.cat((prev_state, next_state), dim=1)
        return self.predictor(combined_xa)


class ResUNet(TemporalBatchMixin, nn.Module):
    """
    A small UNet with residual encoder blocks and transposed-conv upsampling.
    Channels scale like h, 2h, 4h, 8h. Output keeps the input HxW.
    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """

    def __init__(self, in_d, h_d, out_d, is_rnn=False):
        super().__init__()
        self.is_rnn = is_rnn
        # Stem
        self.conv1 = nn.Conv2d(
            in_d, h_d, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)

        # Encoder
        self.enc1 = ResidualBlock(h_d, h_d, stride=1)  # H, W
        self.enc2 = ResidualBlock(h_d, 2 * h_d, stride=2)  # H/2, W/2
        self.enc3 = ResidualBlock(2 * h_d, 4 * h_d, stride=2)  # H/4, W/4
        self.bott = ResidualBlock(4 * h_d, 8 * h_d, stride=2)  # H/8, W/8

        # Decoder upsamples, then fuses skip with a residual block that reduces channels
        self.up3 = nn.ConvTranspose2d(8 * h_d, 4 * h_d, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(8 * h_d, 4 * h_d, stride=1)

        self.up2 = nn.ConvTranspose2d(4 * h_d, 2 * h_d, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(4 * h_d, 2 * h_d, stride=1)

        self.up1 = nn.ConvTranspose2d(2 * h_d, 1 * h_d, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(2 * h_d, 1 * h_d, stride=1)

        # Head
        self.head = nn.Conv2d(h_d, out_d, kernel_size=1)

    @staticmethod
    def _match_size(x, ref):
        # Guards against odd input sizes by resizing the upsample to the skip spatial dims
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(
                x, size=ref.shape[-2:], mode="bilinear", align_corners=False
            )
        return x

    def _forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))

        # Encoder with skips
        s1 = self.enc1(x0)  # h
        s2 = self.enc2(s1)  # 2h
        s3 = self.enc3(s2)  # 4h
        b = self.bott(s3)  # 8h

        # Decoder stage 3
        d3 = self.up3(b)
        d3 = self._match_size(d3, s3)
        d3 = torch.cat([d3, s3], dim=1)  # 4h + 4h = 8h
        d3 = self.dec3(d3)  # → 4h

        # Decoder stage 2
        d2 = self.up2(d3)
        d2 = self._match_size(d2, s2)
        d2 = torch.cat([d2, s2], dim=1)  # 2h + 2h = 4h
        d2 = self.dec2(d2)  # → 2h

        # Decoder stage 1
        d1 = self.up1(d2)
        d1 = self._match_size(d1, s1)
        d1 = torch.cat([d1, s1], dim=1)  # h + h = 2h
        d1 = self.dec1(d1)  # → h

        out = self.head(d1)  # → out_d channels
        return out


class Projector(nn.Module):
    """MLP projector built from a spec string like '256-512-128'."""

    def __init__(self, mlp_spec):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]  # Store output dimension as attribute

    def forward(self, x):
        return self.net(x)


class DetHead(nn.Module):
    """Detection head that pools features and predicts binary maps."""

    def __init__(self, in_d, h_d, out_d):
        super().__init__()
        self.head = nn.Sequential(conv3d2(in_d, h_d, out_d, 1, 1, 3, 1, "same"))
        self.apply(init_module_weights)

    def forward(self, x):
        """Forward pass on predictor output of shape (B, C, T, H, W)."""
        # (Batch, Feature, Time, Height, Width)
        # [8, 8, T, 8, 8]
        x = [F.adaptive_avg_pool2d(x[:, :, t], (8, 8)) for t in range(x.shape[2])]
        x = torch.stack(x, 2)
        # [8, T, 8, 8]
        x = self.head(x).squeeze(1)

        return torch.sigmoid(x)

    @torch.no_grad()
    def score(self, preds, targets):

        scores = []
        for T in range(len(preds) - 1):
            x = preds[T]
            x = [F.adaptive_avg_pool2d(x[:, :, t], (8, 8)) for t in range(x.shape[2])]
            x = torch.stack(x, 2)
            x = self.head(x).squeeze(1)

            y = targets[:, T:]
            x = x[:, T:]

            ap = average_precision_score(
                y.flatten().detach().long().cpu().numpy(),
                x.flatten().detach().cpu().numpy(),
                average="weighted",
            )
            scores.append(ap)

        return scores


class ResnetBlock(nn.Module):
    """ResNet Block."""

    def __init__(self, num_features):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)


class ResnetStack(nn.Module):
    """ResNet stack module."""

    def __init__(self, input_channels, num_features, num_blocks, max_pooling=True):
        super(ResnetStack, self).__init__()
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling
        self.initial_conv = nn.Conv2d(
            input_channels, num_features, kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList(
            [ResnetBlock(num_features) for _ in range(num_blocks)]
        )
        if max_pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.max_pool = nn.Identity()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)
        return x


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    def __init__(
        self,
        width=1,
        stack_sizes=(16, 32, 32),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=2,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(2, 65, 65),
    ):
        super(ImpalaEncoder, self).__init__()
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.input_shape = input_shape
        self.mlp_output_dim = mlp_output_dim

        input_channels = [input_channels] + list(stack_sizes)

        self.stack_blocks = nn.ModuleList(
            [
                ResnetStack(
                    input_channels=input_channels[i],
                    num_features=stack_size * width,
                    num_blocks=num_blocks,
                )
                for i, stack_size in enumerate(stack_sizes)
            ]
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()

        # Compute MLP input dimension dynamically
        with torch.no_grad():
            # Create a dummy input (assuming typical input size for this encoder)
            dummy_input = torch.zeros(1, *self.input_shape)  # (1, C, H, W)
            conv_out = dummy_input
            for stack_block in self.stack_blocks:
                conv_out = stack_block(conv_out)  # b c w h
            flattened_dim = conv_out.view(conv_out.size(0), -1).shape[1]  # c * w * h

        self.mlp = nn.Linear(flattened_dim, self.mlp_output_dim)

        if final_ln:
            self.final_ln = nn.LayerNorm(self.mlp_output_dim)
        else:
            self.final_ln = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            out: [B, D, T, 1, 1]
        """

        # [B, C, T, H, W] --> [T, B, C, H, W]
        (
            _,
            _,
            t,
            _,
            _,
        ) = x.shape
        x = x.permute(2, 0, 1, 3, 4)

        features = []

        for i in range(t):

            conv_out = x[i]

            for i, stack_block in enumerate(self.stack_blocks):
                conv_out = stack_block(conv_out)
                if self.dropout_rate is not None:
                    conv_out = self.dropout(conv_out)

            conv_out = F.relu(conv_out)
            if self.layer_norm:
                conv_out = nn.LayerNorm(conv_out.size()[1:])(conv_out)  # b c w h
            # flatten
            out = conv_out.view(conv_out.size(0), -1)
            out = self.mlp(out)
            out = self.final_ln(out)

            features.append(out)

        features = torch.stack(features, dim=1)

        features = features.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)

        return features


class RNNPredictor(nn.Module):
    """GRU-based predictor for single-step state propagation."""

    def __init__(
        self,
        hidden_size: int = 512,
        action_dim: Optional[int] = 2,
        num_layers: int = 1,
        final_ln: Optional[torch.nn.Module] = None,
    ):
        super(RNNPredictor, self).__init__()

        self.num_layers = num_layers

        self.rnn = torch.nn.GRU(
            input_size=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.final_ln = final_ln
        self.is_rnn = True
        self.context_length = 0

    def forward(self, state, action):
        """
        Propagate one step forward.

        Args:
            state: [B, D, 1, 1, 1]
            action: [B, A, 1]
        Returns:
            next_state: [B, D, 1, 1, 1]
        """
        # This only does one step
        rnn_state = state.flatten(1, 4).unsqueeze(0).contiguous()  # [1, B, D]
        rnn_input = action.squeeze(-1).unsqueeze(0).contiguous()  # [1, B, A]

        next_state, _ = self.rnn(rnn_input, rnn_state)

        next_state = self.final_ln(next_state)

        return next_state[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class SetTransformerEncoder(nn.Module):
    """Permutation-invariant set-transformer encoder over OTU tokens.

    Microbiome encoder f_theta for the EB-JEPA world model. Consumes a *set* of
    OTU tokens per timepoint (order does not matter, padded slots are ignored)
    and produces ONE community vector per timepoint. There is NO positional
    encoding, so the function is permutation invariant over the OTU dimension by
    construction.

    OBS / TOKEN CONTRACT (shared with the data workstream):
        obs is a dict:
          - obs["otu"]:  FloatTensor [B, T, N_max, F]  (features, F = token_dim)
          - obs["mask"]: BoolTensor  [B, T, N_max]     (True = real OTU, False = pad)
        Features are assumed to arrive already CLR'd + per-dim z-scored.

    Output:
        Tensor [B, D, T, 1, 1] where D == self.mlp_output_dim. This matches the
        ImpalaEncoder output convention (H'=W'=1), so the predictor, the
        VC_IDM_Sim_Regularizer ([B,C,T,H,W]), and the planning machinery all work
        unchanged. The builder reads self.mlp_output_dim to set the predictor's
        hidden_size = D, and may read self.final_ln (exposed below).

    Args:
        token_dim (int): per-OTU feature dimension F (default 385 = 384 ProkBERT + 1 CLR).
        d_model (int): transformer working width.
        n_heads (int): attention heads.
        n_layers (int): number of TransformerEncoder layers.
        dim_feedforward (int): FFN width inside each transformer layer.
        dropout (float): dropout inside the transformer (use 0.0 for deterministic
            permutation/mask-invariance checks).
        pool (str): "mean" for masked mean pooling, or "attention" for a learned
            attention (PMA-style) pool. Both are permutation invariant.
        mlp_output_dim (Optional[int]): if given, a final Linear projects the pooled
            d_model vector to this size and D = mlp_output_dim; otherwise D = d_model.
        final_ln (bool): apply a LayerNorm to the output community vector.
    """

    def __init__(
        self,
        token_dim: int = 385,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        pool: str = "mean",
        mlp_output_dim: Optional[int] = None,
        final_ln: bool = True,
    ):
        super().__init__()
        if pool not in ("mean", "attention"):
            raise ValueError(f"pool must be 'mean' or 'attention', got {pool!r}")

        self.token_dim = token_dim
        self.d_model = d_model
        self.pool = pool

        # Token embedding: F -> d_model (no positional encoding on purpose).
        self.input_proj = nn.Linear(token_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Learned attention pooling (PMA with a single seed vector). Permutation
        # invariant: scores depend only on per-token content, and the softmax +
        # weighted sum is order independent.
        if pool == "attention":
            self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.pool_query, std=0.02)
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )

        # Optional final projection so the builder can pick the embedding dim D.
        if mlp_output_dim is not None and mlp_output_dim != d_model:
            self.output_proj = nn.Linear(d_model, mlp_output_dim)
            out_dim = mlp_output_dim
        else:
            self.output_proj = nn.Identity()
            out_dim = d_model

        # D = encoder output dim; the builder reads this to size the predictor.
        self.mlp_output_dim = out_dim

        if final_ln:
            self.final_ln = nn.LayerNorm(out_dim)
        else:
            self.final_ln = nn.Identity()

        # Match the codebase init convention (truncated-normal Linear weights).
        self.apply(init_module_weights)

    def forward_set(self, tokens, mask):
        """Encode a batch of OTU sets into community vectors.

        Args:
            tokens: [B*, N_max, F] OTU features.
            mask:   [B*, N_max] bool, True = real OTU, False = pad.
        Returns:
            [B*, D] community vectors (D == self.mlp_output_dim).
        """
        # Guard against padded slots leaking into the network through the input
        # projection bias / attention: zero out features in padded positions so
        # the output is invariant to whatever junk sits in pad slots.
        keep = mask.unsqueeze(-1).to(tokens.dtype)  # [B*, N_max, 1]
        tokens = tokens * keep

        h = self.input_proj(tokens)  # [B*, N_max, d_model]

        # src_key_padding_mask: True marks positions to IGNORE -> pass ~mask.
        pad_mask = ~mask  # [B*, N_max]
        h = self.transformer(h, src_key_padding_mask=pad_mask)  # [B*, N_max, d_model]

        if self.pool == "mean":
            # Masked mean over real OTUs only (permutation invariant). Clamp the
            # denominator so a community with >=1 real OTU never divides by zero;
            # the contract guarantees at least one real OTU per row.
            keep = mask.unsqueeze(-1).to(h.dtype)  # [B*, N_max, 1]
            h = h * keep
            denom = keep.sum(dim=1).clamp(min=1.0)  # [B*, 1]
            pooled = h.sum(dim=1) / denom  # [B*, d_model]
        else:  # attention pooling (PMA, single learned query)
            bstar = h.shape[0]
            q = self.pool_query.expand(bstar, -1, -1)  # [B*, 1, d_model]
            pooled, _ = self.pool_attn(
                q, h, h, key_padding_mask=pad_mask, need_weights=False
            )
            pooled = pooled.squeeze(1)  # [B*, d_model]

        pooled = self.output_proj(pooled)  # [B*, D]
        pooled = self.final_ln(pooled)
        return pooled

    def forward(self, obs):
        """Encode a microbiome trajectory to the EB-JEPA state convention.

        Args:
            obs: dict with
                "otu":  FloatTensor [B, T, N_max, F]
                "mask": BoolTensor  [B, T, N_max] (True = real OTU)
        Returns:
            Tensor [B, D, T, 1, 1].
        """
        otu = obs["otu"]
        mask = obs["mask"]

        b, t, n_max, f = otu.shape

        # Fold T into the batch so the set-transformer just sees [B*T, N_max, F],
        # mirroring how TemporalBatchMixin folds time for the conv encoders.
        tokens = otu.reshape(b * t, n_max, f)
        mask_flat = mask.reshape(b * t, n_max)

        pooled = self.forward_set(tokens, mask_flat)  # [B*T, D]
        d = pooled.shape[-1]

        # [B*T, D] -> [B, T, D] -> [B, D, T] -> [B, D, T, 1, 1]
        pooled = pooled.reshape(b, t, d)
        out = pooled.permute(0, 2, 1).contiguous()  # [B, D, T]
        out = out.unsqueeze(-1).unsqueeze(-1)  # [B, D, T, 1, 1]
        return out


class InverseDynamicsModel(nn.Module):
    """
    Predicts the action that caused a transition from state_t to state_t_plus_1.
    Used as auxiliary task for representation learning.
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(init_module_weights)

    def forward(self, state_t, state_t_plus_1):
        """
        Args:
            state_t: State at time t, shape [B, D]
            state_t_plus_1: State at time t+1, shape [B, D]
        Returns:
            predicted_action: Action predicted to transform state_t to state_t_plus_1, shape [B, A]
        """
        combined_states = torch.cat([state_t, state_t_plus_1], dim=1)
        return self.model(combined_states)
