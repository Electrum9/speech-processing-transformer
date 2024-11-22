import torch
from torch import nn
from typing import Optional, Tuple

from models.frontend import DefaultFrontend

from models.layers import (
    LayerNorm,
    Conv2dSubsampling,
    PositionwiseFeedForward,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    ConvolutionModule
)
from utils import make_pad_mask, repeat


class TransformerEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        conv,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.conv = conv

    def forward(self, x, mask):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        if isinstance(x, tuple):
            x, pos_emb = x[0], x[1]
        else:
            x, pos_emb = x, None
        # TODO: attention with residual connection
        # mask is used in the attention module
        # x -> norm1 -> att -> dropout -> + -> x
        # |_______________________________|

        x_norm = self.norm1(x)
        x_att = self.self_attn(x_norm)
        x_drop = self.dropout(x_att)
        x = x + x_drop

        # TODO: convolution layer 
        # x -> conv -> x
        # this defaults to nn.Identity()
        # unless conformer_kernel_size > 0 in the encoder

        x = self.conv(x)

        # TODO: feed-forward network with residual connection
        # x -> norm2 -> ffn -> dropout -> + -> x
        # |_______________________________|

        x_norm2 = self.norm2(x)
        x_ffn = self.feed_forward(x_norm2)
        x_drop2 = self.dropout(x_ffn)
        x = x + x_drop2

        if pos_emb is not None:
            x = (x, pos_emb)
        return x, mask

class TransformerEncoder(torch.nn.Module):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        position_embedding_type: str = "absolute", # can be [relative, absolute, none]
        conformer_kernel_size: int = 0,
    ):
        super().__init__()
        self._output_size = output_size
        self.frontend = DefaultFrontend()
        self.embed = Conv2dSubsampling(input_size, output_size, positional_dropout_rate, position_embedding_type)

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
        )

        if position_embedding_type == 'absolute':
            attn_class = MultiHeadedAttention
        elif position_embedding_type == 'relative':
            attn_class = RelPositionMultiHeadedAttention

        self.encoders = repeat(
            num_blocks,
            lambda lnum: TransformerEncoderLayer(
                output_size,
                attn_class(
                    attention_heads, output_size, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                ConvolutionModule(output_size, conformer_kernel_size, nn.GELU()) if conformer_kernel_size > 0 else torch.nn.Identity()
            ),
        )
        self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
        Returns:
            position embedded tensor and mask
        """

        # wav -> mel-filterbank
        xs_pad, ilens = self.frontend(xs_pad, ilens)
        
        # prepare masks
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if isinstance(xs_pad, tuple):
            xs_pad, pos_emb = xs_pad[0], xs_pad[1]

        # TODO: apply convolutional subsampling, i.e., self.embed
        xs_pad = self.embed(xs_pad)

        # TODO: forward encoder layers
        xs_pad = self.encoders(xs_pad)

        # apply another layer norm at the end
        xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)

        return xs_pad, olens
