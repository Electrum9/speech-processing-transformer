import configargparse
import torch

from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from utils import add_sos_eos, LabelSmoothingLoss


class ASRModel(torch.nn.Module):
    def __init__(self, params: configargparse.Namespace):
        """E2E ASR model implementation.

        Args:
            params: The training options
        """
        super().__init__()

        self.ignore_id = params.text_pad
        self.sos = params.odim - 1
        self.eos = self.sos

        self.encoder = TransformerEncoder(
            input_size=params.idim,
            output_size=params.hidden_dim,
            attention_heads=params.attention_heads,
            linear_units=params.linear_units,
            num_blocks=params.eblocks,
            dropout_rate=params.edropout,
            positional_dropout_rate=params.edropout,
            attention_dropout_rate=params.edropout,
            position_embedding_type=params.eposition_embedding_type,
            conformer_kernel_size=params.econformer_kernel_size,
        )
        self.decoder = TransformerDecoder(
            vocab_size=params.odim,
            encoder_output_size=params.hidden_dim,
            attention_heads=params.attention_heads,
            linear_units=params.linear_units,
            num_blocks=params.dblocks,
            dropout_rate=params.ddropout,
            positional_dropout_rate=params.ddropout,
            self_attention_dropout_rate=params.ddropout,
            src_attention_dropout_rate=params.ddropout,
        )
        self.criterion_att = LabelSmoothingLoss(
            size=params.odim,
            padding_idx=self.ignore_id,
            smoothing=params.label_smoothing,
            normalize_length=False,
        )

    def forward(
        self,
        xs,
        xlens,
        ys,
        ylens,
    ):
        """Forward propogation for ASRModel

        :params torch.Tensor xs- Speech feature input
        :params list xlens- Lengths of unpadded feature sequences
        :params torch.LongTensor ys_ref- Padded Text Tokens
        :params list ylen- Lengths of unpadded text sequences
        """
        xlens = torch.tensor(xlens, dtype=torch.long, device=xs.device)
        ylens = torch.tensor(ylens, dtype=torch.long, device=xs.device)

        # TODO: implement forward of the ASR model

        # 1. Encoder forward (CNN + Transformer)

        # 2. Compute Loss by calling self.calculate_loss()

        return loss_att

    def calculate_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # TODO: Implement decoder forward + loss calculation

        # 1. Forward decoder

        # 2. Compute attention loss using self.criterion_att()
        
        return loss_att

    def decode_greedy(self, xs, xlens):
        """Perform Greedy Decoding using trained ASRModel

        :params torch.Tensor xs- Speech feature input, (batch, time, dim)
        :params list xlens- Lengths of unpadded feature sequences, (batch,)
        """

        xlens = torch.tensor(xlens, dtype=torch.long, device=xs.device)

        # TODO: Encoder forward (CNN + Transformer)

        # TODO: implement greedy decoding
        # Hints:
        # - Start from <sos> and predict new tokens step-by-step until <eos>. You need a loop.
        # - You may need to set a maximum decoding length.
        # - You can use self.decoder.forward_one_step() for each step which has caches

        return predictions
