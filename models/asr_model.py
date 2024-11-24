import configargparse
import torch

from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from utils import add_sos_eos, LabelSmoothingLoss, make_pad_mask


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
        xs, xs_lens = self.encoder(xs, xlens)

        # 2. Compute Loss by calling self.calculate_loss()
        loss_att, acc = self.calculate_loss(xs, xs_lens, ys, ylens)

        return loss_att, acc

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
        out = self.decoder(encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens)

        # 2. Compute attention loss using self.criterion_att()
        loss_att = self.criterion_att(out, ys_out_pad)

        eq = (torch.argmax(out, dim=-1) == ys_out_pad)
        mask = ~make_pad_mask(ys_in_lens)
        eq = eq * mask.cuda()
        acc = eq.sum(axis=-1) / ys_in_lens
        
        return loss_att, acc

    def decode_greedy(self, xs, xlens):
        """Perform Greedy Decoding using trained ASRModel

        :params torch.Tensor xs- Speech feature input, (batch, time, dim)
        :params list xlens- Lengths of unpadded feature sequences, (batch,)
        """

        batch, *_ = xs.shape
        xlens = torch.tensor(xlens, dtype=torch.long, device=xs.device)
        xs, xs_lens = self.encoder(xs, xlens)

        candidates = [[(0.0, [self.sos], None)] for _ in range(batch)]
        eos_candidates = [[] for _ in range(batch)] # Reached EOS

        for i in range(max_decode_len):
            new_candidates = [[] for j in range(batch)]

            for b in range(batch):
                for score, seq, cache in candidates[b]:
                    ys_in_pad = torch.tensor([seq], dtype=torch.long, device=xs.device)
                    ys_in_lens = torch.tensor([len(seq)], dtype=torch.long, device=xs.device)

                    scores, cache = self.decoder.forward_one_step(xs[b:b+1], xs_lens[b:b+1], ys_in_pad, ys_in_lens, cache)

                    log_probs = torch.log_softmax(scores[0, -1], dim=-1)
                    top_k_log_probs, top_k_indices = log_probs.topk(beam_size)

                    for log_prob, idx in zip(top_k_log_probs, top_k_indices):
                        new_score = score + log_prob.item()
                        new_seq = seq + [idx.item()]

                        if idx.item() == self.eos:
                            eos_candidates[b].append((new_score, new_seq))
                        else:
                            new_candidates[b].append((new_score, new_seq, cache))

            for b in range(batch):
                candidates[b] = sorted(new_candidates[b], key=lambda x: x[0], reverse=True)[:beam_size] # cull to top k

            if all(len(eos_candidates[b]) >= beam_size for b in range(batch)):
                break

        predictions = []
        for b in range(batch):
            eos_candidates[b].sort(key=lambda x: x[0], reverse=True)
            best_hypothesis = eos_candidates[b][0][1] if eos_candidates[b] else candidates[b][0][1]
            predictions.append(best_hypothesis[1:-1]) # get rid of sos/eos

        return predictions
