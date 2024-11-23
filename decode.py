import argparse
import configargparse
import json
import sys
import os
import jiwer
import logging
import torch

from models.asr_model import ASRModel
from loader import create_loader
from utils import to_device
import sentencepiece as spm


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_parser(parser=None, required=True):
    if parser is None:
        parser = configargparse.ArgumentParser(
            description="Decode a trained model",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )

    parser.add("--config", is_config_file=True, help="config file path")

    parser.add_argument(
        "--recog_json",
        default="dump/raw/test_bilingual/data.json",
        type=str,
        help="Recognition Features",
    )
    parser.add_argument(
        "--exp_dir", default="exp/train_001", type=str, help="Training exp directory"
    )
    parser.add_argument(
        "--ckpt_name", type=str, help="Checkpoint name"
    )
    parser.add_argument(
        "--bpe",
        type=str,
        default="bpe.model",
        help="Filename of the dictionary/vocabulary file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="monolingual",
        help="Controls if LID tags should be added. Can be [multilingual, monolingual]",
    )
    parser.add_argument("--decode_tag", default="test", type=str, help="Decoding tag")

    return parser


def main(cmd_args):
    ## Return the arguments from parser
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)

    ## Prepare logging
    log_dir = os.path.join(args.exp_dir, f"decode_{args.decode_tag}_{args.ckpt_name}")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "decode.log"),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    ## Load data json
    with open(args.recog_json, "rb") as f:
        recog_json = json.load(f)

    s = spm.SentencePieceProcessor(model_file=args.bpe)
    args.bpe = s

    ## Load model
    with open(os.path.join(args.exp_dir, "model.json"), "rb") as f:
        train_params = json.load(f)
    train_params = argparse.Namespace(**train_params)
    model = ASRModel(train_params)
    checkpoint = torch.load(os.path.join(args.exp_dir, "ckpts", args.ckpt_name))
    model.load_state_dict(checkpoint["model_state_dict"])
    if torch.cuda.is_available():
        model = model.cuda()
    logging.info(
        f'Load model from {os.path.join(args.exp_dir, "ckpts", args.ckpt_name)}'
    )
    logging.info(str(model))

    ## Create decoding loader
    _, test_loader, _ = create_loader(recog_json, train_params, is_train=False)

    ## Start decoding
    output_dict = {}
    target_dict = {}
    with torch.no_grad():
        model.eval()
        for i, (feats, feat_lens, target, target_lens, test_keys) in enumerate(
            test_loader
        ):
            logging.info(f"Batch {i}")
            if target is not None:
                feats, feat_lens, target, target_lens = to_device(
                    (feats, feat_lens, target, target_lens),
                    next(model.parameters()).device,
                )
            else:
                feats, feat_lens = to_device(
                    (feats, feat_lens), next(model.parameters()).device
                )

            # preds: list of lists of ints
            preds = model.decode_greedy(feats, feat_lens)

            for key, pred in zip(test_keys, preds):
                tokens = [x for x in pred if x != train_params.text_pad]
                tokens = args.bpe.decode(tokens)

                output_dict[key] = tokens.strip()
                print(output_dict[key])

            if target is not None:
                for key, ref in zip(test_keys, target):
                    tokens = [x for x in ref if x != train_params.text_pad]
                    tokens = args.bpe.decode(tokens)
                    target_dict[key] = tokens.strip()

    # Write hypotheses to a file
    output_file = os.path.join(log_dir, "decoded_hyp.txt")
    with open(output_file, "w") as fp:
        for key in sorted(list(output_dict.keys())):
            fp.write(f"{key} {output_dict[key]}\n")

    # Calculate WER if target exists
    if len(target_dict) > 0:
        all_refs = []
        all_hyps = []
        for key in sorted(list(output_dict.keys())):
            all_hyps.append(output_dict[key])
            all_refs.append(target_dict[key])

        wer = jiwer.wer(all_refs, all_hyps)
        logging.info(f"Decode: wer={wer:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
