# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import II

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data.data_utils import post_process
from fairseq.dataclass import FairseqDataclass
from fairseq.logging.meters import safe_round
from fairseq.tasks import FairseqTask


@dataclass
class CtcCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )


@register_criterion("ctc", dataclass=CtcCriterionConfig)
class CtcCriterion(FairseqCriterion):
    def __init__(
        self, cfg: CtcCriterionConfig, task: FairseqTask, rdrop_alpha: int = 0.0
    ):
        super().__init__(task)
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        ) #0
        self.pad_idx = task.target_dictionary.pad() #1
        self.eos_idx = task.target_dictionary.eos() #2
        self.post_process = cfg.post_process #'letter'

        self.rdrop_alpha = rdrop_alpha #0

        if cfg.wer_args is not None: #x
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None and cfg.wer_kenlm_model != "":
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else: #
            self.w2l_decoder = None

        self.zero_infinity = cfg.zero_infinity #True
        self.sentence_avg = cfg.sentence_avg

    def forward(self, model, sample, reduce=True, **kwargs):
        net_output = model(**sample["net_input"]) #dict:3 'encoder_out'=(686,8,32)   # model.eval() 之后 model.training 就会变成false！
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder  (686,8,32) 过了一个log_softmax

        # CTC loss is calculated over duplicated inputs
        # sample is already duplicated for R-Drop
        if self.rdrop_alpha > 0:  #没做
            for k, v in sample.items():
                if k in ["target", "target_lengths"]:
                    sample[k] = torch.cat([v, v.clone()], dim=0)
                elif k == "net_input":
                    if sample[k]["src_tokens"].size(1) != sample[k]["src_lengths"].size(
                        0
                    ):
                        # for decoder CTC loss
                        sample[k]["src_lengths"] = torch.cat(
                            [
                                sample[k]["src_lengths"],
                                sample[k]["src_lengths"].clone(),
                            ],
                            dim=0,
                        )

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else: #
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]  #
                input_lengths = non_padding_mask.long().sum(-1)  #tensor([686, 686, 686, 686, 686, 686, 686, 686]
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )
        # sample[target] (8,220)
        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx  #（8,220) 前[194,181,220,154,217,185,200,148]全是true，后面都是false 和target_lengt一摸一样
        )
        targets_flat = sample["target"].masked_select(pad_mask) #(1499,)  #把真的有的target挑出来
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]  #sample[tensor([194, 181, 220, 154, 217, 185, 200, 148], device='cuda:0') 和是1499
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,  #(686,8,32) 上面过完log_softmax的
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )  #1499

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),  #8
            "sample_size": sample_size,
        }

        if not model.training:  #(556,6,32)
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()  #(6,556,32)

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],  #(6,363) target 的 id
                    input_lengths,  #tensor([556, 556, 556, 556, 556, 556], device='cuda:0')
                ):  #lp (556,32) t(363,) inp_l(556)
                    lp = lp[:inp_l].unsqueeze(0)  #(1,556,32)

                    decoded = None
                    if self.w2l_decoder is not None:  #x
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]
                    #target
                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )  #全true
                    targ = t[p] #(353,)tensor
                    targ_units = self.task.target_dictionary.string(targ) #valid.ltr那个样子： L I N N E L L ' S | P I C T U R E S | A R E | A | S O R T | O F | U P | G U A R D S | A N D | A T | E M | P A I N T I N G S | A N D | M A S O N ' S | E X Q U I S I T E | I D Y L L S | A R E | A S | N A T I O N A L | A S | A | J I N G O | P O E M | M I S T E R | B I R K E T | F O S T E R ' S | L A N D S C A P E S | S M I L E | A T | O N E | M U C H | I N | T H E | S A M E | W A Y | T H A T | M I S T E R | C A R K E R | U S E D | T O | F L A S H | H I S | T E E T H | A N D | M I S T E R | J O H N | C O L L I E R | G I V E S | H I S | S I T T E R | A | C H E E R F U L | S L A P | O N | T H E | B A C K | B E F O R E | H E | S A Y S | L I K E | A | S H A M P O O E R | I N | A | T U R K I S H | B A T H | N E X T | M A N |
                    targ_units_arr = targ.tolist() #[15,10,9,9,5,15,15,27..] (363)
                    #predict
                    toks = lp.argmax(dim=-1).unique_consecutive() #(293,) 取每个时间步t最大的id，合并连续相同
                    pred_units_arr = toks[toks != self.blank_idx].tolist()  #self.blank_idx是0 是<s> # (289,) [16,
                                                                            #这里面是有4的 所以有分隔符
                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)  #明白了，targ和predict都没有去掉｜ 所以是一样的

                    targ_words = post_process(targ_units, self.post_process).split()  #["LINNELL'S", 'PICTURES', 'ARE', 'A', 'SORT', 'OF', 'UP', 'GUARDS', 'AND', 'AT', 'EM', 'PAINTINGS', 'AND', "MASON'S", 'EXQUISITE', 'IDYLLS', 'ARE', 'AS', 'NATIONAL', 'AS', 'A', 'JINGO', 'POEM', 'MISTER', 'BIRKET', "FOSTER'S", 'LANDSCAPES', 'SMILE', 'AT', 'ONE', 'MUCH', 'IN', 'THE', 'SAME', 'WAY', 'THAT', 'MISTER', 'CARKER', 'USED', 'TO', 'FLASH', 'HIS', 'TEETH', 'AND', 'MISTER', 'JOHN', 'COLLIER', 'GIVES', 'HIS', 'SITTER', 'A', 'CHEERFUL', 'SLAP', 'ON', 'THE', 'BACK', 'BEFORE', 'HE', 'SAYS', 'LIKE', 'A', 'SHAMPOOER', 'IN', 'A', 'TURKISH', 'BATH', 'NEXT', 'MAN']

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:  #这个
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
