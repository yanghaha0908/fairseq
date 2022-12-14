# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import torch.nn as nn

@dataclass
class HubertCriterionConfig(FairseqDataclass):
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "weight for predictive loss for masked frames"},
    )
    pred_nomask_weight: float = field(
        default=0.0,
        metadata={"help": "weight for predictive loss for unmasked frames"},
    )
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )


@register_criterion("hubertce", dataclass=HubertCriterionConfig)
class HubertCECriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            pred_masked_weight,
            pred_nomask_weight,
            loss_weights=None,
            log_keys=None,
    ):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight  # 1
        self.pred_nomask_weight = pred_nomask_weight  # 0
        self.loss_weights = loss_weights  # [10.0]
        self.log_keys = [] if log_keys is None else log_keys

        #new


    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(target_list=sample["target_list"], **sample["net_input"])  # logit_m_list ??? logit_u_list
        loss = 0.0   #???x:8,477,768) ,padding_mask(8,477),features:8,477,768)
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"

        loss_m_list = []
        logp_m_list = model.get_logits(net_output, True) #[(1936,504)]
        targ_m_list = net_output["targ_m_list"]  #tuple:1 (1936,)
        assert self.pred_masked_weight == 0 or len(logp_m_list) > 0
        for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
            loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)  #torch.float32,torch.int64
            loss_m_list.append(loss_m)
            logging_output[f"loss_m_{i}"] = loss_m.detach().item()  # {'loss_m_0': 12147.861328125}
        if self.pred_masked_weight > 0:  # self.pred_masked_weight=1
            loss += self.pred_masked_weight * sum(loss_m_list)  # ??????batch???m loss!
            sample_size += targ_m_list[0].numel()  # 1936

        loss_u_list = []
        logp_u_list = model.get_logits(net_output, False)
        targ_u_list = net_output["targ_u_list"]
        assert self.pred_nomask_weight == 0 or len(logp_u_list) > 0
        for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
            loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
            loss_u_list.append(loss_u)
            logging_output[f"loss_u_{i}"] = loss_u.detach().item()
        if self.pred_nomask_weight > 0:   #self.pred_nomask_weight=0 x  ?????????????????????mask ??????????????????1 ??????????????????0?????????
            loss += self.pred_nomask_weight * sum(loss_u_list)   # ??????????????????self.pred_nomask_weight?????????
            sample_size += targ_u_list[0].numel()

        if self.loss_weights is not None:   #self.loss_weights:[10]
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(net_output) #feature_pen
            if torch.is_tensor(extra_losses): #x
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1: #x
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, n, coef in zip(extra_losses, names, self.loss_weights): #p:0.2904, n='features_pen',coef=10.0
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size    #1936   #sample_size =0
                    loss += p   #0
                    logging_output[f"loss_{n}"] = p.item()




        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            **logging_output,
        }

        for lk in self.log_keys:  # x
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        def compute_correct(logits,targets):   #??????????????? ??????????????????
            if logits.numel() == 0:
                return 0, 0
            else:
                assert logits.dim() > 1, logits.shape
                max_index = logits.argmax(dim=1)  # 1936
                corr = (max_index==targets).sum().item() # 3
                count = len(targets)  # 1936  ??????tensor????????????len ???numel  #numel()??????:??????????????????????????????
                return corr, count

        with torch.no_grad():
            for logp_m,tar_m in zip(logp_m_list,targ_m_list):
                corr_m, count_m = compute_correct(logp_m,tar_m)  # 2???1936  1936???time steps??????????????????2???  ??????????????????int
                logging_output[f"correct_m_{i}"] = corr_m
                logging_output[f"count_m_{i}"] = count_m

            for logp_u,tar_u in zip(logp_u_list,targ_u_list):
                corr_u, count_u = compute_correct(logp_u,tar_u)  # 5???1880
                logging_output[f"correct_u_{i}"] = corr_u
                logging_output[f"count_u_{i}"] = count_u

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)  # 0
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)  # 0
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)  # 0

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log[lk] for log in logging_outputs)
                tmp = counts[re.sub("correct", "count", lk)]
                if tmp != 0:
                    metrics.log_scalar(lk, val / tmp)

        # add log  ntokens,nsentences
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
