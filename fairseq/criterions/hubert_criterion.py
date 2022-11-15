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


@register_criterion("hubert", dataclass=HubertCriterionConfig)
class HubertCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        pred_masked_weight,
        pred_nomask_weight,
        loss_weights=None,
        log_keys=None,
    ):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight  #1
        self.pred_nomask_weight = pred_nomask_weight  #0
        self.loss_weights = loss_weights  #[10.0]
        self.log_keys = [] if log_keys is None else log_keys

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(target_list=sample["target_list"], **sample["net_input"]) #logit_m_list 和 logit_u_list
        loss = 0.0
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"

        loss_m_list = []
        logp_m_list = model.get_logits(net_output, True) #[(1936,505)]
        targ_m_list = model.get_targets(net_output, True) ##(1936,) 全0  scalar的值则是真实类别的index!! 之前存的时候y都是存在第0位 看那个网页 真神奇
        assert self.pred_masked_weight == 0 or len(logp_m_list) > 0
        for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
            loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
            loss_m_list.append(loss_m)
            logging_output[f"loss_m_{i}"] = loss_m.detach().item()  #{'loss_m_0': 12147.861328125}
        if self.pred_masked_weight > 0:  #self.pred_masked_weight=1
            loss += self.pred_masked_weight * sum(loss_m_list)  #一个batch的m loss!
            sample_size += targ_m_list[0].numel()  #1936

        loss_u_list = []
        logp_u_list = model.get_logits(net_output, False)
        targ_u_list = model.get_targets(net_output, False)
        assert self.pred_nomask_weight == 0 or len(logp_u_list) > 0
        for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
            loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
            loss_u_list.append(loss_u)
            logging_output[f"loss_u_{i}"] = loss_u.detach().item()
        if self.pred_nomask_weight > 0:   #self.pred_nomask_weight=0 x  哦哦如果不想要mask 应该把它改成1 把上面的改成0改才对
            loss += self.pred_nomask_weight * sum(loss_u_list)   # 其实只需要改self.pred_nomask_weight！！！
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

        for lk in self.log_keys:  #x
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        def compute_correct(logits):
            if logits.numel() == 0:
                return 0, 0
            else:
                assert logits.dim() > 1, logits.shape
                max = logits.argmax(-1) == 0  #(1936,bool)   ( logits.argmax(-1) 是[1936] ，所以是在判断 相似度最大的项是不是和第0项一样，以此来判断是不是选对了，如果对了 那个sample赋值true   #有两个是true
                min = logits.argmin(-1) == 0 #(1936,)
                both = max & min
                corr = max.long().sum().item() - both.long().sum().item() #2 说明1936个time steps中只分类对了2个  max=min 所有项的值都一样也不算分对
                count = max.numel() #1936
                return corr, count

        with torch.no_grad():
            for i, logp_m in enumerate(logp_m_list):
                corr_m, count_m = compute_correct(logp_m)  #2，1936  1936个time steps中只分类对了2个
                logging_output[f"correct_m_{i}"] = corr_m
                logging_output[f"count_m_{i}"] = count_m

            for i, logp_u in enumerate(logp_u_list):
                corr_u, count_u = compute_correct(logp_u)  #5，1880
                logging_output[f"correct_u_{i}"] = corr_u
                logging_output[f"count_u_{i}"] = count_u

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)  #0
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)  #0
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)  #0

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
                tmp=counts[re.sub("correct", "count", lk)]
                if tmp!=0:
                    metrics.log_scalar(lk, val / tmp)
                
        #add log  ntokens,nsentences
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
