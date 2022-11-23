# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
)
import io

logger = logging.getLogger(__name__)


def load_fbank(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()  # 'id	audio	n_frames	tgt_text	speaker' #第一行
        for ind, line in enumerate(f):   #ind:0  line:'1272/128104/1272-128104-0000.flac	93680'
            items = line.strip().split("\t") #['1272/128104/1272-128104-0000.flac', '93680']
            #assert len(items) == 3, line #change   #id  audio  n_frames  tgt_text  speaker
            sz = int(items[2])   #n_frames 584  #items[1]  '/mnt/data/librispeech/data4fairseq/npyfiles/dev-clean-fbank80/1272-128104-0000.npy'
            if min_keep is not None and sz < min_keep/160:  #change  #min_keep:32000, max_keep=None
                n_short += 1
            elif max_keep is not None and sz > max_keep/160: #change
                n_long += 1
            else:
                names.append("/data/ygr/librispeech/"+ items[1][35:] )  #change  #'/data/ygr/librispeech/npyfiles/dev-clean-fbank80/1272-128104-0000.npy'
                inds.append(ind)  # 0
                sizes.append(sz)
    tot = ind + 1   #train:281241 #valid:tot:2703 也是2703  #train:names[list:280531] #valid:names[list:2666] 那几个也是
    if min_keep==None:
        logger.info(
            (
                f"max_keep={max_keep}, min_keep={min_keep}, "
                f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
                f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
            )
        )
    else:
        logger.info(
            (
                f"max_keep={max_keep}, min_keep={min_keep/160}, "
                f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
                f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
            )
        )  #max_keep=None, min_keep=32000, loaded 2666, skipped 37 short and 0 long, longest-loaded=522320, shortest-loaded=32000
    return root, names, inds, tot, sizes


def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f] #2632，3012，2662,2765
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths)) #把code_lengths 累加，最前面加个0  [0,2632,5644,8306]
        offsets = [(offsets[i], offsets[i + 1]) for i in inds] #[ (0,2632),(2632,5644),(5644,8306)]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]  #{list:2703} #统计valid.km 每一行的长度
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]   #{list:2666}  #只取那些长度合适，留下来的
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate  #5.84
        dur_from_label = lengths[i] / label_rate #5.84
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


class HubertfbankDataset(FairseqDataset):  #change
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        single_target: bool = False,
    ):
        self.audio_root, self.audio_names, inds, tot, self.sizes = load_fbank(  #change
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )  # finetune时 max_keep=None, min_keep=None, loaded 2703, skipped 0 short and 0 long, longest-loaded=522320, shortest-loaded=23120
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, float)
            else label_rates
        )
        self.store_labels = store_labels
        if store_labels: #false
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]  #[ (0,2632),(2632,5644),(5644,8306) ]
        assert label_processors is None or len(label_processors) == self.num_labels
        for label_path, label_rate in zip(label_paths, self.label_rates):  #计算音频和label的时间长度，若超过0.1 invalid
            verify_label_lengths(
                self.sizes, sample_rate, label_path, label_rate, inds, tot
            )    #没有warning 说明完美匹配

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )   #consider
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )

    def get_fbank(self, index):  #change
        wav_path = self.audio_names[index]  #"/data/ygr/librispeech/npyfiles.rank4/train-fbank80/2929-85685-0079.npy"
        fbank = np.load(wav_path, allow_pickle=True)  # {ndarray:2972,80)}
        fbank = torch.from_numpy(fbank).float()  #{Tensor:(2972,80)}

        return fbank

    def get_label(self, index, label_idx): #index:58766,label_idx:0  #num_labels=1
        if self.store_labels:  #false
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f: #打开train.km   哦就是可能有多个label
                offset_s, offset_e = self.label_offsets_list[label_idx][index]  #offset_e:138931979  offset_s:138926373
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)  #str: "数字 数字 数字 " 把这一段的label读出来

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)  #按那个字典处理过的 处理过的  #(1486) [6,37,108,....,] {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '0': 4, '1': 5, '2': 6,
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]  #num_labels={1}

    def __getitem__(self, index):
        wav = self.get_fbank(index)  #change
        labels = self.get_labels(index)
        return {"id": index, "source": wav, "label_list": labels}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)   #3223，
            end = size - diff + start  #253223
        return wav[start:end, :], start  #change

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]    #{list:2} 一个是 （2972，80） 一个是（2790，80） 每一个都是{dict:3}   'id':58766 'source':  label_list={list:1}
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]  #list:2 （2972，80），2790，80）
        audio_sizes = [s.shape[0] for s in audios]  #change  #[2972, 2790]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else: #
            audio_size = min(min(audio_sizes), int(self.max_sample_size/160)) #change  #audio_size:250000   # 1562.5  self.max_sample_size:250000  在yaml里有
        collated_audios, padding_mask, audio_starts = self.collater_audio(    #这一步主要是裁剪
            audios, audio_size
        )    #测过了 反正是全false


        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]  # [[],[],[],[],[] ]五个label lists
        targets_list, lengths_list, ntokens_list = self.collater_label(  #截取后的label_list 长度list 这一个batch总label长度
            targets_by_label, audio_size, audio_starts
        )

        net_input = {"source":  collated_audios, "padding_mask": padding_mask}   #破案了破案了！ 之前没想明白改成了audios，audios是没裁剪，没padding 打包的！  #修改
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),  #tensor([ 58766, 252596,  85361, 250124,  71503])
            "net_input": net_input,
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size,80) #change    #(2,1562,80) #(5,250000)  全0  #80!
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)   #(5,250000)  全 false
            # if self.pad_audio else None
        )   #(2,1562,80)
        audio_starts = [0 for _ in audios] #[0, 0, 0, 0, 0]
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:  #短了 padding
                assert self.pad_audio  #False  已经取的是最短的了，所以不用补
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts  #collated_audios（464,1702,80）padding_mask (464,1702,80） 全false # audio_starts:[3223, 188605, 9448, 117415, 153946]  裁的那段的起点

    def collater_frm_label(self, targets, audio_size, audio_starts, label_rate, pad):
        assert label_rate > 0
        s2f = 0.5 #label_rate / self.sample_rate   #change   fbank的time shift默认是100 即10ms产生一个frame，一秒产生100个  而label的采样率为50 所以是0.5   #0.003125  #label_rate=50  self.sample_rate=16000
        frm_starts = [int(round(s * s2f)) for s in audio_starts]  #[10, 589, 30, 367, 481] [3223, 188605, 9448, 117415, 153946]
        frm_size = int(round(audio_size * s2f)) #851    原来的#781
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)   #判断一下 label 够不够frm_size   label原总长度-label_starts
        targets = [t[s : s + frm_size] for t, s in zip(targets, frm_starts)] # {list:5} (781,)   发现都够 等比例嘛  不应该出现不够的情况吧？
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets]) #tensor([781, 781, 781, 781, 781])
        ntokens = lengths.sum().item() #3905
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)   #没step这个！  #（464，878） 后面padding了1
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1.0:
                targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):    #测了一下 如果长度list是[100,44,55,8,300,9200] 输出[5 4 0 2 1 3]
        if self.shuffle:  #True
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav
