import numpy as np
import torch

def fake(a):
    return a+1

features=[]
source=torch.ones([4,2,2])
print(source.shape)
print(source)
for i in range(source.shape[0]):
    source[i, :, :]=fake(source[i, :, :])

print(source.shape)
print(source)
# for i in range(source.shape[0]):
#     features.append(source[i,:,:])
# fbank_features_test1 = torch.stack(features, dim=0)
# print(fbank_features_test1.shape)   #(8,2,2)
#
# for i in range(source.shape[0]):
#     if i == 0:
#         source_first = source[i, :, :]  # （1614，80）
#         # source_first = self.specaug_transform(
#         #     source_first)  # AssertionError: spectrogram must be a 2-D tensor. 所以只能一个个做
#         source_first = source_first.unsqueeze(0)
#     else:
#         source_tmp = source[i, :, :]
#         # source_tmp = self.specaug_transform(source_tmp)
#         source_tmp = source_tmp.unsqueeze(0)
#         source_first = torch.cat((source_first, source_tmp))
#
# print(source_first.shape)
#
# if ((fbank_features_test1==source_first).all()):
#     print("same")
# # print(t1.shape)
# print(t1)
# exit(0)
#
#
# torch=torch.ones([2,2,3])
# new=torch.Tensor([torch1,torch])
# print(new)
# exit(0)
# print(torch1)
# print(torch)
# print(torch.shape)  #(2,3

# torch1=torch.Tensor([[1,2,3],[4,1,2]])
# l1=[torch1]
# torch2=torch.Tensor([1,2,3])
# l2=[torch2]
# a=torch1.argmax(dim=-1)
# targets=torch.Tensor([2,0])
# print(a)
# corr = (a==targets).sum().item()
# print(corr)
# exit(0)
# for t1,t2 in zip(l1,l2):
#     print(t1,t2)
#  t1 = torch.FloatTensor(7,1143,80)
# t2 = torch.FloatTensor(1143,80)
# t3=torch.full([t1.shape[0]],t1.shape[1])
# print(t3)
# lista=[t1,t2]
# ta = torch.cat(lista, dim=0).reshape(len(lista),-1,80)
# print(ta.shape)
# print(ta)

#torch.Size([2, 1143, 80])

# stats_npz_path1 = "/data/ygr/global_cmvn.npy"  # 原始路径 /mnt/lustre/sjtu/home/xc915/data/Gigaspeech/gigaspeech-s-rawwav/global_cmvn.npy “两个是一样的”
# stats1 = np.load(stats_npz_path1, allow_pickle=True).tolist()
# print(stats1["mean"], stats1["std"])
#
#
# stats_npz_path2 = "/data/ygr/librispeech/npyfiles/global_cmvn.npy"  # 原始路径 /mnt/lustre/sjtu/home/xc915/data/Gigaspeech/gigaspeech-s-rawwav/global_cmvn.npy “两个是一样的”
# stats2 = np.load(stats_npz_path2, allow_pickle=True).tolist()
# print(stats2["mean"], stats2["std"])
#
# stats_npz_path3 = "/data/ygr/librispeech/npyfiles.rank0/global_cmvn.npy"  # 原始路径 /mnt/lustre/sjtu/home/xc915/data/Gigaspeech/gigaspeech-s-rawwav/global_cmvn.npy “两个是一样的”
# stats3 = np.load(stats_npz_path3, allow_pickle=True).tolist()
# #print(stats3["mean"], stats3["std"])
#
# print(stats1["mean"]==stats2["mean"].all())
# print(stats2["mean"]==stats3["mean"].all())

# if stats1["mean"]==stats2["mean"].all():
#     print("stats1_mean==stats2_mean")
# if stats2["mean"]==stats3["mean"]:
#     print("stats2_mean==stats3_mean")
#
# if stats1["std"]==stats2["std"]:
#     print("stats1_std==stats2_std")
# if stats2["std"]==stats3["std"]:
#     print("stats2_std==stats3_std")