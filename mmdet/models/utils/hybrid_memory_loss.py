# Ge et al. Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID.  # noqa
# Written by Yixiao Ge.

import torch
import torch.nn.functional as F
from torch import autograd, nn
from abc import ABC 

from mmdet.utils import all_gather_tensor

# try:
# PyTorch >= 1.6 supports mixed precision training
from torch.cuda.amp import custom_fwd, custom_bwd
class HM(autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, indexes, features, momentum, aux_inputs, augmem_factor):
        ctx.features = features
        ctx.aux_inputs = aux_inputs
        ctx.augmem_factor = augmem_factor
        ctx.momentum = momentum
        outputs = inputs.mm(ctx.features.t())
        all_inputs = all_gather_tensor(inputs)
        all_indexes = all_gather_tensor(indexes)
        ctx.save_for_backward(all_inputs, all_indexes)
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
        
        # momentum update
        for i, (x, y) in enumerate(zip(inputs, indexes)):
            if ctx.aux_inputs is not None:
                num_aug = len(ctx.aux_inputs)
                features_y_aux = torch.stack([ ctx.aux_inputs[i_a][i] for i_a in range(num_aug)]) # torch.Size([2, 256])
                features_y_aux = torch.cat((x[None], features_y_aux), dim=0) # torch.Size([3, 256])
                feat_aux_sim = x[None].mm(features_y_aux.t()) / ctx.augmem_factor # torch.Size([1, 3])
                feat_aux_sim = feat_aux_sim.softmax(dim=1)
                x = (features_y_aux * feat_aux_sim.t()).sum(0)
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None, None, None
# except:
#     class HM(autograd.Function):

#         @staticmethod
#         def forward(ctx, inputs, indexes, features, momentum, aux_inputs):
#             ctx.features = features
#             ctx.aux_inputs = aux_inputs
#             ctx.momentum = momentum
#             outputs = inputs.mm(ctx.features.t())
#             all_inputs = all_gather_tensor(inputs)
#             all_indexes = all_gather_tensor(indexes)
#             ctx.save_for_backward(all_inputs, all_indexes)
#             return outputs

#         @staticmethod
#         def backward(ctx, grad_outputs):
#             inputs, indexes = ctx.saved_tensors
#             grad_inputs = None
#             if ctx.needs_input_grad[0]:
#                 grad_inputs = grad_outputs.mm(ctx.features)
            
#             # momentum update
#             for i, (x, y) in enumerate(zip(inputs, indexes)):
#                 if ctx.aux_inputs is not None:
#                     num_aug = len(ctx.aux_inputs)
#                     features_y_aux = torch.stack([ ctx.aux_inputs[i_a][i] for i_a in range(num_aug)])
#                     features_y_aux = torch.cat((x[None], features_y_aux), dim=0)
#                     feat_aux_sim = x[None].mm(features_y_aux.t()) / 0.5
#                     feat_aux_sim = feat_aux_sim.softmax(dim=1)
#                     x = (features_y_aux * feat_aux_sim.t()).sum()
#                 ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
#                 ctx.features[y] /= ctx.features[y].norm()

#             return grad_inputs, None, None, None, None

def hm(inputs, indexes, features, momentum=0.5, aux_inputs=None, augmem_factor=0.0):
    return HM.apply(
        inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device), aux_inputs, augmem_factor
    )

class HybridMemory(nn.Module):
    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_memory = num_memory

        self.momentum = momentum
        self.temp = temp

        self.idx = torch.zeros(num_memory).long()

        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())
    
    @torch.no_grad()
    def _init_ids(self, ids):
        self.idx.data.copy_(ids.long().to(self.labels.device))

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))
    
    @torch.no_grad()
    def get_cluster_ids(self, indexes):
        return self.labels[indexes].clone()

    def forward(self, results, indexes):
        inputs = results
        inputs = F.normalize(inputs, p=2, dim=1)

        # inputs: B*2048, features: N*2048
        inputs = hm(inputs, indexes, self.features, self.momentum) #B*N, similarity
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums

        targets = self.labels[indexes].clone()
        labels = self.labels.clone() #shape: N, unique label num: u

        sim = torch.zeros(labels.max() + 1, B).float().cuda() #u*B
        sim.index_add_(0, labels, inputs.t().contiguous()) #
        nums = torch.zeros(labels.max() + 1, 1).float().cuda() #many instances belong to a cluster, so calculate the number of instances in a cluster
        nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda()) #u*1
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim) #average features in each cluster, u*B
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous()) #sim: u*B, mask:u*B, masked_sim: B*u
        return F.nll_loss(torch.log(masked_sim + 1e-6), targets)


try:
    # PyTorch >= 1.6 supports mixed precision training
    from torch.cuda.amp import custom_fwd, custom_bwd
    class HMUniqueUpdate(autograd.Function):

        @staticmethod
        @custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            unique = set()
            for x, y in zip(inputs, indexes):
                if y.item() in unique:
                    continue
                else:
                    unique.add(y.item())
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None
except:
    class HMUniqueUpdate(autograd.Function):

        @staticmethod
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            unique = set()
            for x, y in zip(inputs, indexes):
                if y.item() in unique:
                    continue
                else:
                    unique.add(y.item())
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None


def hmuniqueupdate(inputs, indexes, features, momentum=0.5):
    return HMUniqueUpdate.apply(
        inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device)
    )


class HybridMemoryMultiFocalPercent(nn.Module):
    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2, top_percent=0.1):
        super(HybridMemoryMultiFocalPercent, self).__init__()
        print("=======using HybridMemoryMultiFocalPercent======")
        self.num_features = num_features
        self.num_memory = num_memory

        self.momentum = momentum
        self.temp = temp

        #for mutli focal
        self.top_percent = top_percent

        self.idx = torch.zeros(num_memory).long()
        self.gt_labels = torch.zeros(num_memory).long()

        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())

        self.register_buffer("split_labels", torch.zeros(num_memory).long())
    
    @torch.no_grad()
    def _init_ids(self, ids):
        self.idx.data.copy_(ids.long().to(self.labels.device))

    @torch.no_grad()
    def _init_gt_labels(self, gt_labels):
        self.gt_labels.data.copy_(gt_labels.long().to(self.gt_labels.device))

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))

    @torch.no_grad()
    def _update_label_split(self, labels):
        self.split_labels.data.copy_(labels.long().to(self.split_labels.device))
    
    @torch.no_grad()
    def get_features(self, indexes):
        return self.features[indexes].clone()
    
    @torch.no_grad()
    def get_cluster_ids(self, indexes):
        return self.labels[indexes].clone()

    def forward(self, results, indexes, aux_inputs=None, augmem_factor=0.0, split_label=False):
        inputs = results
        inputs = F.normalize(inputs, p=2, dim=1)
        if aux_inputs is not None: # 2 * torch.Size([9, 256])
            aux_inputs_new = []
            for aux_input in aux_inputs:
                aux_input = F.normalize(aux_input, p=2, dim=1)
                aux_inputs_new.append(aux_input)
            aux_inputs = aux_inputs_new

        # inputs: B*2048, features: N*2048
        if augmem_factor != 0.0:
            inputs = hm(inputs, indexes, self.features, self.momentum, aux_inputs=aux_inputs, augmem_factor=augmem_factor) #B*N, similarity
        else:
            inputs = hm(inputs, indexes, self.features, self.momentum) #B*N, similarity
        inputs /= self.temp

        if aux_inputs is not None:
            aux_inputs_list = [inputs, ]
            for aux_input in aux_inputs:
                aux_input = aux_input.mm(self.features.t().clone().detach()) / self.temp
                aux_inputs_list.append(aux_input)
            inputs = torch.stack(aux_inputs_list).mean(dim=0)

        def masked_softmax_multi_focal(vec, mask, dim=1, targets=None, epsilon=1e-6):
            exps = torch.exp(vec) # c*B
            one_hot_pos = torch.nn.functional.one_hot(targets, num_classes=exps.shape[1]) # B*C
            # assert exps.shape==one_hot_pos.shape
            one_hot_neg = one_hot_pos.new_ones(size=one_hot_pos.shape)
            one_hot_neg = one_hot_neg - one_hot_pos # B*C
            masked_exps = exps * mask.float().clone()
            neg_exps = exps.new_zeros(size=exps.shape) # B*C
            neg_exps[one_hot_neg>0] = masked_exps[one_hot_neg>0]
            ori_neg_exps = neg_exps # 负样例相似度
            neg_exps = neg_exps/neg_exps.sum(dim=1, keepdim=True) # 负样例相似度归一化
            new_exps = masked_exps.new_zeros(size=exps.shape)
            new_exps[one_hot_pos>0] = masked_exps[one_hot_pos>0] # 正样例相似度
            # topk_values, topk_indexes = neg_exps.topk(dim=1, k=self.topk)

            sorted, indices = torch.sort(neg_exps, dim=1, descending=True) # 根据负样例的相似度进行排序
            sorted_cum_sum = torch.cumsum(sorted, dim=1) # 返回维度dim中输入元素的累计和。
            sorted_cum_diff = (sorted_cum_sum - self.top_percent).abs() # 根据相似度累加的结果选择样本
            sorted_cum_min_indices = sorted_cum_diff.argmin(dim=1) # 截止到指定的某个样本
            min_values = sorted[torch.range(0, sorted.shape[0]-1).long(), sorted_cum_min_indices] # 每个样本对应的负样本
            min_values = min_values.unsqueeze(dim=-1) * ori_neg_exps.sum(dim=1, keepdim=True)
            ori_neg_sum = ori_neg_exps.sum(dim=1, keepdim=True)
            ori_neg_exps[ori_neg_exps<min_values] = 0
            # print((ori_neg_exps/ori_neg_sum).sum(dim=1)[:20])

            new_exps[one_hot_neg>0] = ori_neg_exps[one_hot_neg>0]

            masked_exps = new_exps

            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums

        B = inputs.size(0)
        if split_label:
            targets = self.split_labels[indexes].clone() # 聚类伪标签
            labels = self.split_labels.clone() # shape: N, unique label num: c
        else:
            targets = self.labels[indexes].clone() # 聚类伪标签
            labels = self.labels.clone() # shape: N, unique label num: c
        # print(labels.max())
        sim = torch.zeros(labels.max() + 1, B).float().cuda() # c*B
        sim.index_add_(0, labels, inputs.t().contiguous()) # N_all x 1  B x N_all  -> c*B 将同类样本累加
        nums = torch.zeros(labels.max() + 1, 1).float().cuda() #many instances belong to a cluster, so calculate the number of instances in a cluster
        nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda()) # u*1 每个类的数量 
        mask = (nums > 0).float() # 选取类数量大于0的
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim) #average features in each cluster, c*B
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax_multi_focal(sim.t().contiguous(), mask.t().contiguous(), targets=targets) #sim: u*B, mask:u*B, masked_sim: B*u
        return F.nll_loss(torch.log(masked_sim + 1e-6), targets)














class CrossHybridMemoryMultiFocalPercent(nn.Module):
    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2, top_percent=0.1):
        super(CrossHybridMemoryMultiFocalPercent, self).__init__()
        print("=======using HybridMemoryMultiFocalPercent======")
        self.num_features = num_features 
        self.num_memory = num_memory

        self.momentum = momentum
        self.temp = temp

        #for mutli focal
        self.top_percent = top_percent

        self.register_buffer("gcn_features", torch.zeros(num_memory, num_features))

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.gcn_features.data.copy_(features.float().to(self.gcn_features.device))


    def forward(self, results, indexes, features, labels, batchhard_posfeat=None, pos_option=0, split_label=False, temp_set=None):
        inputs = results
        inputs = F.normalize(inputs, p=2, dim=1)
        ori_inputs = inputs

        # inputs: B*2048, features: N*2048
        inputs = hm(inputs, indexes, features, self.momentum) #B*N, similarity torch.Size([10, 18048])
        
        if temp_set is None:
            inputs /= self.temp
        else:
            inputs /= temp_set

        B = inputs.size(0)

        def masked_softmax_multi_focal(vec, mask, dim=1, targets=None, epsilon=1e-6):
            exps = torch.exp(vec) # c*B torch.Size([10, 18048])
            one_hot_pos = torch.nn.functional.one_hot(targets, num_classes=exps.shape[1]) # B*C torch.Size([10, 18048])
            # assert exps.shape==one_hot_pos.shape
            one_hot_neg = one_hot_pos.new_ones(size=one_hot_pos.shape) # torch.Size([10, 18048])
            one_hot_neg = one_hot_neg - one_hot_pos # B*C
            masked_exps = exps * mask.float().clone()
            new_exps = masked_exps.new_zeros(size=exps.shape)
            

            # # 挑选最难正例参与损失计算
            # pos_exps = exps.new_ones(size=exps.shape) * 2
            # pos_exps[one_hot_pos>0] = masked_exps[one_hot_pos>0]
            # sorted, indices = torch.sort(pos_exps, dim=1) # torch.Size([20, 18048])
            # pos_exps[pos_exps>sorted[:, 0][:, None]] = 0
            # new_exps[one_hot_pos>0] = pos_exps[one_hot_pos>0] # 正样例相似度 torch.Size([21, 18048])
            # new_exps[torch.range(0, sorted.shape[0]-1).long(), targets] = masked_exps[torch.range(0, sorted.shape[0]-1).long(), targets]
            # 所有正例参与损失计算
            new_exps[one_hot_pos>0] = masked_exps[one_hot_pos>0] # 正样例相似度
            # # 挑选特定比例的正样本
            # pos_exps = exps.new_zeros(size=exps.shape) # B*C
            # pos_exps[one_hot_pos>0] = masked_exps[one_hot_pos>0]
            # ori_pos_exps = pos_exps # 负样例相似度
            # pos_exps = pos_exps/pos_exps.sum(dim=1, keepdim=True) # 负样例相似度归一化
            # sorted, indices = torch.sort(pos_exps, dim=1, descending=True) # 根据负样例的相似度进行排序
            # sorted_cum_sum = torch.cumsum(sorted, dim=1) # 返回维度dim中输入元素的累计和。
            # sorted_cum_diff = (sorted_cum_sum - self.top_percent).abs() # 根据相似度累加的结果选择样本
            # sorted_cum_min_indices = sorted_cum_diff.argmin(dim=1) # 截止到指定的某个样本
            # min_values = sorted[torch.range(0, sorted.shape[0]-1).long(), sorted_cum_min_indices] # 每个样本对应的负样本
            # min_values = min_values.unsqueeze(dim=-1) * ori_pos_exps.sum(dim=1, keepdim=True)
            # ori_pos_sum = ori_pos_exps.sum(dim=1, keepdim=True)
            # ori_pos_exps[ori_pos_exps>min_values] = 0
            # new_exps[one_hot_pos>0] = ori_pos_exps[one_hot_pos>0]


            # 初始化负例样本
            neg_exps = exps.new_zeros(size=exps.shape) # B*C
            neg_exps[one_hot_neg>0] = masked_exps[one_hot_neg>0]
            ori_neg_exps = neg_exps # 负样例相似度
            neg_exps = neg_exps/neg_exps.sum(dim=1, keepdim=True) # 负样例相似度归一化
            # 挑选符合条件的负样本
            sorted, indices = torch.sort(neg_exps, dim=1, descending=True) # 根据负样例的相似度进行排序
            sorted_cum_sum = torch.cumsum(sorted, dim=1) # 返回维度dim中输入元素的累计和。
            sorted_cum_diff = (sorted_cum_sum - self.top_percent).abs() # 根据相似度累加的结果选择样本
            sorted_cum_min_indices = sorted_cum_diff.argmin(dim=1) # 截止到指定的某个样本
            min_values = sorted[torch.range(0, sorted.shape[0]-1).long(), sorted_cum_min_indices] # 每个样本对应的负样本
            min_values = min_values.unsqueeze(dim=-1) * ori_neg_exps.sum(dim=1, keepdim=True)
            ori_neg_sum = ori_neg_exps.sum(dim=1, keepdim=True)
            ori_neg_exps[ori_neg_exps<min_values] = 0
            new_exps[one_hot_neg>0] = ori_neg_exps[one_hot_neg>0]

            masked_exps = new_exps
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums

        
        if split_label:
            labels = torch.arange(labels.size(0)).cuda()


        targets = labels[indexes].clone() # 聚类伪标签 torch.Size([10])
        labels = labels.clone() # shape: N, unique label num: c torch.Size([18048])

        # print(labels.max())
        sim = torch.zeros(labels.max() + 1, B).float().cuda() # torch.Size([8048, 10])
        sim.index_add_(0, labels, inputs.t().contiguous()) # N_all x 1  B x N_all  -> c*B 将同类样本累加
        nums = torch.zeros(labels.max() + 1, 1).float().cuda() # torch.Size([18048, 1]) many instances belong to a cluster, so calculate the number of instances in a cluster
        nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda()) # torch.Size([18048, 1]) u*1 每个类的数量 
        mask = (nums > 0).float() # torch.Size([18048, 1]) 选取类数量大于0的
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim) # torch.Size([18048, 10]) average features in each cluster, c*B
        mask = mask.expand_as(sim) # torch.Size([18048, 10])
        
        sim = sim.t().contiguous()
        if batchhard_posfeat is not None:
            batch_sim = ori_inputs.mm(batchhard_posfeat.t()) 
            
            if temp_set is None:
                batch_sim /= self.temp
            else:
                batch_sim /= temp_set

            for i, index_i in enumerate(indexes):
                # 挑选正样本
                ind_b = (targets[i] == targets)
                if pos_option == 0:
                    sim_i_pos = batch_sim[i][ind_b].max()
                elif pos_option == 1:
                    sim_i_pos = batch_sim[i][ind_b].mean()
                elif pos_option == 2:
                    sim_i_pos = batch_sim[i][ind_b].min()
                sim[i][targets[i]] = sim_i_pos
        # if pos_option > 0:
        #     sim_insmemory = inputs.contiguous()
        #     if pos_option == 1:
        #         sim_th = 0.9
        #     elif pos_option == 2:
        #         sim_th = 0.8
        #     elif pos_option == 3:
        #         sim_th = 0.7
        #     elif pos_option == 4:
        #         sim_th = 0.6
        #     for i, index_i in enumerate(indexes):
        #         ind_m = (labels[index_i] == labels) | (torch.arange(labels.size(0)).cuda() == index_i) 
        #         sim_i_pos = sim_insmemory[i][ind_m]
        #         sim_th = min(sim_th, sim_i_pos.max())
        #         sim_i_pos = sim_i_pos[sim_i_pos >= sim_th]
        #         sim_i_pos = sim_i_pos.mean()

        masked_sim = masked_softmax_multi_focal(sim, mask.t().contiguous(), targets=targets) #其实是转过来了 sim: u*B, mask:u*B, masked_sim: B*u

        return F.nll_loss(torch.log(masked_sim + 1e-6), targets)




























class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))



class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(ClusterMemory, self).__init__()
        print("=======using ClusterMemory======")
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.idx = torch.zeros(num_samples).long()
        self.gt_labels = torch.zeros(num_samples).long()

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer("labels", torch.zeros(num_samples).long())

    @torch.no_grad()
    def _init_ids(self, ids):
        self.idx.data.copy_(ids.long().to(self.labels.device))

    @torch.no_grad()
    def _init_gt_labels(self, gt_labels):
        self.gt_labels.data.copy_(gt_labels.long().to(self.gt_labels.device))

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        num_fins = features.size(0)
        self.features[:num_fins].data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))

    def forward(self, inputs, indexes):
        inputs = F.normalize(inputs, dim=1).cuda()
        targets = self.labels[indexes].clone()

        outputs = cm(inputs, targets, self.features, self.momentum)

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        return loss








