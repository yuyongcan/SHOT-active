import math
import random
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from torch.nn import Softmax,CrossEntropyLoss
import loss
import network
from data_list import data_load, data_load_Q
from pl import obtain_label
from utils import get_embedding, SAM_first
from image_target import New_model

def resume_ckpt(args, last=False):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    label_statistic = []
    out_statistic = []
    max_prob_statistic = []
    # KLD_statistic = []

    path = './DA/' + args.dset + '/' + args.name
    for i in range(15):
        pathF = path + "/target_F_" + str(i) + args.savename + ".pt"
        pathB = path + "/target_B_" + str(i) + args.savename + ".pt"
        pathC = path + "/target_C_" + str(i) + args.savename + ".pt"
        netF.load_state_dict(torch.load(pathF))
        netB.load_state_dict(torch.load(pathB))
        netC.load_state_dict(torch.load(pathC))
        netB.eval()
        netF.eval()
        netC.eval()

        curr_out, curr_pred = obtain_current_pred(dset_loaders['test'], netF, netB, netC, args)  ## out after softmax
        label_statistic.append(curr_pred)
        out_statistic.append(curr_out)
        max_prob, _ = torch.max(curr_out, dim=1)
        max_prob_statistic.append(max_prob)

    counting = torch.zeros(curr_pred.size(0)).cuda()
    kl_sum = torch.zeros(curr_pred.size(0)).cuda()
    out_mean = torch.zeros(curr_out.size()).cuda()
    prob_mean = torch.zeros(max_prob.size()).cuda()
    for i in range(0, len(label_statistic)):
        if i > 0:
            for idx in range(counting.size(0)):
                if label_statistic[i][idx] != label_statistic[i - 1][idx]:
                    counting[idx] += 1.

            kl_D = KLD(out_statistic[i], out_statistic[i - 1])
            kl_sum += kl_D
        out_mean += out_statistic[i] / len(out_statistic)
        prob_mean += max_prob_statistic[i] / len(max_prob_statistic)
    # print(counting)
    # srt_idx = counting.argsort(descending=True)
    # srt = counting.sort(descending=True)
    if last:
        return counting, out_statistic[-1], max_prob_statistic[-1], _

    return counting, out_mean, prob_mean, kl_sum


def obtain_current_pred(loader, netF, netB, netC, args):
    start_test = True

    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_output = outputs.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    # ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    _, predict = torch.max(all_output, 1)

    return all_output, predict


def get_wrong_list(tgt_loader, netF, netB, netC, total_list, labeled_list, labeled_ind, unlb_list, unlb_idx):
    netF.eval()
    netB.eval()
    netC.eval()
    for _, (input, label, idx) in enumerate(tgt_loader):

        data = input.cuda()
        # target = label.cuda()

        fea = netB(netF(data))
        out = netC(fea)
        sfm_out = nn.Softmax(dim=1)(out)
        _, pred = torch.max(sfm_out, 1)

        new_lb_list, new_lb_ind = labeled_list, labeled_ind
        new_unlb_list, new_unlb_ind = [], []
        for i in range(pred.size(0)):
            if pred[i] != label[i]:
                id = idx[i]
                new_lb_list.append(unlb_list[id])
                new_lb_ind.append(unlb_idx[id])

        for id in range(len(total_list)):
            if str(id) not in new_lb_ind:
                new_unlb_list.append(total_list[id])
                new_unlb_ind.append(id)
    return new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind


def obtain_list(sort_idx, labeled_list, labeled_ind, unlb_list, unlb_idx, total_list, n):
    new_lb_list, new_lb_ind = labeled_list, labeled_ind
    new_unlb_list, new_unlb_ind = [], []

    for i in range(n):
        id = sort_idx[i]
        new_lb_list.append(unlb_list[id])
        new_lb_ind.append(unlb_idx[id])

    for id in range(len(total_list)):
        if id not in new_lb_ind:
            new_unlb_list.append(total_list[id])
            new_unlb_ind.append(id)
    return new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind


@torch.no_grad()
def KM_Query(args, netF, netB, netC, total_list, labeled_list, labeled_ind, unlb_list, unlb_idx, n):
    netF.eval()
    netB.eval()
    netC.eval()
    query_loader = data_load_Q(args, unlb_list)['Q']
    out_bank, emb_bank = get_embedding(args, netF, netB, netC, query_loader)
    # sfm_bank = nn.Softmax(dim=1)(out_bank)
    # score_bank = sfm_bank ** 2 / sfm_bank.sum(dim=0)
    emb_bank = emb_bank.cpu().numpy()

    # weights = -(score_bank*torch.log(score_bank)).sum(1).cpu().numpy()
    # weights = sfm_bank ** 2 / sfm_bank.sum(0)

    new_lb_list, new_lb_ind = labeled_list, labeled_ind

    ## weighted Kmeans
    km = KMeans(n)
    km.fit(emb_bank)

    ## Find nearest neighbors to inference the cluster centroids
    dists = euclidean_distances(km.cluster_centers_, emb_bank)

    sort_idxs = dists.argsort(axis=1)  # range from 0 to len(total)

    for i in range(n):
        ## check if the repeated index in the list
        s_idx = 0
        idx = sort_idxs[i, s_idx]
        total_idx = unlb_idx[idx]

        while total_idx in new_lb_ind:
            s_idx += 1
            idx = sort_idxs[i, s_idx]
            total_idx = unlb_idx[idx]

        assert total_idx not in new_lb_ind

        ## add the quried index in the labeled list and labeled index
        new_lb_ind.append(total_idx)
        new_lb_list.append(total_list[int(total_idx)])

    ## update unlabeled list
    new_unlb_list, new_unlb_ind = [], []
    for id in range(len(total_list)):
        if id not in new_lb_ind:
            new_unlb_list.append(total_list[id])
            new_unlb_ind.append(id)

    return new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind


@torch.no_grad()
def clue_Query(args, netF, netB, netC, total_list, labeled_list, labeled_ind, unlb_list, unlb_idx, n):
    netF.eval()
    netB.eval()
    netC.eval()
    query_loader = data_load_Q(args, unlb_list)['Q']
    out_bank, emb_bank = get_embedding(args, netF, netB, netC, query_loader)
    emb_bank = emb_bank.cpu().numpy()
    score_bank = nn.Softmax(dim=1)(out_bank / args.T) + 1e-8
    weights = -(score_bank * torch.log(score_bank)).sum(1).cpu().numpy()

    new_lb_list, new_lb_ind = labeled_list, labeled_ind

    ## weighted Kmeans
    km = KMeans(n)
    km.fit(emb_bank, sample_weight=weights)

    ## Find nearest neighbors to inference the cluster centroids
    dists = euclidean_distances(km.cluster_centers_, emb_bank)

    sort_idxs = dists.argsort(axis=1)  # range from 0 to len(total)

    for i in range(n):
        ## check if the repeated index in the list
        s_idx = 0
        idx = sort_idxs[i, s_idx]
        total_idx = unlb_idx[idx]

        while total_idx in new_lb_ind:
            s_idx += 1
            idx = sort_idxs[i, s_idx]
            total_idx = unlb_idx[idx]

        assert total_idx not in new_lb_ind

        ## add the quried index in the labeled list and labeled index
        new_lb_ind.append(total_idx)
        new_lb_list.append(total_list[int(total_idx)])

    ## update unlabeled list
    new_unlb_list, new_unlb_ind = [], []
    for id in range(len(total_list)):
        if id not in new_lb_ind:
            new_unlb_list.append(total_list[id])
            new_unlb_ind.append(id)

    return new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind


@torch.no_grad()
def aug_clue_Query(args, netF, netB, netC, total_list, labeled_list, labeled_ind, unlb_list, unlb_idx, n):
    netF.eval()
    netB.eval()
    netC.eval()
    query_loader = data_load_Q(args, unlb_list, aug=True)['Q']
    out_bank, emb_bank, aug_out_bank, aug_emb_bank = get_embedding(args, netF, netB, netC, query_loader, aug=True)
    emb_bank = emb_bank.cpu().numpy()
    score_bank = nn.Softmax(dim=1)(out_bank / args.T) + 1e-8

    # weights = -(score_banweightsk*torch.log(score_bank)).sum(1).cpu().numpy() # entropy

    weights = loss.margin(score_bank)  # .cpu().numpy()
    weights = (weights.max() / weights).cpu().numpy()

    new_lb_list, new_lb_ind = labeled_list, labeled_ind

    ## weighted Kmeans
    km = KMeans(n)
    # km.fit(high_ent_emb, sample_weight=weights)
    km.fit(emb_bank, sample_weight=weights)

    ## Find nearest neighbors to inference the cluster centroids
    dists = euclidean_distances(km.cluster_centers_, emb_bank)

    sort_idxs = dists.argsort(axis=1)  # range from 0 to len(total)

    for i in range(n):
        ## check if the repeated index in the list
        s_idx = 0
        idx = sort_idxs[i, s_idx]
        total_idx = unlb_idx[idx]

        while total_idx in new_lb_ind:
            s_idx += 1
            idx = sort_idxs[i, s_idx]
            total_idx = unlb_idx[idx]

        assert total_idx not in new_lb_ind

        ## add the quried index in the labeled list and labeled index
        new_lb_ind.append(total_idx)
        new_lb_list.append(total_list[int(total_idx)])

    ## update unlabeled list
    new_unlb_list, new_unlb_ind = [], []
    for id in range(len(total_list)):
        if id not in new_lb_ind:
            new_unlb_list.append(total_list[id])
            new_unlb_ind.append(id)

    return new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind


"""@torch.no_grad()
def KM_Query(args, netF, netB, netC, total_list, labeled_list, labeled_ind, unlb_list, unlb_idx, n):
    netF.eval()
    netB.eval()
    netC.eval()
    query_loader = data_load_Q(args, unlb_list)['Q']
    _, emb_bank = get_embedding(args, netF, netB, netC, query_loader)
    emb_bank = emb_bank.cpu().numpy()
    # score_bank = nn.Softmax(dim=1)(out_bank / args.T) + 1e-8
    # weights = -(score_bank*torch.log(score_bank)).sum(1).cpu().numpy()

    new_lb_list, new_lb_ind = labeled_list, labeled_ind

    ## weighted Kmeans
    km = KMeans(n)
    km.fit(emb_bank)

    ## Find nearest neighbors to inference the cluster centroids
    dists = euclidean_distances(km.cluster_centers_, emb_bank)

    sort_idxs = dists.argsort(axis=1)  # range from 0 to len(total)

    for i in range(n):
        ## check if the repeated index in the list
        s_idx = 0
        idx = sort_idxs[i,s_idx] 
        total_idx = unlb_idx[idx]

        while total_idx in new_lb_ind:
            s_idx += 1
            idx = sort_idxs[i,s_idx]
            total_idx = unlb_idx[idx]

        assert total_idx not in new_lb_ind

        ## add the quried index in the labeled list and labeled index
        new_lb_ind.append(total_idx)
        new_lb_list.append(total_list[int(total_idx)])

    ## update unlabeled list
    new_unlb_list, new_unlb_ind = [], []
    for id in range(len(total_list)):
        if str(id) not in new_lb_ind:
            new_unlb_list.append(total_list[id])
            new_unlb_ind.append(id)

    return new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind
"""

@torch.no_grad()
def Rand_Query(args, netF, netB, netC, total_list, labeled_list, labeled_ind, unlb_list, unlb_idx, n):
    netF.eval()
    netB.eval()
    netC.eval()
    rand_list = random.sample(range(len(unlb_list)), n)
    new_lb_list, new_lb_ind = labeled_list, labeled_ind
    new_unlb_list, new_unlb_ind = [], []

    for id in rand_list:
        new_lb_list.append(unlb_list[id])
        new_lb_ind.append(unlb_idx[id])
    for i in range(len(total_list)):
        if not i in new_lb_ind:
            new_unlb_list.append(total_list[i])
            new_unlb_ind.append(i)
    return new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind


@torch.no_grad()
def Entropy_Query(args, netF, netB, netC, total_list, labeled_list, labeled_ind, unlb_list, unlb_idx, n):
    netF.eval()
    netB.eval()
    netC.eval()
    tgt_loader = data_load_Q(args, unlb_list)['Q']
    out_bank, _ = get_embedding(args, netF, netB, netC, tgt_loader)
    out_bank = nn.Softmax(dim=1)(out_bank / args.T) + 1e-8
    H = loss.Entropy(out_bank)
    sort_idx = H.argsort(axis=0, descending=True)

    new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind = obtain_list(sort_idx, labeled_list, labeled_ind, unlb_list,
                                                                       unlb_idx, total_list, n)
    return new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind


@torch.no_grad()
def margin_Query(args, netF, netB, netC, total_list, labeled_list, labeled_ind, unlb_list, unlb_idx, n):
    netF.eval()
    netB.eval()
    netC.eval()
    new_lb_list, new_lb_ind = labeled_list, labeled_ind
    tgt_loader = data_load_Q(args, unlb_list)['Q']
    out_bank, _ = get_embedding(args, netF, netB, netC, tgt_loader)

    score_bank = nn.Softmax(dim=1)(out_bank)
    margin_bank = loss.margin(score_bank)

    sort_idx = margin_bank.argsort(axis=0)

    new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind = obtain_list(sort_idx, labeled_list, labeled_ind, unlb_list,
                                                                       unlb_idx, total_list, n)
    return new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind


@torch.no_grad()
def Max_Query(args, netF, netB, netC, total_list, labeled_list, labeled_ind, unlb_list, unlb_idx, n):
    netF.eval()
    netB.eval()
    netC.eval()
    tgt_loader = data_load_Q(args, unlb_list)['Q']
    out_bank, _ = get_embedding(args, netF, netB, netC, tgt_loader)

    score_bank = nn.Softmax(dim=1)(out_bank)
    max_bank, _ = torch.max(score_bank, dim=1)

    sort_idx = max_bank.argsort(axis=0)
    print(sort_idx)

    new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind = obtain_list(sort_idx, labeled_list, labeled_ind, unlb_list,
                                                                       unlb_idx, total_list, n)
    return new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind

@torch.no_grad()
def CET_Query(args, netF, netB, netC, total_list, labeled_list, labeled_ind, unlb_list, unlb_idx, n):
    netF.eval()
    netB.eval()
    netC.eval()
    # sfmax = nn.Softmax(dim=1)
    CEL = CrossEntropyLoss(reduce=False)

    # with torch.no_grad():
    #     tgt_loader = data_load_Q(args, unlb_list)['Q']
    #     loss_total = None
    #     for _, (input, _, idx) in enumerate(tgt_loader):
    #         input = input.cuda()
    #         out = netC(netB(netF(input)))
    #         # out=sfmax(out)
    #         _, pseudo_labels = torch.max(out, dim=1)
    #         loss_batch = CEL(out, pseudo_labels)
    #         if loss_total is None:
    #             loss_total = loss_batch
    #         else:
    #             loss_total = torch.cat((loss_total, loss_batch),dim=0)
    # sort_idx = loss_total.argsort(axis=0, descending=True)

    tgt_loader = data_load_Q(args, unlb_list)['Q']
    out_bank, _ = get_embedding(args, netF, netB, netC, tgt_loader)
    _,pseudo_labels = torch.max(out_bank,dim=1)
    loss_total= CEL(out_bank,pseudo_labels)
    sort_idx = loss_total.argsort(axis=0, descending=True)

    print(sort_idx)

    new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind = obtain_list(sort_idx, labeled_list, labeled_ind, unlb_list,
                                                                       unlb_idx, total_list, n)

    return new_lb_list, new_lb_ind, new_unlb_list, new_unlb_ind

def SAM_Query(args, netF, netB, netC, total_list, labeled_list, labeled_ind, unlb_list, unlb_idx, n):
    netF_CP, netB_CP, netC_CP = New_model(args)
    netF_CP.load_state_dict(netF.state_dict())
    netB_CP.load_state_dict(netB.state_dict())
    netC_CP.load_state_dict(netC.state_dict())
    netB_CP=netB_CP.cuda()
    netF_CP=netF_CP.cuda()
    netC_CP=netC_CP.cuda()

    param_group = []
    learning_rate = args.pro
    netF_CP.train()
    netB_CP.train()
    netC_CP.train()
    for k, v in netF_CP.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB_CP.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    # sfmax=nn.Softmax(dim=1)
    optimizer = SAM_first(param_group,optim.SGD,args.pro)
    optimizer.zero_grad()
    tgt_loader = data_load_Q(args, unlb_list)['Q']
    CEL=CrossEntropyLoss()
    for _, (input, _, idx) in enumerate(tgt_loader):
        input = input.cuda()
        out = netC_CP(netB_CP(netF_CP(input)))
        # out = sfmax(out)
        _,pseudo_labels=torch.max(out,dim=1)
        loss=CEL(out,pseudo_labels)
        loss.backward(retain_graph=True)
    optimizer.step()

    return CET_Query(args, netF_CP, netB_CP, netC_CP, total_list, labeled_list, labeled_ind, unlb_list, unlb_idx, n)

