from pytorch_lightning import seed_everything
seed_everything(2020)
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import loss
import network
import query
from data_list import make_dataset, data_load, data_load_active
from get_args import get_target_args
from loss import CrossEntropyLabelSmooth
from pl import obtain_label
from utils import op_copy, lr_scheduler, cal_acc, print_args


def KLD(sfm, sft):
    return -torch.mean(torch.sum(sfm.log() * sft, dim=1))


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            # if args.refine:
            #     mem_label = refine_label(dset_loaders['test'], netF, netB, netC, args)
            # else:
            mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter,
                                                                            acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()

    # if args.issave:   
    #     torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
    #     torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
    #     torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netB,


def per_run_train(args, netF, netB, netC, l_list, u_list, n):
    dset_loaders = data_load_active(args, l_list, u_list)

    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False
    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            v.requires_grad = True
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            v.requires_grad = True
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.per_run_epoch * len(dset_loaders["u"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        # t_now = time.time()
        # print('1:',t_now-t)

        try:
            inputs_test_all, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["u"])
            inputs_test_all, _, tar_idx = iter_test.next()

        # t_now = time.time()
        # print('1-1:',t_now-t)

        try:
            inputs_src_all, label_src, _ = iter_label.next()
        except:
            iter_label = iter(dset_loaders["l"])
            inputs_src_all, label_src, _ = iter_label.next()
        inputs_src = inputs_src_all[0].cuda()
        inputs_src_aug = inputs_src_all[1].cuda()
        label_src = label_src.cuda()

        inputs_test = inputs_test_all[0].cuda()
        inputs_test_aug = inputs_test_all[1].cuda()
        if inputs_test.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        features_test_aug = netB(netF(inputs_test_aug))
        outputs_test_aug = netC(features_test_aug)

        outputs_test_all = torch.cat((outputs_test, outputs_test_aug), dim=0)

        outputs_source = netC(netB(netF(inputs_src)))
        outputs_source_aug = netC(netB(netF(inputs_src_aug)))
        outputs_source_all = torch.cat((outputs_source, outputs_source_aug), dim=0)

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, label_src)
        classifier_loss += nn.CrossEntropyLoss()(outputs_source_aug, label_src)

        if args.ent_par:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        if args.mix_ratio:
            rho = np.random.beta(0.75, 0.75)

            lt_all = torch.cat((inputs_src, inputs_src_aug), dim=0)
            ut_all = torch.cat((inputs_test, inputs_test_aug), dim=0)

            mix_img = ut_all * rho + lt_all * (1 - rho)
            mix_target = outputs_test_all * rho + outputs_source_all * (1 - rho)

            # mix_img = inputs_src * rho + inputs_test * (1-rho)
            # mix_target = outputs_source * rho + outputs_test_aug * (1 - rho)

            mix_out = netC(netB(netF(mix_img)))

            classifier_loss += KLD(nn.Softmax(dim=1)(mix_out), nn.Softmax(dim=1)(mix_target)) * args.mix_ratio

        if args.th_ratio:  # and iter_num > interval_iter :
            sfm_lt = nn.Softmax(dim=1)(outputs_source)
            mean_prob = torch.max(sfm_lt, dim=1)[0].mean()
            if args.tau:
                mean_prob *= args.tau
            prob_ut, pred = torch.max(nn.Softmax(dim=1)(outputs_test), 1)

            index = None
            for i in range(prob_ut.size(0)):
                if prob_ut[i] >= mean_prob:
                    if index is None:
                        index = torch.tensor(i).cuda().unsqueeze(0)
                    else:
                        index = torch.cat((index, torch.tensor(i).cuda().unsqueeze(0)), dim=0)

            out_th = outputs_test_aug[index]
            pl_th = pred[index]
            th_loss = nn.CrossEntropyLoss()(out_th, pl_th.detach())

            pi = torch.tensor(np.pi, dtype=torch.float).cuda()
            mu = 0.5 - torch.cos(torch.minimum(pi, (2 * pi * iter_num) / max_iter)) / 2

            classifier_loss += th_loss * args.th_ratio * mu

        if args.w_th_ratio:
            sfm_lt = nn.Softmax(dim=1)(outputs_source)
            mean_prob = torch.max(sfm_lt, dim=1)[0].mean()
            if args.tau:
                mean_prob *= args.tau
            sfm_ut = nn.Softmax(dim=1)(outputs_test)
            prob_ut, pred = torch.max(sfm_ut, 1)

            with torch.no_grad():
                index = None
                for i in range(prob_ut.size(0)):
                    if prob_ut[i] >= mean_prob:
                        if index is None:
                            index = torch.tensor(i).cuda().unsqueeze(0)
                        else:
                            index = torch.cat((index, torch.tensor(i).cuda().unsqueeze(0)), dim=0)
                sfm_pl = sfm_ut[index]
                sfm_cls = sfm_lt[index]
                # pl_th = pred[index]
                sfm_pl = sfm_pl * sfm_cls.mean(0) / sfm_pl.mean(0)
                # sfm_pl = sfm_pl ** 2
                sfm_pl = sfm_pl / sfm_pl.sum(dim=1, keepdim=True)
                _, pl_th = torch.max(sfm_pl, dim=1)

            out_th = outputs_test_aug[index]
            pl_th = pred[index]
            th_loss = nn.CrossEntropyLoss()(out_th, pl_th.detach())

            pi = torch.tensor(np.pi, dtype=torch.float).cuda()
            mu = 0.5 - torch.cos(torch.minimum(pi, (2 * pi * iter_num) / max_iter)) / 2

            classifier_loss += th_loss * args.th_ratio * mu

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 1 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Round:{}; Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, n, iter_num, max_iter,
                                                                                      acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Round:{}; Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, n, iter_num, max_iter,
                                                                                      acc_s_te)
                # acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                # log_str = 'Task: {}, Round:{}; Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, n, iter_num, max_iter, acc_s_te) + '\n' + acc_list

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()
            # t_now = time.time()
            # print('4-1:',t_now-t)
        # if iter_num == 5:
        #     exit(0)
    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, str(n) + "_target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, str(n) + "_target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, str(n) + "_target_C_" + args.savename + ".pt"))

    return netF, netB, netC


def New_model(args):
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    # if args.resume:
    #     path = osp.join('san', args.da, args.dset, 'stoc' ,names[args.s][0] + names[args.t][0].upper())
    if not args.resume:
        model_dir = args.output_dir_src
        modelpath = model_dir + '/source_F.pt'
        netF.load_state_dict(torch.load(modelpath))
        modelpath = model_dir + '/source_B.pt'
        netB.load_state_dict(torch.load(modelpath))
        modelpath = model_dir + '/source_C.pt'
        netC.load_state_dict(torch.load(modelpath))

    else:
        model_dir = args.output_dir_tgt
        modelpath = model_dir + '/target_F_par_0.3.pt'
        netF.load_state_dict(torch.load(modelpath))
        modelpath = model_dir + '/target_B_par_0.3.pt'
        netB.load_state_dict(torch.load(modelpath))
        modelpath = model_dir + '/target_C_par_0.3.pt'
        netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    log_str = 'Load model from:' + model_dir + '\n'
    print(log_str)
    for k, v in netC.named_parameters():
        v.requires_grad = False
    return netF, netB, netC


def Tune_C(args, netF, netB, netC, l_list, u_list, n):
    dset_loaders = data_load_active(args, l_list, u_list)
    netF.eval()
    netB.eval()
    # t = time.time()
    # print(t)

    param_group = []
    for k, v in netF.named_parameters():
        v.requires_grad = False
    for k, v in netB.named_parameters():
        v.requires_grad = False
    for k, v in netC.named_parameters():
        v.requires_grad = True
        param_group += [{'params': v, 'lr': args.lr * args.lr_decay2 * 5}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.per_run_epoch * len(dset_loaders["l"])
    interval_iter = max_iter // 5
    iter_num = 0

    netC.train()

    while iter_num < max_iter:

        try:
            inputs_src, label_src, _ = iter_label.next()
        except:
            iter_label = iter(dset_loaders["l"])
            inputs_src, label_src, _ = iter_label.next()
        inputs_src = inputs_src.cuda()
        label_src = label_src.cuda()

        # t_now = time.time()
        # print('2:',t_now-t)

        if inputs_src.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        outputs_source = netC(netB(netF(inputs_src)))

        classifier_loss = torch.tensor(0.0).cuda()

        src_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, label_src)
        classifier_loss += args.src_par * src_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netC.eval()
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Round:{}; Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, n, iter_num, max_iter,
                                                                                      acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Round:{}; Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, n, iter_num, max_iter,
                                                                                      acc_s_te)
                # acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                # log_str = 'Task: {}, Round:{}; Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, n, iter_num, max_iter, acc_s_te) + '\n' + acc_list

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            # netF.train()
            # netB.train()
            netC.train()

    return netF, netB, netC


def main_active(args):
    ## set base network
    netF, netB, netC = New_model(args)
    total_list = make_dataset(args.t_dset_path)
    l_list, l_idx, u_list = [], [], total_list
    u_idx = [i for i in range(len(u_list))]

    if args.budgets > 1:
        budgets = int(args.budgets)
    else:
        budgets = len(total_list) * args.budgets
    B = int(budgets / args.runs)

    if args.query == "clue":
        Query_func = query.clue_Query
    elif args.query == 'aug_clue':
        Query_func = query.aug_clue_Query
    elif args.query == 'shot':
        Query_func = query.shot_Query
    elif args.query == 'rand':
        Query_func = query.Rand_Query
    elif args.query == 'ent':
        Query_func = query.Entropy_Query
    elif args.query == 'mg':
        Query_func = query.margin_Query
    elif args.query == 'gt':
        Query_func = query.Error_Query
    elif args.query == 'pl':
        Query_func = query.PL_Query
    elif args.query == 'max':
        Query_func = query.Max_Query
    elif args.query == 'km':
        Query_func = query.KM_Query
    elif args.query == 'fw':
        Query_func = query.fw_Query
    elif args.query == 'sam':
        Query_func = query.SAM_Query
    elif args.query == 'cet':
        Query_func = query.CET_Query
    else:
        raise RuntimeError("Query Type Error.")

    for run in range(args.runs):
        print(len(total_list))
        l_list, l_idx, u_list, u_idx = Query_func(args, netF, netB, netC, total_list, l_list, l_idx, u_list, u_idx, B)
        # exit()
        print(len(l_list), len(u_list))

        query_list = l_list[0:]
        sta = [0] * args.class_num
        for i in range(len(query_list)):
            cls = query_list[i][1]
            sta[cls] += 1
        print(sta)
        print(sta.index(max(sta)), sta.index(min(sta)), np.var(sta))
        log_str = str(sta.index(max(sta))) + ' ' + str(sta.index(min(sta))) + ' ' + str(np.var(sta)) + '\n'
        print('Number of queried samples:', len(l_list))
        log_str += 'Number of queried samples:' + str(len(l_list))

        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        # netF, netB, netC = New_model(args)

        # if args.test2:
        #     netF, netB, netC = per_mixmatch(args, netF, netB, netC, l_list, u_list, n=run + 1)
        # else:
        netF, netB, netC = per_run_train(args, netF, netB, netC, l_list, u_list, n=run + 1)

    return


if __name__ == "__main__":
    args = get_target_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


    l = len(names)
    if args.t:
        l = 1

    for i in range(l):
        if l != 1:
            if i == args.s:
                continue
            args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir_tgt = osp.join('ckps/target', args.da, args.dset,
                                       names[args.s][0].upper() + names[args.t][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, args.query+'_'+args.disc,
                                   names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        # args.out_file.write(print_args(args)+'\n')
        print_args(args)
        args.out_file.flush()


        main_active(args)
