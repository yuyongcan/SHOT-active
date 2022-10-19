import torch
import numpy as np
import torch.nn as nn
from scipy.spatial.distance import cdist

def obtain_label(loader, netF, netB, netC, args, cal_dist = False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea) 
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        if cal_dist:
            return dd
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int')

def refine_label(loader, netF, netB, netC, args, source_loader=None):
    ##### Calc labeled data class center
    source_c = torch.zeros(args.class_num, args.bottleneck).cuda()
    count = torch.zeros(args.class_num).cuda()
    start_test = True
    with torch.no_grad():
        iter_test = iter(source_loader)
        for _ in range(len(source_loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()

            feas = netB(netF(inputs))
            outputs = netC(feas)

            count[labels.tolist()] += 1
            source_c[labels.tolist()] += feas

            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
        for i in range(count.size(0)):
            if count[i]:
                source_c[i] /=count[i]
        # center /= count

    ##### Calc SHOT center
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        source_c = torch.cat((source_c, torch.ones(source_c.size(0), 1).cuda()), 1)
        source_c = (source_c.t() / torch.norm(source_c, p=2, dim=1)).t().float().cpu().numpy()
        # xi = 0

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)
    # print(initc.shape, all_fea.shape)
    # exit(0)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        print(aff.shape)
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        initc = initc * args.xi + source_c * (1-args.xi)
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    print(initc.shape, source_c.shape)
    # exit(0)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int')

def NN_label(loader, netF, netB, netC, args, cal_dist = False, source_loader=None):
    source_c = torch.zeros(args.class_num, args.bottleneck).cuda()
    count = torch.zeros(args.class_num).cuda()
    start_test = True
    with torch.no_grad():
        iter_test = iter(source_loader)
        for _ in range(len(source_loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()

            feas = netB(netF(inputs))
            outputs = netC(feas)

            count[labels.tolist()] += 1
            source_c[labels.tolist()] += feas

            if start_test:
                all_src_fea = feas.float().cpu()
                all_src_output = outputs.float().cpu()
                all_src_label = labels.float()
                start_test = False
            else:
                all_src_fea = torch.cat((all_src_fea, feas.float().cpu()), 0)
                all_src_output = torch.cat((all_src_output, outputs.float().cpu()), 0)
                all_src_label = torch.cat((all_src_label, labels.float()), 0)
        for i in range(count.size(0)):
            if count[i]:
                source_c[i] /=count[i]

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)



    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        all_src_fea = torch.cat((all_src_fea, torch.ones(all_src_fea.size(0), 1)), 1)
        all_src_fea = (all_src_fea.t() / torch.norm(all_src_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    all_src_fea = all_src_fea.float().cpu().numpy()
    # dd = all_fea * all_src_fea.t()
    dd = cdist(all_fea, all_src_fea, args.distance)
    # K = all_output.size(1)
    # aff = all_output.float().cpu().numpy()
    # initc = aff.transpose().dot(all_fea)
    # initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    # cls_count = np.eye(K)[predict].sum(axis=0)
    # labelset = np.where(cls_count>args.threshold)
    # labelset = labelset[0]
    # print(labelset)

    # dd = cdist(all_fea, initc[labelset], args.distance)
    
    pred_label = dd.argmin(axis=1)
    pred_label = all_src_label.cpu().numpy()[pred_label]
    # pred_label = labelset[pred_label]

    # for round in range(1):
    #     aff = np.eye(K)[pred_label]
    #     initc = aff.transpose().dot(all_fea) 
    #     initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    #     dd = cdist(all_fea, initc[labelset], args.distance)
    #     if cal_dist:
    #         return dd
    #     pred_label = dd.argmin(axis=1)
    #     pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int')

def Threshold(source_loader, netF, netB, netC, args):
    # source_sfm = torch.zeros(args.class_num, args.bottleneck).cuda()
    # count = torch.zeros(args.class_num).cuda()
    max_prob = 0.
    start_test = True
    with torch.no_grad():
        iter_test = iter(source_loader)
        for _ in range(len(source_loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()

            feas = netB(netF(inputs))
            outputs = netC(feas)

            sfm = nn.Softmax(dim=1)(outputs)
            max_, _ = sfm.max(dim=1)
            # print(max_, len(source_loader))
            max_prob += torch.sum(max_) / inputs.size(0)
    theta = max_prob / len(source_loader)
    # print(theta)
    # exit(0)

    return theta