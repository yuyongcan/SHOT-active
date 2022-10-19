import torch
import loss
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np

class SAM_first(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM_first, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self):
        # assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        # closure()
        # self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

@torch.no_grad()
def get_embedding(args, netF, netB, netC, tgt_loader, cat_data=False, aug=False):
    netF.eval()
    netB.eval()
    netC.eval()
    # out_bank = torch.zeros([len(tgt_loader.dataset), args.class_num]).cuda()
    # fea_bank = torch.zeros([len(tgt_loader.dataset), args.bottleneck]).cuda()
    out_bank = None
    fea_bank = None
    if aug:
        # aug_fea_bank = torch.zeros([len(tgt_loader.dataset), args.bottleneck]).cuda()
        aug_fea_bank = None

    for _, (input, _, idx) in enumerate(tgt_loader):
        if aug:
            data = input[0].cuda()
            aug_data = input[1].cuda()
        else:
            data = input.cuda()

        fea = netB(netF(data))
        out = netC(fea)
        # fea_bank[idx] = fea
        # out_bank[idx] = out
        if fea_bank is None:
            fea_bank = fea
            out_bank = out
            if aug:
                aug_fea = netB(netF(aug_data))
                aug_out = netC(aug_fea)
                aug_fea_bank = aug_fea
                aug_out_bank = aug_out
                # aug_fea_bank[idx] = aug_fea
        else:
            fea_bank = torch.cat((fea_bank, fea), dim=0)
            out_bank = torch.cat((out_bank, out), dim=0)
            if aug:
                aug_fea = netB(netF(aug_data))
                aug_out = netC(aug_fea)
                aug_fea_bank = torch.cat((aug_fea_bank, aug_fea), dim=0)
                aug_out_bank = torch.cat((aug_out_bank, aug_out), dim=0)

    if aug:
        return out_bank, fea_bank, aug_out_bank, aug_fea_bank

    return out_bank, fea_bank

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        # print(acc)
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    print(s)