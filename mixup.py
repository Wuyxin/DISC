import torch
import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mix_up(args, x1, x2, y1, y2, n_classes=None):
    # y1, y2 should be one-hot label, which means the shape of y1 and y2 should be [bsz, n_classes]
    length = min(len(x1), len(x2))
    x1 = x1[:length]
    x2 = x2[:length]
    y1 = y1[:length]
    y2 = y2[:length]
    if n_classes is None:
        n_classes = y1.shape[1]
    else:
        n_classes = n_classes
    bsz = len(x1)
    l = np.random.beta(args.mix_alpha, args.mix_alpha, [bsz, 1])
    if len(x1.shape) == 4:
        l_x = np.tile(l[..., None, None], (1, *x1.shape[1:]))
    else:
        l_x = np.tile(l, (1, *x1.shape[1:]))
    l_y = np.tile(l, [1, n_classes])

    mixed_x = torch.tensor(l_x, dtype=torch.float32).to(x1.device) * x1 + \
              torch.tensor(1-l_x, dtype=torch.float32).to(x2.device) * x2
    mixed_y = torch.tensor(l_y, dtype=torch.float32).to(y1.device) * y1 + \
              torch.tensor(1-l_y, dtype=torch.float32).to(y2.device) * y2
    return mixed_x, mixed_y


def cut_mix_up(args, x1, x2, y1, y2):
    length = min(len(x1), len(x2))
    x1 = x1[:length]
    x2 = x2[:length]
    y1 = y1[:length]
    y2 = y2[:length]
    input = torch.cat([x1,x2])
    target = torch.cat([y1,y2])
    rand_index = torch.cat([torch.arange(len(y2)) + len(y1), torch.arange(len(y1))])
    lam = np.random.beta(args.alpha, args.alpha)
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    return input, lam * target_a + (1-lam) * target_b


# The implementation in LISA
def mix_forward(args, group_len, x, y, g, y_onehot, model):
    if len(x) == 4:
        # LISA for CUB, CMNIST, CelebA
        if np.random.rand() < args.mix_ratio:
            mix_type = 1
        else:
            mix_type = 2
        if mix_type == 1:
            # mix different A within the same feature Y
            mix_group_1 = [x[0], x[1], y_onehot[0], y_onehot[1]]
            mix_group_2 = [x[2], x[3], y_onehot[2], y_onehot[3]]
        elif mix_type == 2:
            # mix different Y within the same feature A
            mix_group_1 = [x[0], x[2], y_onehot[0], y_onehot[2]]
            mix_group_2 = [x[1], x[3], y_onehot[1], y_onehot[3]]
        if args.cut_mix:
            mixed_x_1, mixed_y_1 = cut_mix_up(args, mix_group_1[0], mix_group_1[1], mix_group_1[2],
                                              mix_group_1[3])
            mixed_x_2, mixed_y_2 = cut_mix_up(args, mix_group_2[0], mix_group_2[1], mix_group_2[2],
                                              mix_group_2[3])
        else:
            mixed_x_1, mixed_y_1 = mix_up(args, mix_group_1[0], mix_group_1[1], mix_group_1[2],
                                          mix_group_1[3])
            mixed_x_2, mixed_y_2 = mix_up(args, mix_group_2[0], mix_group_2[1], mix_group_2[2],
                                          mix_group_2[3])
        all_mix_x = [mixed_x_1, mixed_x_2]
        all_mix_y = [mixed_y_1, mixed_y_2]
        all_group = torch.ones(
            len(mixed_x_1) + len(mixed_x_2)) * 3  # all the mixed samples are set to be from group 3
        all_y = torch.ones(len(mixed_x_1) + len(mixed_x_2)).cuda()
        all_mix_x = torch.cat(all_mix_x, dim=0)
        all_mix_y = torch.cat(all_mix_y, dim=0)

    else:
        # MetaDataset group by label, the mixup should be performed within the label group.
        all_mix_x, all_mix_y, all_group, all_y = [], [], [], []
        for i in range(group_len):
            bsz = len(x[i])
            if args.cut_mix:
                mixed_x, mixed_y = cut_mix_up(args, x[i][: bsz // 2], x[i][bsz // 2:], y_onehot[i][:bsz // 2],
                                              y_onehot[i][bsz // 2:])
                all_group.append(g[i][:len(mixed_x)])
                all_y.append(y[i][:len(mixed_x)])
                assert len(mixed_x) == len(all_y[-1])
            else:
                mixed_x, mixed_y = mix_up(args, x[i][:bsz // 2], x[i][bsz // 2:],
                                          y_onehot[i][:bsz // 2], y_onehot[i][bsz // 2:])
                all_group.append(g[i][:len(mixed_x)])
                all_y.append(y[i][:len(mixed_x)])
            all_mix_x.append(mixed_x)
            all_mix_y.append(mixed_y)
        all_mix_x = torch.cat(all_mix_x, dim=0)
        all_mix_y = torch.cat(all_mix_y, dim=0)
        all_group = torch.cat(all_group)
        all_y = torch.cat(all_y)
    outputs = model(all_mix_x.cuda())
    return outputs, all_y, all_group, all_mix_y
 