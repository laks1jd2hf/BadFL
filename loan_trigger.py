import torch
import torch.nn as nn
import main
import logging
import argparse
import copy
import random
import numpy as np
from torch.autograd import Variable

logger = logging.getLogger("logger")
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)


def cuda(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi
    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi / torch.norm(v))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        raise ValueError('Values of p different from 2 and Inf are currently not supported...')
    return v


def loan_trigger(helper, local_model, target_model, noise_trigger, intinal_trigger):
   
    logger.info("start trigger fine-tuning")
    init = False
    # load model
    model = copy.deepcopy(local_model)
    model.copy_params(target_model.state_dict())
    model.eval()
    print(noise_trigger)
    pre_trigger = torch.tensor(noise_trigger).cuda()
    aa = copy.deepcopy(intinal_trigger)
    aa = torch.tensor(aa).cuda()
    count = 0

    for e in range(1):
        corrects = 0
        datasize = 0
        for poison_id in helper.params['adversary_list']:
            count += 1
            if count > 6:
                break
            poison_data = helper.statehelper_dic[poison_id].get_poison_trainloader()
            data_iterator = poison_data
            for batch_id, (datas, labels) in enumerate(data_iterator):
                datasize += len(datas)
                x = Variable(cuda(datas, True))
                y = Variable(cuda(labels, True))
                y_target = torch.LongTensor(y.size()).fill_(int(helper.params['poison_label_swap']))
                y_target = Variable(cuda(y_target, True), requires_grad=False)
                if not init:
                    noise = copy.deepcopy(pre_trigger)
                    noise = Variable(cuda(noise, True), requires_grad=True)
                    init = True

                for ix in range(len(x)):
                    for iy in range(len(x[ix])):
                        if iy in [10, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                                  52]:
                            continue
                        else:
                            x[ix][iy] = 0

                output = model((x + noise).float())
                classloss = nn.functional.cross_entropy(output, y_target)
               
                loss = classloss
                model.zero_grad()
                if noise.grad:
                    noise.grad.fill_(0)
                loss.backward(retain_graph=True)

                noise = noise - noise.grad * 0.1
                for ix in range(len(noise)):
                    if ix in [10, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]:
                        noise[ix] = 0
                    else:
                        continue

                
                delta_noise = noise - aa
                noise = aa + proj_lp(delta_noise, 10, 2)
             

                noise = Variable(cuda(noise.data, True), requires_grad=True)
                pred = output.data.max(1)[1]
                correct = torch.eq(pred, y_target).float().mean().item()
                corrects += pred.eq(y_target.data.view_as(pred)).cpu().sum().item()
                if batch_id % 10 == 0:
                    logger.info('batchid:{},correct:{},noise:{}'.format(batch_id, correct * 100, noise.data.norm()))

        # logger.info(noise.data)
        acc = 100.0 * (float(corrects) / float(datasize))
        main.logger.info('_Train :epoch {:3d},Accuracy: {}/{} ({:.4f}%)'.format(e, corrects, datasize, acc))
        logger.info('noise{}'.format(noise))
        logger.info('noise:{}'.format(noise.data.norm()))
    return noise


if __name__ == '__main__':
    np.random.seed(1)
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()
    loan_trigger(args)
