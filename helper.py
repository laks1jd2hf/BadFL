import heapq
from shutil import copyfile
from collections import defaultdict
import math
import torch
from torch.autograd import Variable
import logging
import sklearn.metrics.pairwise as smp
from torch.nn.functional import log_softmax
import torch.nn.functional as F
import torch.nn as nn
import time
import functools
import hdbscan
import os
import json
import numpy as np
import config
import copy
import main
import utils.csv_record
import random
from collections import Counter

logger = logging.getLogger("logger")


class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None

        self.params = params
        self.name = name
        self.best_loss = math.inf
        aa = self.params['aggregation_methods']
        bb = self.params['attack_methods']
        alpha = self.params['alpha_loss']
        beta = self.params['beta_loss']
        gamma = self.params['gamma_loss']
        geadmask = self.params['gradmask_ratio']
        self.folder_path = f'saved_models/model_{self.name}_{current_time}_{bb}_{aa}_{alpha}_{beta}_{gamma}_{geadmask}'
        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f'current path: {self.folder_path}')
        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path
        self.fg = FoolsGold(use_memory=self.params['fg_use_memory'])

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def model_global_norm(model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_dist_norm(model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_max_values(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer.data - target_params[name].data)))
        return squared_sum

    @staticmethod
    def model_max_values_var(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer - target_params[name])))
        return sum(squared_sum)

    @staticmethod
    def get_one_vec(model, variable=False):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        if variable:
            sum_var = Variable(torch.cuda.FloatTensor(size).fill_(0))
        else:
            sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            if variable:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer).view(-1)
            else:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var

    @staticmethod
    def model_dist_norm_var(model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        sum_var = sum_var.to(config.device)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
                    layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def cos_sim_loss(self, model, target_vec):
        model_vec = self.get_one_vec(model, variable=True)
        target_var = Variable(target_vec, requires_grad=False)
        # target_vec.requires_grad = False
        cs_sim = torch.nn.functional.cosine_similarity(
            self.params['scale_weights'] * (model_vec - target_var) + target_var, target_var, dim=0)
        # cs_sim = cs_loss(model_vec, target_vec)
        logger.info("los")
        logger.info(cs_sim.data[0])
        logger.info(torch.norm(model_vec - target_var).data[0])
        loss = 1 - cs_sim

        return 1e3 * loss

    def model_cosine_similarity(self, model, target_params_variables,
                                model_id='attacker'):

        cs_list = list()
        cs_loss = torch.nn.CosineSimilarity(dim=0)
        for name, data in model.named_parameters():
            if name == 'decoder.weight':
                continue

            # model_update = 100 * (data.view(-1) - target_params_variables[name].view(-1)) + target_params_variables[
            #     name].view(-1)    yuanlai
            model_update = (data.view(-1) - target_params_variables[name].view(-1)) + target_params_variables[
                name].view(-1)

            cs = F.cosine_similarity(model_update,
                                     target_params_variables[name].view(-1), dim=0)
            cs_list.append(cs)

        cos_los_submit = 1 - (sum(cs_list) / len(cs_list))

        return sum(cs_list) / len(cs_list)
        # return (1-cos_los_submit)   yuanlai
        # return 1e3 * cos_los_submit

    def accum_similarity(self, last_acc, new_acc):

        cs_list = list()

        cs_loss = torch.nn.CosineSimilarity(dim=0)

        for name, layer in last_acc.items():
            cs = cs_loss(Variable(last_acc[name], requires_grad=False).view(-1),
                         Variable(new_acc[name], requires_grad=False).view(-1))

            cs_list.append(cs)
        cos_los_submit = 1 * (1 - sum(cs_list) / len(cs_list))

        return sum(cos_los_submit)

    @staticmethod
    def clip_weight_norm(model, clip):
        total_norm = Helper.model_global_norm(model)
        logger.info("total_norm: " + str(total_norm) + "clip_norm: " + str(clip))
        max_norm = clip
        clip_coef = max_norm / (total_norm + 1e-6)
        current_norm = total_norm
        if total_norm > max_norm:
            for name, layer in model.named_parameters():
                layer.data.mul_(clip_coef)
            current_norm = Helper.model_global_norm(model)
            logger.info("clip~~~ norm after clipping: " + str(current_norm))
        return current_norm

    @staticmethod
    def dp_noise(param, sigma):

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    def accumulate_weight(self, weight_accumulator, epochs_submit_update_dict, state_keys, num_samples_dict):
        """
         return Args:
             updates: dict of (num_samples, update), where num_samples is the
                 number of training samples corresponding to the update, and update
                 is a list of variable weights
            state_key:用户数
         """
        if self.params['aggregation_methods'] == config.AGGR_FOOLSGOLD or self.params[
            'aggregation_methods'] == config.AGGR_KRUM \
                or self.params['aggregation_methods'] == config.AGGR_TRIMMED_MEAN or self.params[
            'aggregation_methods'] == config.AGGR_BULYAN or \
                self.params['aggregation_methods'] == config.AGGR_DNC or self.params[
            'aggregation_methods'] == config.AGGR_MEDIAN or \
                self.params['aggregation_methods'] == config.AGGR_MKRUM:
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_gradients = epochs_submit_update_dict[state_keys[i]][0]  # agg 1 interval
                num_samples = num_samples_dict[state_keys[i]]
                updates[state_keys[i]] = (num_samples, copy.deepcopy(local_model_gradients))
            return None, updates

        else:  # for flame 存的是参数
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_update_list = epochs_submit_update_dict[state_keys[i]]
                update = dict()
                num_samples = num_samples_dict[state_keys[i]]
                for name, data in local_model_update_list[0].items():
                    update[name] = torch.zeros_like(data)

                for j in range(0, len(local_model_update_list)):
                    local_model_update_dict = local_model_update_list[j]
                    for name, data in local_model_update_dict.items():
                        weight_accumulator[name].add_(local_model_update_dict[name])
                        update[name].add_(local_model_update_dict[name])
                        detached_data = data.cpu().detach().numpy()
                        # print(detached_data.shape)
                        detached_data = detached_data.tolist()
                        # print(detached_data)
                        local_model_update_dict[name] = detached_data  # from gpu to cpu

                updates[state_keys[i]] = (num_samples, update)

            return weight_accumulator, updates

    def init_weight_accumulator(self, target_model):
        weight_accumulator = dict()
        for name, data in target_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)

        return weight_accumulator

    def CRFL(self, weight_accumulator, target_model, epoch):
        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue

            update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["no_models"])
            # update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["number_of_total_participants"])
            newvalue = data + update_per_layer
            target_model.state_dict()[name].copy_(newvalue)

            # clip the global model
            if self.params['type'] == config.TYPE_MNIST:
                dynamic_thres = epoch * 0.5 + 2
            elif self.params['type'] == config.TYPE_LOAN:
                dynamic_thres = epoch * 0.8 + 2
            elif self.params['type'] == config.TYPE_DDOS:
                dynamic_thres = epoch * 0.025 + 2
            elif self.params['type'] == config.TYPE_CIFAR:
                dynamic_thres = epoch * 0.5 + 2
            elif self.params['type'] == config.TYPE_CIFAR100:
                dynamic_thres = epoch * 0.8 + 2
            elif self.params['type'] == config.TYPE_FMNIST:
                dynamic_thres = epoch * 0.5 + 2
            param_clip_thres = self.params["param_clip_thres"]
            if dynamic_thres < param_clip_thres:
                param_clip_thres = dynamic_thres

        current_norm = Helper.clip_weight_norm(target_model, param_clip_thres)

        # csv_record.add_norm_result(current_norm)
        # add noise
        logger.info(f" epoch: {epoch} add noise on the global model!")
        for name, param in target_model.state_dict().items():
            newvalue = param + Helper.dp_noise(param, self.params['sigma_param']).to(config.device)
            target_model.state_dict()[name].copy_(newvalue)
        return True

    def fedavglr(self, weight_accumulator, target_model, epocht, epoch_interval):
        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue

            update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["no_models"]) * (
                    1 / epocht + 1)
            # update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["number_of_total_participants"])
            # update_per_layer = update_per_layer * 1.0 / epoch_interval

            newvalue = data + update_per_layer
            target_model.state_dict()[name].copy_(newvalue)

        return True

    def average_shrink_models(self, weight_accumulator, target_model, epoch_interval):

        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue
            update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["no_models"])
            # update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["number_of_total_participants"])
            # update_per_layer = update_per_layer * 1.0 / epoch_interval
            # if self.params['diff_privacy']:
            #     update_per_layer.add_(self.dp_noise(data, self.params['sigma']))

            newvalue = data + update_per_layer
            target_model.state_dict()[name].copy_(newvalue)

        return True

    @staticmethod
    def row_into_parameters(row, target_model):
        offset = 0
        for param in target_model.parameters():
            new_size = functools.reduce(lambda x, y: x * y, param.shape)
            current_data = row[offset:offset + new_size]
            param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
            offset += new_size

    # compute L2-norm
    @staticmethod
    def krum_create_distances(users_grads):
        distances = defaultdict(dict)
        for i in range(len(users_grads)):
            for j in range(i):
                distances[i][j] = distances[j][i] = np.linalg.norm(users_grads[i] - users_grads[j])
        return distances

    @staticmethod
    # get the updated weights from users
    def collect_gradients(updates, users_grads):
        users_gradsss = []
        for name, data in updates.items():
            users_gradsss.append(data[1])  # gradient

        for i in range(len(users_gradsss)):
            grads = np.concatenate([users_gradsss[i][j].cpu().numpy().flatten() for j in range(len(users_gradsss[i]))])
            users_grads[i, :] = grads
        return users_grads

    @staticmethod
    def collect_weights(updates, users_params):
        users_gradsss = []
        for name, data in updates.items():
            users_gradsss.append(data[1])  # gradient

        for i in range(len(users_gradsss)):
            grads = np.concatenate([users_gradsss[i][name].cpu().numpy().flatten() for name in users_gradsss[i]])
            users_params[i, :] = grads
        return users_params

    def krum(self, target_model, updates, users_count, corrupted_count, agent_name_keys, distances=None,
             return_index=False, debug=False):
        current_weights = np.concatenate([i.data.cpu().numpy().flatten() for i in target_model.parameters()])
        users_grads = np.empty((users_count, len(current_weights)), dtype=current_weights.dtype)
        users_grads = Helper.collect_gradients(updates, users_grads)
        velocity = np.zeros(current_weights.shape, users_grads.dtype)

        # if not return_index:
        #     assert users_count >= 2 * corrupted_count + 1, (
        #         'users_count>=2*corrupted_count + 3', users_count, corrupted_count)
        if corrupted_count > 5:
            corrupted_count = 5  # 保证只有5个用户
        logger.info(corrupted_count)
        non_count = users_count - corrupted_count
        minimal_error = 1e20
        minimal_error_index = -1
        if distances is None:
            distances = Helper.krum_create_distances(users_grads)
        for user in distances.keys():
            errors = sorted(distances[user].values())
            current_error = sum(errors[:non_count])

            if current_error < minimal_error:
                minimal_error = current_error
                minimal_error_index = user
        main.logger.info(minimal_error_index)
        current_grads = users_grads[minimal_error_index]

        # velocity = self.params['momentum'] * velocity - self.params['lr'] * current_grads
        # current_weights += velocity
        current_weights = current_weights - self.params['lr'] * current_grads
        Helper.row_into_parameters(current_weights, target_model)
        return True

    def mkrum(self, target_model, updates, corrupted_count, users_count, distances=None, return_index=False):
        current_weights = np.concatenate([i.data.cpu().numpy().flatten() for i in target_model.parameters()])
        users_grads = np.empty((users_count, len(current_weights)), dtype=current_weights.dtype)
        users_grads = Helper.collect_gradients(updates, users_grads)
        velocity = np.zeros(current_weights.shape, users_grads.dtype)

        if corrupted_count > 5:
            corrupted_count = 5
        logger.info(corrupted_count)
        # if not return_index:
        #     assert users_count >= 2 * corrupted_count + 1, (
        #         'users_count>=2*corrupted_count + 3', users_count, corrupted_count)
        non_malicious_count = users_count - corrupted_count
        minimal_error = 1e20
        minimal_error_index = -1
        selection_set = []
        krumscore = []
        if distances is None:
            distances = Helper.krum_create_distances(users_grads)
        for user in distances.keys():
            errors = sorted(distances[user].values())
            current_error = sum(errors[:non_malicious_count])
            krumscore.append(current_error)
        aa = krumscore[0]
        krumscore[0] = krumscore[1]
        krumscore[1] = aa
        result = map(krumscore.index, heapq.nsmallest(non_malicious_count, krumscore))
        for i in result:
            selection_set.append(users_grads[i])

        current_grads = np.empty((users_grads.shape[1],), users_grads.dtype)
        users_gradsss = np.array(selection_set)
        for i, param_across_users in enumerate(users_gradsss.T):
            current_grads[i] = np.mean(param_across_users)

        # velocity = self.params['momentum'] * velocity - self.params['lr'] * current_grads
        # current_weights += velocity
        current_weights = current_weights - self.params['lr'] * current_grads
        Helper.row_into_parameters(current_weights, target_model)
        return True

    def trimmed_mean(self, target_model, updates, users_count, corrupted_count):
        current_weights = np.concatenate([i.data.cpu().numpy().flatten() for i in target_model.parameters()])
        users_grads = np.empty((users_count, len(current_weights)), dtype=current_weights.dtype)
        users_grads = Helper.collect_gradients(updates, users_grads)
        velocity = np.zeros(current_weights.shape, users_grads.dtype)

        number_to_consider = int(users_grads.shape[0] - 2) - 1
        current_grads = np.empty((users_grads.shape[1],), users_grads.dtype)
        print("grad收集完毕")
        for i, param_across_users in enumerate(users_grads.T):
            med = np.median(param_across_users)
            good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[:number_to_consider]
            current_grads[i] = np.mean(good_vals) + med

        print("current_grad计算完毕")
        # velocity = self.params['momentum'] * velocity - self.params['lr'] * current_grads
        # current_weights += velocity
        current_weights = current_weights - self.params['lr'] * current_grads
        print("current_weight计算完毕")
        Helper.row_into_parameters(current_weights, target_model)
        print("转化成model")
        return True

    def computekrum(self, users_grads, users_count, corrupted_count, distances=None, return_index=False, debug=False):
        # if not return_index:
        #     assert users_count >= 2 * corrupted_count + 1, (
        #         'users_count>=2*corrupted_count + 3', users_count, corrupted_count)
        non_malicious_count = users_count - corrupted_count
        minimal_error = 1e20
        minimal_error_index = -1

        if distances is None:
            distances = self.krum_create_distances(users_grads)
        for user in distances.keys():
            errors = sorted(distances[user].values())
            current_error = sum(errors[:non_malicious_count])
            if current_error < minimal_error:
                minimal_error = current_error
                minimal_error_index = user

        if return_index:
            return minimal_error_index
        else:
            return users_grads[minimal_error_index]

    def computetrimmed(self, users_grads, users_count, corrupted_count):
        number_to_consider = int(users_grads.shape[0] - corrupted_count) - 1
        current_grads = np.empty((users_grads.shape[1],), users_grads.dtype)

        for i, param_across_users in enumerate(users_grads.T):
            med = np.median(param_across_users)
            good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[:number_to_consider]
            current_grads[i] = np.mean(good_vals) + med
        return current_grads

    def bulyan(self, target_model, updates, users_count, corrupted_count):
        current_weights = np.concatenate([i.data.cpu().numpy().flatten() for i in target_model.parameters()])
        users_grads = np.empty((users_count, len(current_weights)), dtype=current_weights.dtype)
        users_grads = Helper.collect_gradients(updates, users_grads)
        velocity = np.zeros(current_weights.shape, users_grads.dtype)
        print("grad收集完毕")

        if corrupted_count > 1:  # 保证每轮不超过一个恶意用户
            corrupted_count = 1
        logger.info(corrupted_count)
        # assert users_count >= 4 * corrupted_count + 3
        set_size = users_count - 2 * corrupted_count
        selection_set = []

        distances = Helper.krum_create_distances(users_grads)
        while len(selection_set) < set_size:
            currently_selected = self.computekrum(users_grads, users_count - len(selection_set), corrupted_count,
                                                  distances, True)
            selection_set.append(users_grads[currently_selected])
            # remove the selected from next iterations:
            distances.pop(currently_selected)
            for remaining_user in distances.keys():
                distances[remaining_user].pop(currently_selected)

        current_grads = self.computetrimmed(np.array(selection_set), len(selection_set), 2)
        print("current_grad计算完毕")
        # velocity = self.params['momentum'] * velocity - self.params['lr'] * current_grads
        # current_weights += velocity
        current_weights = current_weights - self.params['lr'] * current_grads
        print("current_weight计算完毕")
        Helper.row_into_parameters(current_weights, target_model)
        print("转化成model")
        return True

    def median(self, target_model, users_count, updates):

        current_weights = np.concatenate([i.data.cpu().numpy().flatten() for i in target_model.parameters()])
        users_grads = np.empty((users_count, len(current_weights)), dtype=current_weights.dtype)
        users_grads = Helper.collect_gradients(updates, users_grads)
        velocity = np.zeros(current_weights.shape, users_grads.dtype)

        number_to_consider = int(users_grads.shape[0] - 2) - 1
        current_grads = np.empty((users_grads.shape[1],), users_grads.dtype)
        print("grad收集完毕")
        for i, param_across_users in enumerate(users_grads.T):
            med = np.median(param_across_users)
            current_grads[i] = med

        print("current_grad计算完毕")
        # velocity = self.params['momentum'] * velocity - self.params['lr'] * current_grads
        # current_weights += velocity
        current_weights = current_weights - self.params['lr'] * current_grads
        print("current_weight计算完毕")
        Helper.row_into_parameters(current_weights, target_model)
        print("转化成model")
        return True

    def foolsgold_update(self, target_model, updates):
        client_grads = []
        alphas = []
        names = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            names.append(name)

        adver_ratio = 0
        for i in range(0, len(names)):
            _name = names[i]
            if _name in self.params['adversary_list']:
                adver_ratio += alphas[i]
        adver_ratio = adver_ratio / sum(alphas)
        poison_fraction = adver_ratio * self.params['poisoning_per_batch'] / self.params['batch_size']
        logger.info(f'[foolsgold agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[foolsgold agg] considering poison per batch poison_fraction: {poison_fraction}')

        target_model.train()
        # train and update
        optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
                                    momentum=self.params['momentum'],
                                    weight_decay=self.params['decay'])

        optimizer.zero_grad()
        agg_grads, wv, alpha = self.fg.aggregate_gradients(client_grads, names)
        for i, (name, params) in enumerate(target_model.named_parameters()):
            agg_grads[i] = agg_grads[i] * self.params["eta"]
            if params.requires_grad:
                params.grad = agg_grads[i].to(config.device)
        optimizer.step()
        wv = wv.tolist()
        utils.csv_record.add_weight_result(names, wv, alpha)
        return True, names, wv, alpha

    def fltrust(self, target_model, updates, server_update, agent_name_keys):
        user_cs = dict()
        user_nor = dict()
        update = dict()
        # 加载用户模型参数
        for idx, data in updates.items():
            update[idx] = data[1]

        for i in range(0, len(agent_name_keys)):
            cs_list = list()
            for name in server_update:
                cs = F.cosine_similarity(update[agent_name_keys[i]][name].view(-1),
                                         server_update[name].view(-1).float(), dim=0)
                cs_list.append(cs)
            # cos_los_submit = 1 - (sum(cs_list) / len(cs_list))
            cos_los_submit = sum(cs_list) / len(cs_list)
            m = nn.ReLU(inplace=True)
            cos = torch.tensor(cos_los_submit).clone().detach()
            user_cs[i] = m(cos)
        logger.info(user_cs)
        logger.info("相似度计算完毕，并relu")
        serverl2 = 0
        for name in server_update:
            serverl2 += torch.sum(torch.pow(server_update[name].data, 2))
        serverl2 = math.sqrt(serverl2)
        logger.info(serverl2)

        for i in range(0, len(agent_name_keys)):
            squared_sum = 0
            for name in update[agent_name_keys[i]]:
                squared_sum += torch.sum(torch.pow(update[agent_name_keys[i]][name].data, 2))
                # user_nor[i] += torch.sum(torch.pow(update[agent_name_keys[i]][name].data, 2))
            squared_sum = math.sqrt(squared_sum)
            user_nor[i] = serverl2 / squared_sum
            print('用户l2:', squared_sum)
            print('用户系数:', user_nor[i])
            for name in update[agent_name_keys[i]]:
                update[agent_name_keys[i]][name] = update[agent_name_keys[i]][name] * user_nor[i]

        logger.info("标准化结束")
        sum_cs = 0
        for i in user_cs:
            sum_cs += user_cs[i]
        logger.info(sum_cs)
        for name, data in target_model.state_dict().items():
            namelayer = 0
            for i in range(0, len(agent_name_keys)):
                namelayer += user_cs[i] * update[agent_name_keys[i]][name]
            newvalue = data + namelayer / sum_cs
            target_model.state_dict()[name].copy_(newvalue)
        logger.info('聚合完毕')

        return True

    @staticmethod
    def mediumnum(num):
        listnum = [num[i] for i in range(len(num))]
        listnum.sort()
        lnum = len(num)
        if lnum % 2 == 1:
            i = int((lnum + 1) / 2) - 1
            return listnum[i]
        else:
            i = int(lnum / 2) - 1
            return (listnum[i] + listnum[i + 1]) / 2

    def flame(self, target_model, updates, agent_name_keys):
        update = dict()
        # 加载用户模型参数
        for idx, data in updates.items():
            update[idx] = data[1]  # 各个用户的本地模型参数

        # 保存所有client 参数到np里
        # current_weights = np.concatenate([i.data.cpu().numpy().flatten() for i in target_model.parameters()])
        # users_params = np.empty((len(agent_name_keys), len(current_weights)), dtype=current_weights.dtype)
        # users_params = Helper.collect_weights(updates, users_params)
        # print(users_params.shape)

        # 计算模型之间的余弦相似度
        logger.info('计算余弦相似度')
        cos_bb = defaultdict(dict)
        for i in range(0, len(agent_name_keys)):
            for j in range(i + 1):
                cs_list = list()
                for name in update[agent_name_keys[i]]:
                    cs = F.cosine_similarity(update[agent_name_keys[i]][name].view(-1),
                                             update[agent_name_keys[j]][name].view(-1).float(), dim=0)
                    cs_list.append(cs)

                cos_submit = 1 - (sum(cs_list) / len(cs_list))
                print(f"({i},{j}):{cos_submit}")
                cos_bb[i][j] = cos_bb[j][i] = cos_submit.item()

        cos_distance = []
        for i in range(0, len(agent_name_keys)):
            aa = []
            for j in range(0, len(agent_name_keys)):
                aa.append(cos_bb[i][j])
            cos_distance.append(aa)
        logger.info(cos_distance)

        # cos_aa = defaultdict(dict)
        # for i in range(0, len(users_params)):
        #     for j in range(i + 1):
        #         t1 = np.array(users_params[i])
        #         t2 = np.array(users_params[j])
        #         t1_norm = np.linalg.norm(t1)
        #         t2_norm = np.linalg.norm(t2)
        #         cos = np.dot(t1, t2) / (t1_norm * t2_norm)
        #         print(f"({i},{j}):{cos}")
        #         cos_aa[i][j] = cos_aa[j][i] = 1 - min(1, cos)
        # cos_distance = []
        # for i in range(0, len(agent_name_keys)):
        #     aa = []
        #     for j in range(0, len(agent_name_keys)):
        #         aa.append(cos_aa[i][j])
        #     cos_distance.append(aa)
        # print(cos_distance)

        # 聚类
        logger.info('余弦相似度计算完毕，开始聚类！')
        clusterer = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=6, allow_single_cluster=True)
        clusterer.fit(cos_distance)
        logger.info(clusterer.labels_)
        max_index = clusterer.labels_.max()  # 样本最多的类的标签
        logger.info(f'样本最多的类别id：{max_index}')

        if max_index==-1:  #全部是离群点，跳过本轮聚合
            return True   #skip

        # 保存聚出的正常用户
        benign_id = []
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_index:
                benign_id.append(i)
        logger.info(benign_id)

        logger.info('聚类结束，计算欧式距离！')
        # 计算所有用户的 S_t
        last_global_model = dict()
        for name, data in target_model.state_dict().items():
            last_global_model[name] = target_model.state_dict()[name].clone()  # 获取上一轮全局模型参数

        distance_list = []
        for i in range(0, len(agent_name_keys)):
            squared_sum = 0
            for name in update[agent_name_keys[i]]:
                squared_sum += torch.sum(
                    torch.pow(update[agent_name_keys[i]][name].data - last_global_model[name].data, 2))
            squared_sum = math.sqrt(squared_sum)
            distance_list.append(squared_sum)

        S_t = self.mediumnum(distance_list)
        logger.info(S_t)

        logger.info('裁剪本地模型')
        for i in benign_id:  # 聚出的正常用户
            for name in update[agent_name_keys[i]]:
                update[agent_name_keys[i]][name] = min(1, S_t / distance_list[i]) * (
                        update[agent_name_keys[i]][name].data - last_global_model[name].data) + last_global_model[
                                                       name].data

        logger.info('聚合模型')
        for name, param in target_model.state_dict().items():
            newvalue = torch.zeros_like(param)
            for i in benign_id:
                newvalue = newvalue + update[agent_name_keys[i]][name].data
            newvalue = newvalue / len(benign_id)  # 取平均
            target_model.state_dict()[name].copy_(newvalue)

        logger.info('加噪声')
        sigma_a = S_t * (1 / self.params['flame_eps']) * math.sqrt(2 * math.log(1.25 / self.params['flame_dalta']))
        logger.info(sigma_a)
        for name, param in target_model.state_dict().items():
            newvalue = param + Helper.dp_noise(param, sigma_a).to(config.device)
            target_model.state_dict()[name].copy_(newvalue)

        logger.info('聚合完毕')

        return True

    def geometric_median_update(self, target_model, updates, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6,
                                max_update_norm=None):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
               """
        points = []
        alphas = []
        names = []
        for name, data in updates.items():
            points.append(data[1])  # update
            alphas.append(data[0])  # num_samples
            names.append(name)

        adver_ratio = 0
        for i in range(0, len(names)):
            _name = names[i]
            if _name in self.params['adversary_list']:
                adver_ratio += alphas[i]
        adver_ratio = adver_ratio / sum(alphas)
        poison_fraction = adver_ratio * self.params['poisoning_per_batch'] / self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        alphas = np.asarray(alphas, dtype=np.float64) / sum(alphas)
        alphas = torch.from_numpy(alphas).float()

        # alphas.float().to(config.device)
        median = Helper.weighted_average_oracle(points, alphas)
        num_oracle_calls = 1

        # logging
        obj_val = Helper.geometric_median_objective(median, points, alphas)
        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append(log_entry)
        if verbose:
            logger.info('Starting Weiszfeld algorithm')
            logger.info(log_entry)
        logger.info(f'[rfa agg] init. name: {names}, weight: {alphas}')
        # start
        wv = None
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = torch.tensor([alpha / max(eps, Helper.l2dist(median, p)) for alpha, p in zip(alphas, points)],
                                   dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = Helper.weighted_average_oracle(points, weights)
            num_oracle_calls += 1
            obj_val = Helper.geometric_median_objective(median, points, alphas)
            log_entry = [i + 1, obj_val,
                         (prev_obj_val - obj_val) / obj_val,
                         Helper.l2dist(median, prev_median)]
            logs.append(log_entry)
            if verbose:
                logger.info(log_entry)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                wv = copy.deepcopy(weights)
                break
            logger.info(
                f'[rfa agg] iter:  {i}, prev_obj_val: {prev_obj_val}, obj_val: {obj_val}, abs dis: {abs(prev_obj_val - obj_val)}')
            logger.info(f'[rfa agg] iter:  {i}, weight: {weights}')
            wv = copy.deepcopy(weights)
        alphas = [Helper.l2dist(median, p) for p in points]

        update_norm = 0
        for name, data in median.items():
            update_norm += torch.sum(torch.pow(data, 2))
        update_norm = math.sqrt(update_norm)

        if max_update_norm is None or update_norm < max_update_norm:
            for name, data in target_model.state_dict().items():
                update_per_layer = median[name] * (self.params["eta"])
                if self.params['diff_privacy']:
                    update_per_layer.add_(self.dp_noise(data, self.params['sigma']))
                data.add_(update_per_layer)
            is_updated = True
        else:
            logger.info('\t\t\tUpdate norm = {} is too large. Update rejected'.format(update_norm))
            is_updated = False

        # utils.csv_record.add_weight_result(names, wv.cpu().numpy().tolist(), alphas)

        return num_oracle_calls, is_updated, names, wv.cpu().numpy().tolist(), alphas

    @staticmethod
    def Privacy_account(num_items_train, threshold_epochs, nummodel, numuser, sigma, privacys):
        clipthr = 0.1  # 1 cifar 0.1
        #
        q_s = nummodel / numuser
        delta_s = clipthr / num_items_train
        noise_scale = delta_s * np.sqrt(2 * q_s * threshold_epochs * np.log(1 / sigma)) / privacys  # 修改
        return noise_scale

    @staticmethod
    def get_1_norm(params_a):
        sum = 0
        if isinstance(params_a, np.ndarray) == True:
            sum += pow(np.linalg.norm(params_a, ord=2), 2)
        else:
            for i in params_a.keys():
                if 'num_batches_tracked' in i:
                    continue
                if len(params_a[i]) == 1:
                    sum += pow(np.linalg.norm(params_a[i].cpu().numpy(), ord=2), 2)
                else:
                    a = copy.deepcopy(params_a[i].cpu().numpy())
                    for j in a:
                        x = copy.deepcopy(j.flatten())
                        sum += pow(np.linalg.norm(x, ord=2), 2)
        norm = np.sqrt(sum)
        return norm

    @staticmethod
    def clipping(w, clipthr=20):
        norms = Helper.get_1_norm(w)
        if norms > clipthr:
            w_local = copy.deepcopy(w)
            for i in w.keys():
                w_local[i] = copy.deepcopy(w[i] * clipthr / Helper.get_1_norm(w))
        else:
            w_local = copy.deepcopy(w)
        return w_local

    @staticmethod
    def noise_add(agent_name_keys, noise_scale, w):
        w_noise = copy.deepcopy(w)
        for k in range(len(agent_name_keys)):
            for name in w[agent_name_keys[k]].keys():
                noise = np.random.normal(0, noise_scale, w[agent_name_keys[k]][name].size())
                noise = torch.from_numpy(noise).float().to(config.device)
                w_noise[agent_name_keys[k]][name] = w_noise[agent_name_keys[k]][name] + noise
        return w_noise

    def fedLDP(self, target_model, updates, agent_name_keys):
        update = dict()
        # 加载用户模型参数
        for name, data in updates.items():
            update[name] = data[1]

        # if self.params['type'] == config.TYPE_CIFAR:
        #     epocha = self.params['epochs'] / 100
        # if self.params['type'] == config.TYPE_CIFAR100:
        #     epocha = self.params['epochs'] / 10
        # else:
        #     epocha = self.params['epochs']

        noise_scale = copy.deepcopy(
            Helper.Privacy_account(self.params['num_items_train'], self.params['epochs'], self.params['no_models'],
                                   self.params['number_of_total_participants'], self.params['sigma'],
                                   self.params['privacy_budget']))
        print('噪声方差计算完毕', noise_scale)

        for idx in range(len(agent_name_keys)):
            update[agent_name_keys[idx]] = copy.deepcopy(Helper.clipping(update[agent_name_keys[idx]]))
        print('clip用户参数完毕')

        update = Helper.noise_add(agent_name_keys=agent_name_keys, noise_scale=noise_scale, w=update)
        print('用户参数加噪完毕')

        for name, data in target_model.state_dict().items():
            w_avg = copy.deepcopy(update[agent_name_keys[0]][name])
            for i in range(1, len(agent_name_keys)):
                w_avg += update[agent_name_keys[i]][name]

            update_per_layer = w_avg * (self.params["eta"] / self.params["no_models"])
            newvalue = data + update_per_layer

            if 'num_batches_tracked' in name:
                continue
            target_model.state_dict()[name].copy_(newvalue)

        return True

    def fedCDP(self, target_model, epoch, agent_name_keys, updates):
        update = dict()
        # 加载用户模型参数
        for name, data in updates.items():
            update[name] = data[1]

        for name in update[agent_name_keys[0]]:
            print(name)

        updateweight = [np.expand_dims(update[agent_name_keys[0]][name].cpu(), -1) for name in
                        update[agent_name_keys[0]]]
        num_weights = len(updateweight)
        print(num_weights)
        for i in range(1, len(agent_name_keys)):
            updateweight = [
                np.concatenate((updateweight[idx], np.expand_dims(update[agent_name_keys[i]][name].cpu(), -1)), -1)
                for idx, name in enumerate(update[agent_name_keys[i]])]
        # 求norm，
        Norm = [np.sqrt(np.sum(
            np.square(updateweight[name]), axis=tuple(range(updateweight[name].ndim)[:-1]), keepdims=True)) for name
            in range(num_weights)]
        print('求norm完毕')

        # 求median
        median = [np.median(Norm[name], axis=-1, keepdims=True) for name
                  in range(num_weights)]
        print('求median完毕')
        print(median)
        # clip update
        factor = [Norm[i] / median[i] for i in range(num_weights)]
        for i in range(num_weights):
            factor[i][factor[i] > 1.0] = 1.0

        ClippedUpdates = [updateweight[i] / factor[i] for i in range(num_weights)]
        print('clip模型完毕')
        mm = float(ClippedUpdates[0].shape[-1])
        # 求均值，在参数均值上加上noise作为新的global model
        MeanClippedUpdates = [np.mean(ClippedUpdates[i], -1) for i in range(num_weights)]
        GaussianNoise = [(1.0 / mm * np.random.normal(loc=0.0, scale=float(self.params['sigmas'] * median[i]),
                                                      size=MeanClippedUpdates[i].shape)) for i in range(num_weights)]
        Sanitized_Updates = [MeanClippedUpdates[i] + GaussianNoise[i] for i in range(num_weights)]
        print('模型梯度计算完毕')
        logger.info(f" epoch: {epoch} add noise on the global model!")
        for idx, name in enumerate(target_model.state_dict()):
            newvalue = target_model.state_dict()[name].cpu() + Sanitized_Updates[idx]
            target_model.state_dict()[name].copy_(newvalue)
        return True

    def DnC(self, target_model, updates, users_count, m):
        # print(updates)
        # 确定形状
        current_weights = np.concatenate([i.data.cpu().numpy().flatten() for i in target_model.parameters()])
        users_grads = np.empty((users_count, len(current_weights)), dtype=current_weights.dtype)
        # 获取 user_grads 梯度
        users_grads = Helper.collect_gradients(updates, users_grads)
        # 单个梯度/参数的形状
        velocity = np.zeros(current_weights.shape, users_grads.dtype)
        # ddos:(10,28580)
        print("grad收集完毕")
        every_user_grads_list = []

        for i, param_across_users in enumerate(users_grads):
            # ddos (28580,)
            # print("聚合时，单个用户梯度的形状：", param_across_users.shape)
            every_user_grads_list.append(param_across_users)

        delta = np.array(every_user_grads_list)
        # c: 过滤分数
        c = self.params['filter_parameters']
        # m: 恶意客户端数量
        # m = len(self.params['adversary_list'])
        # b: 计算均值的人数
        b = self.params['compute_average_person_number']
        # d: 参与梯度聚合的人数 （for ddos 10人）
        d = len(delta)
        # 循环次数
        niter = self.params['niter']
        # 计算出来的梯度
        current_grads, _ = self.computeDnC(delta, c, m, b, d, niter)

        print("current_grad计算完毕")
        # velocity = self.params['momentum'] * velocity - self.params['lr'] * current_grads
        # current_weights += velocity
        current_weights = current_weights - self.params['lr'] * current_grads
        print("current_weight计算完毕")
        Helper.row_into_parameters(current_weights, target_model)
        print("转化成model")
        return True

    def computeDnC(self, delta, c, m, b, d, niter):
        """
            :param delta: 需要进行DeC的矩阵，梯度矩阵  type → ndarray
            :param c: (超参)过滤分数
            :param m: (超参)恶意客户端数
            :param b: (超参)子样本的维度，计算均值的人数
            :param d: 输入梯度集合的维度，即梯度的个数,即参与聚合的人数，总人数
            :param niter: (超参)循环次数
            :return:监测过后的梯度矩阵（聚合后的矩阵集合） type → ndarray
        """

        def get_index_of_min(nums, n):
            # 获取最小的前n个的index
            index = []
            for i in range(n):
                min_value = np.min(nums)
                index_tmp = nums.index(min_value)
                nums[index_tmp] = 9
                index.append(index_tmp)
            print("差异score:", nums)
            print("选出前 ", n, " 小的序号:", index)
            return index

        def autoNorm(data):  # 传入一个列表
            mins = np.min(np.array(data))  # 返回data矩阵中每一列中最小的元素，返回一个列表
            maxs = np.max(np.array(data))  # 返回data矩阵中每一列中最大的元素，返回一个列表
            ranges = maxs - mins  # 差值区间
            normData = (np.array(data) - mins) / ranges
            normData = list(normData)
            return normData

        delta_i = []
        i = 0
        L_good = []
        while i < niter:
            i += 1
            # 随机 r
            r = random.sample(range(0, d), b)
            r.sort()
            for j in range(len(r)):
                delta_i.append(delta[r[j]])
            # 对 delta_i 中的 b个n维向量求平均
            mu = np.average(delta_i, axis=0)
            delta_c = delta_i - mu
            # 运行 cifar\mnist 数据集的时候，内存溢出 报错
            # mnist: (5, 431080)
            print(delta_c.shape)
            _, _, vt = np.linalg.svd(delta_c, full_matrices=False)
            # v 为 vt（右奇异矩阵） 的 top
            v = vt[0]
            # s 为 outlier score
            s = []
            delta_tmp = delta - mu
            for k in range(d):
                # 每个选出的梯度与奇异矩阵的top特征向量进行计算向量积，获得 outlier score
                s.append(np.dot(delta_tmp[k], v))
            # 归一化
            s = autoNorm(s)
            L = get_index_of_min(s, d - c * m)
            L_good.append(L)
        # L_final的类型是set
        L_final = set(L_good[0]).intersection(*L_good[1:])
        final_list = []
        for l in range(len(L_final)):
            final_list.append(delta[list(L_final)[l]])
        print("进行平均的梯度列表：", final_list)
        delta_a = np.average(final_list, axis=0)
        return delta_a, L_final

    @staticmethod
    def l2dist(p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        squared_sum = 0
        for name, data in p1.items():
            squared_sum += torch.sum(torch.pow(p1[name] - p2[name], 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def geometric_median_objective(median, points, alphas):
        """Compute geometric median objective."""
        temp_sum = 0
        for alpha, p in zip(alphas, points):
            temp_sum += alpha * Helper.l2dist(median, p)
        return temp_sum

        # return sum([alpha * Helper.l2dist(median, p) for alpha, p in zip(alphas, points)])

    @staticmethod
    def weighted_average_oracle(points, weights):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = torch.sum(weights)

        weighted_updates = dict()

        for name, data in points[0].items():
            weighted_updates[name] = torch.zeros_like(data)
        for w, p in zip(weights, points):  # 对每一个agent
            for name, data in weighted_updates.items():
                temp = (w / tot_weights).float().to(config.device)
                temp = temp * (p[name].float())
                # temp = w / tot_weights * p[name]
                if temp.dtype != data.dtype:
                    temp = temp.type_as(data)
                data.add_(temp)

        return weighted_updates

    def save_model(self, model=None, epoch=0, val_loss=0):
        if model is None:
            model = self.target_model
        if self.params['save_model']:
            # save_model
            logger.info("saving model")
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params['save_on_epochs']:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    def update_epoch_submit_dict(self, epochs_submit_update_dict, global_epochs_submit_dict, epoch, state_keys):

        epoch_len = len(epochs_submit_update_dict[state_keys[0]])
        for j in range(0, epoch_len):
            per_epoch_dict = dict()
            for i in range(0, len(state_keys)):
                local_model_update_list = epochs_submit_update_dict[state_keys[i]]
                local_model_update_dict = local_model_update_list[j]
                per_epoch_dict[state_keys[i]] = local_model_update_dict

            global_epochs_submit_dict[epoch + j] = per_epoch_dict

        return global_epochs_submit_dict

    def save_epoch_submit_dict(self, global_epochs_submit_dict):
        with open(f'{self.folder_path}/epoch_submit_update.json', 'w') as outfile:
            json.dump(global_epochs_submit_dict, outfile, ensure_ascii=False, indent=1)

    def estimate_fisher(self, model, criterion,
                        data_loader, sample_size, batch_size=64):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        if self.params['type'] == 'text':
            data_iterator = range(0, data_loader.size(0) - 1, self.params['bptt'])
            hidden = model.init_hidden(self.params['batch_size'])
        else:
            data_iterator = data_loader

        for batch_id, batch in enumerate(data_iterator):
            data, targets = self.get_batch(data_loader, batch,
                                           evaluation=False)
            if self.params['type'] == 'text':
                hidden = self.repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                loss = criterion(output.view(-1, self.n_tokens), targets)
            else:
                output = model(data)
                loss = log_softmax(output, dim=1)[range(targets.shape[0]), targets.data]
                # loss = criterion(output.view(-1, ntokens
            # output, hidden = model(data, hidden)
            loglikelihoods.append(loss)
            # loglikelihoods.append(
            #     log_softmax(output.view(-1, self.n_tokens))[range(self.params['batch_size']), targets.data]
            # )

            # if len(loglikelihoods) >= sample_size // batch_size:
            #     break
        logger.info(loglikelihoods[0].shape)
        # estimate the fisher information of the parameters.
        loglikelihood = torch.cat(loglikelihoods).mean(0)
        logger.info(loglikelihood.shape)
        loglikelihood_grads = torch.autograd.grad(loglikelihood, model.parameters())

        parameter_names = [
            n.replace('.', '__') for n, p in model.named_parameters()
        ]
        return {n: g ** 2 for n, g in zip(parameter_names, loglikelihood_grads)}

    def consolidate(self, model, fisher):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            model.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            model.register_buffer('{}_estimated_fisher'
                                  .format(n), fisher[n].data.clone())

    def ewc_loss(self, model, lamda, cuda=False):
        try:
            losses = []
            for n, p in model.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(model, '{}_estimated_mean'.format(n))
                fisher = getattr(model, '{}_estimated_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
            return (lamda / 2) * sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )


class FoolsGold(object):
    def __init__(self, use_memory=False):
        self.memory = None
        self.memory_dict = dict()
        self.wv_history = []
        self.use_memory = use_memory

    def aggregate_gradients(self, client_grads, names):
        cur_time = time.time()
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()

        # if self.memory is None:
        #     self.memory = np.zeros((num_clients, grad_len))
        self.memory = np.zeros((num_clients, grad_len))
        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]] += grads[i]
            else:
                self.memory_dict[names[i]] = copy.deepcopy(grads[i])
            self.memory[i] = self.memory_dict[names[i]]
        # self.memory += grads

        if self.use_memory:
            wv, alpha = self.foolsgold(self.memory)  # Use FG
        else:
            wv, alpha = self.foolsgold(grads)  # Use FG
        logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(
                len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)
        print('model aggregation took {}s'.format(time.time() - cur_time))
        return agg_grads, wv, alpha

    def foolsgold(self, grads):
        """
        :param grads:
        :return: compute similatiry and return weightings
        """
        n_clients = grads.shape[0]
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)

        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))

        wv[wv > 1] = 1
        wv[wv < 0] = 0

        alpha = np.max(cs, axis=1)

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # wv is the weight
        return wv, alpha
