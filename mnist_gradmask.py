import utils.csv_record as csv_record
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import main
import test
import copy
import config
from mnist_trigger import mnist_trigger


def grad_mask_mnist(helper, local_model, target_model, ratio=0.5):
    """Generate a gradient mask based on the given dataset"""
    model = copy.deepcopy(local_model)
    model.copy_params(target_model.state_dict())
    model.train()
    model.zero_grad()
    count = 0
    for poison_id in helper.params['adversary_list']:
        _, data_iterator = helper.train_data[poison_id]
        for batch_id, (datas, labels) in enumerate(data_iterator):
            input = datas.cuda()
            label = labels.cuda()
            output = model(input)
            loss = nn.functional.cross_entropy(output, label)
            loss.backward(retain_graph=True)

    mask_grad_list = []
    grad_list = []
    grad_abs_sum_list = []
    k_layer = 0
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            grad_list.append(parms.grad.abs().view(-1))
            grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())
            k_layer += 1
    grad_list = torch.cat(grad_list).cuda()
    _, indices = torch.topk(-1 * grad_list, int(len(grad_list) * ratio))  # 保留
    mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
    mask_flat_all_layer[indices] = 1.0
    count = 0
    percentage_mask_list = []
    k_layer = 0
    grad_abs_percentage_list = []
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            gradients_length = len(parms.grad.abs().view(-1))
            mask_flat = mask_flat_all_layer[count:count + gradients_length].cuda()
            mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

            count += gradients_length

            percentage_mask1 = mask_flat.sum().item() / float(gradients_length) * 100.0

            percentage_mask_list.append(percentage_mask1)

            grad_abs_percentage_list.append(grad_abs_sum_list[k_layer] / np.sum(grad_abs_sum_list))

            k_layer += 1

    model.zero_grad()

    return mask_grad_list


def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy)


def MnistTrain(helper, start_epoch, local_model, target_model, is_poison, agent_name_keys, noise_trigger,
               intinal_trigger):
    epochs_submit_update_dict = dict()  # 字典
    epochs_change_update_dict = dict()
    num_samples_dict = dict()
    current_number_of_adversaries = 0  # 攻击者数量
    for temp_name in agent_name_keys:  # agent是包含正常用户、攻击者的列表
        if temp_name in helper.params['adversary_list']:
            current_number_of_adversaries += 1
    poisonupdate_dict = dict()  # 被控制用户的模型
    user_grads = []
    server_update = dict()
    models = copy.deepcopy(local_model)
    models.copy_params(target_model.state_dict())
    IsTrigger = False  # 是否进行过trigger调整
    IsGrad_mask = False
    tuned_trigger = noise_trigger
    mask_grad_list = []
    poicount = []
    ###每个用户（包括正常、攻击）

    for model_id in range(helper.params['no_models']):
        epochs_local_update_list = []
        last_local_model = dict()
        client_grad = []  # only works for aggr_epoch_interval=1
        for name, data in target_model.state_dict().items():
            last_local_model[name] = target_model.state_dict()[name].clone()  # 获取上一轮模型参数

        agent_name_key = agent_name_keys[model_id]
        model = local_model
        normalmodel = copy.deepcopy(local_model)
        # 获取上一轮全局模型
        model.copy_params(target_model.state_dict())
        normalmodel.copy_params(target_model.state_dict())  # 未受到攻击时的模型
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        normalOptimizer = torch.optim.SGD(normalmodel.parameters(), lr=helper.params['lr'],
                                          momentum=helper.params['momentum'],
                                          weight_decay=helper.params['decay'])
        model.train()
        normalmodel.train()
        adversarial_index = -1
        localmodel_poison_epochs = helper.params['poison_epochs']
        if is_poison and agent_name_key in helper.params['adversary_list']:
            localmodel_poison_epochs = helper.params['sum_poison_epochs']

        # 每轮训练
        for epoch in range(start_epoch, start_epoch + helper.params['aggr_epoch_interval']):
            target_params_variables = dict()
            for name, param in target_model.named_parameters():
                target_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)

            # 保证当前的攻击者数量满足假设
            if helper.params['aggregation_methods'] == config.AGGR_KRUM or helper.params[
                'aggregation_methods'] == config.AGGR_MKRUM or helper.params[
                'aggregation_methods'] == config.AGGR_TRIMMED_MEAN:
                if current_number_of_adversaries > 5:
                    if agent_name_key in helper.params['adversary_list']:
                        poicount.append(agent_name_key)
                    if len(poicount) > 5:
                        is_poison = False
            elif helper.params['aggregation_methods'] == config.AGGR_BULYAN:
                if current_number_of_adversaries > 1:
                    if agent_name_key in helper.params['adversary_list']:
                        poicount.append(agent_name_key)
                    if len(poicount) > 1:
                        is_poison = False

            # 如果本轮进行投毒，并且用户在攻击者列表种
            if is_poison and agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                main.logger.info('poison_now')

                if not IsTrigger:
                    tuned_trigger = mnist_trigger(helper, local_model, target_model, noise_trigger, intinal_trigger)
                    IsTrigger = True

                # 生成梯度掩码，限制参数更新范围，每个epoch只生成一次mask
                if helper.params['gradmask_ratio'] != 1:
                    main.logger.info('make grad_mask start !!!')
                    if not IsGrad_mask:
                        mask_grad_list = grad_mask_mnist(helper, local_model, target_model,
                                                         helper.params['gradmask_ratio'])
                        IsGrad_mask = True
                    main.logger.info('make grad_mask end !!!')

                # main.logger.info("进行投毒的trigger：{}".format(tuned_trigger))
                poison_lr = helper.params['poison_lr']
                internal_epoch_num = helper.params['internal_poison_epochs']
                step_lr = helper.params['poison_step_lr']

                # 正常训练时优化器：noramlOptimizer
                main.logger.info('normally training')
                _, data_iterator = helper.train_data[agent_name_key]
                normalData_size = 0
                for batch_id, batch in enumerate(data_iterator):
                    normalOptimizer.zero_grad()
                    normalData, normalTargets = helper.get_batch(data_iterator, batch, evaluation=False)
                    normalData_size += len(normalData)
                    normaloutput = normalmodel(normalData)
                    loss = nn.functional.cross_entropy(normaloutput, normalTargets)
                    loss.backward()
                    normalOptimizer.step()
                normal_params_variables = dict()
                for name, param in normalmodel.named_parameters():
                    normal_params_variables[name] = normalmodel.state_dict()[name].clone().detach().requires_grad_(
                        False)
                # # 如果投毒保存正常模型更新，以防止不发起攻击
                # normalmodel_updates_dict = dict()
                #
                # for name, data in normalmodel.state_dict().items():
                #     normalmodel_updates_dict[name] = torch.zeros_like(data)
                #     normalmodel_updates_dict[name] = (data - last_local_model[name])

                main.logger.info('save normal model, normally training ending')

                # 进行攻击时的优化器
                poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
                                                   momentum=helper.params['momentum'],
                                                   weight_decay=helper.params['decay'])

                # 动态调整学习率
                scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                                 milestones=[0.2 * internal_epoch_num,
                                                                             0.8 * internal_epoch_num], gamma=0.1)
                temp_local_epoch = (epoch - 1) * internal_epoch_num

                for internal_epoch in range(1, internal_epoch_num + 1):
                    temp_local_epoch += 1
                    _, data_iterator = helper.train_data[agent_name_key]
                    poison_data_count = 0
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list = []

                    for batch_id, batch in enumerate(data_iterator):
                        # 生成后门样本训练
                        data, targets, poison_num = helper.get_poison_batch(batch, noise_trigger=tuned_trigger,
                                                                            evaluation=False)
                        poison_optimizer.zero_grad()
                        dataset_size += len(data)
                        poison_data_count += poison_num
                        output = model(data)
                        class_loss = nn.functional.cross_entropy(output, targets)
                        loss = class_loss
                        if helper.params['attack_methods'] == 'SCBA':
                            malDistance_Loss = helper.model_dist_norm_var(model, normal_params_variables)
                            distance_loss = helper.model_dist_norm_var(model, target_params_variables)
                            ###与其他被控制用户模型更新的相似度
                            sum_cs = 0
                            otheradnum = 0
                            if poisonupdate_dict:
                                for otherAd in helper.params['adversary_list']:
                                    if otherAd == agent_name_key:
                                        continue
                                    else:
                                        if otherAd in poisonupdate_dict.keys():
                                            otheradnum += 1
                                            otherAd_variables = dict()
                                            for name, data in poisonupdate_dict[otherAd].items():
                                                otherAd_variables[name] = poisonupdate_dict[otherAd][
                                                    name].clone().detach().requires_grad_(False)

                                            sum_cs += helper.model_cosine_similarity(model, otherAd_variables)
                            loss = class_loss + helper.params['alpha_loss'] * distance_loss + helper.params[
                                'beta_loss'] * malDistance_Loss + \
                                   helper.params['gamma_loss'] * sum_cs
                            loss.backward()
                        elif helper.params['attack_methods'] == 'a little':
                            distance_loss = helper.model_dist_norm_var(model, target_params_variables)
                            loss = 0.2 * class_loss + 0.8 * distance_loss
                            loss.backward()
                        elif helper.params['attack_methods'] == 'sybil attack':
                            loss = class_loss
                            loss.backward()
                        elif helper.params['attack_methods'] == 'modelre':
                            loss = class_loss
                            loss.backward()

                        # 在梯度上施加 mask，限制更新
                        if helper.params['gradmask_ratio'] != 1:
                            apply_grad_mask(model, mask_grad_list)

                        # get gradients
                        if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD or \
                                helper.params['aggregation_methods'] == config.AGGR_KRUM or \
                                helper.params['aggregation_methods'] == config.AGGR_TRIMMED_MEAN or \
                                helper.params['aggregation_methods'] == config.AGGR_BULYAN or \
                                helper.params['aggregation_methods'] == config.AGGR_DNC or \
                                helper.params['aggregation_methods'] == config.AGGR_MEDIAN or \
                                helper.params['aggregation_methods'] == config.AGGR_MKRUM:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        poison_optimizer.step()

                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                    if step_lr:
                        scheduler.step()
                        main.logger.info(f'Current lr: {scheduler.get_lr()}')

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    distance = helper.model_dist_norm_var(model, target_params_variables)
                    main.logger.info(
                        '___PoisonTrain {} , epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%), train_poison_data_count: {}, distance：{}'.format(model.name, epoch,
                                                                                                     agent_name_key,
                                                                                                     internal_epoch,
                                                                                                     total_l, correct,
                                                                                                     dataset_size,
                                                                                                     acc,
                                                                                                     poison_data_count,
                                                                                                     distance))
                    csv_record.train_result.append(
                        [agent_name_key, temp_local_epoch,
                         epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])
                    num_samples_dict[agent_name_key] = dataset_size


                # internal epoch finish
                main.logger.info(f'Global model norm: {helper.model_global_norm(target_model)}.')
                main.logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                                 f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

                if helper.params['attack_methods'] == 'modelre':
                    main.logger.info(f'will scale.')
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                                                                                   model=model, is_poison=False,
                                                                                   visualize=False,
                                                                                   agent_name_key=agent_name_key)
                    csv_record.test_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                          epoch=epoch,
                                                                                          model=model,
                                                                                          noise_trigger=tuned_trigger,
                                                                                          is_poison=True,
                                                                                          visualize=False,
                                                                                          agent_name_key=agent_name_key)
                    csv_record.posiontest_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                    clip_rate = helper.params['scale_weights_poison']
                    main.logger.info(f"Scaling by  {clip_rate}")
                    if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD or \
                            helper.params['aggregation_methods'] == config.AGGR_KRUM or \
                            helper.params['aggregation_methods'] == config.AGGR_TRIMMED_MEAN or \
                            helper.params['aggregation_methods'] == config.AGGR_BULYAN or \
                            helper.params['aggregation_methods'] == config.AGGR_DNC or \
                            helper.params['aggregation_methods'] == config.AGGR_MEDIAN or \
                            helper.params['aggregation_methods'] == config.AGGR_MKRUM:
                        client_gradss = [i * clip_rate for i in client_grad]
                        client_grad = client_gradss
                    else:
                        for key, value in model.state_dict().items():
                            target_value = last_local_model[key]
                            new_value = target_value + (value - target_value) * clip_rate
                            model.state_dict()[key].copy_(new_value)
                    distance = helper.model_dist_norm(model, target_params_variables)
                    main.logger.info(
                        f'Scaled Norm after poisoning: '
                        f'{helper.model_global_norm(model)}, distance: {distance}')
                    csv_record.scale_temp_one_row.append(epoch)
                    csv_record.scale_temp_one_row.append(round(distance, 4))

                distance = helper.model_dist_norm(model, target_params_variables)
                main.logger.info(f"Total norm for {current_number_of_adversaries} "
                                 f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")

            # 如果本轮不投毒
            else:
                temp_local_epoch = (epoch - 1) * helper.params['internal_epochs']
                for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
                    temp_local_epoch += 1

                    _, data_iterator = helper.train_data[agent_name_key]
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list = []
                    old_gradient = {}
                    for batch_id, batch in enumerate(data_iterator):

                        optimizer.zero_grad()
                        data, targets = helper.get_batch(data_iterator, batch, evaluation=False)
                        dataset_size += len(data)
                        output = model(data)
                        loss = nn.functional.cross_entropy(output, targets)
                        loss.backward()

                        # get gradients
                        if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD or \
                                helper.params['aggregation_methods'] == config.AGGR_KRUM or \
                                helper.params['aggregation_methods'] == config.AGGR_TRIMMED_MEAN or \
                                helper.params['aggregation_methods'] == config.AGGR_BULYAN or \
                                helper.params['aggregation_methods'] == config.AGGR_DNC or \
                                helper.params['aggregation_methods'] == config.AGGR_MEDIAN or \
                                helper.params['aggregation_methods'] == config.AGGR_MKRUM:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        optimizer.step()

                        if helper.params['FL_WBC'] == True:
                            if batch_id != 0:
                                for name, p in model.named_parameters():
                                    if 'weight' in name:
                                        grad_tensor = p.grad.data.cpu().numpy()
                                        grad_diff = grad_tensor - old_gradient[name]
                                        pertubation = np.random.laplace(0, helper.params['pert_strength'],
                                                                        size=grad_tensor.shape).astype(np.float32)
                                        pertubation = np.where(abs(grad_diff) > abs(pertubation), 0, pertubation)
                                        p.data = torch.from_numpy(
                                            p.data.cpu().numpy() + pertubation * helper.params['lr']).to(
                                            config.device)
                            for name, p in model.named_parameters():
                                if 'weight' in name:
                                    old_gradient[name] = p.grad.data.cpu().numpy()

                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    main.logger.info(
                        '___Train {},  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%)'.format(model.name, epoch, agent_name_key, internal_epoch,
                                                           total_l, correct, dataset_size,
                                                           acc))
                    csv_record.train_result.append([agent_name_key, temp_local_epoch,
                                                    epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])

                    num_samples_dict[agent_name_key] = dataset_size

                    # if helper.params["batch_track_distance"]:
                    #     main.logger.info(
                    #         f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                    #         f'Distance to the global model: {dis2global_list}. ')

                # test local model after internal epoch finishing
                epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                                                                               model=model, is_poison=False,
                                                                               visualize=True,
                                                                               agent_name_key=agent_name_key)
                csv_record.test_result.append([agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total,
                                               helper.model_dist_norm(model, target_params_variables)])

            # 如果本次整个训练进行投毒
            if is_poison:
                # 如果用户是攻击者，并且在本轮进行投毒,输出投毒准确率
                if agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                          epoch=epoch,
                                                                                          model=model,
                                                                                          noise_trigger=tuned_trigger,
                                                                                          is_poison=True,
                                                                                          visualize=True,
                                                                                          agent_name_key=agent_name_key)
                    csv_record.posiontest_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total,
                         helper.model_dist_norm(model, target_params_variables)])

                    Ad_model_dict = dict()
                    for name, data in model.state_dict().items():
                        Ad_model_dict[name] = model.state_dict()[name].clone().detach().requires_grad_(False)
                    poisonupdate_dict[agent_name_key] = Ad_model_dict

            # 本轮进行模型权重更新
            # update the model weight

            local_model_update_dict = dict()
            local_model_weightss_dict = dict()

            for name, data in model.state_dict().items():
                local_model_update_dict[name] = torch.zeros_like(data)
                local_model_update_dict[name] = (data - last_local_model[name])
                local_model_weightss_dict[name] = torch.zeros_like(data)
                local_model_weightss_dict[name] = copy.deepcopy(data)
                last_local_model[name] = copy.deepcopy(data)


            if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD or \
                    helper.params['aggregation_methods'] == config.AGGR_KRUM or \
                    helper.params['aggregation_methods'] == config.AGGR_TRIMMED_MEAN or \
                    helper.params['aggregation_methods'] == config.AGGR_BULYAN or \
                    helper.params['aggregation_methods'] == config.AGGR_DNC or \
                    helper.params['aggregation_methods'] == config.AGGR_MEDIAN or \
                    helper.params['aggregation_methods'] == config.AGGR_MKRUM:
                epochs_local_update_list.append(client_grad)

            elif helper.params['aggregation_methods'] == config.AGGR_FLAME:  # 传的是参数
                epochs_local_update_list.append(local_model_weightss_dict)
            else:
                epochs_local_update_list.append(local_model_update_dict)

        #每个用户将自己的模型更新发送到
        epochs_change_update_dict[agent_name_key] = epochs_local_update_list

    for name, params in epochs_change_update_dict.items():
        epochs_submit_update_dict[name] = epochs_change_update_dict[name]

    if helper.params['aggregation_methods'] == config.AGGR_FLTRUST:
        last_params_variabless = dict()
        for name, param in target_model.state_dict().items():
            last_params_variabless[name] = target_model.state_dict()[name].clone()
        optimizer = torch.optim.SGD(models.parameters(), lr=0.1,
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])  # 0.01
        models.train()
        temp_local_epoch = start_epoch - 1
        for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
            temp_local_epoch += 1
            data_iterator = helper.test_data  # 用测试集进行训练一个全局模型
            total_loss = 0.
            correct = 0
            dataset_size = 0
            for batch_id, batch in enumerate(data_iterator):
                if dataset_size > 500:  # 500
                    break
                data, targets = helper.get_batch(data_iterator, batch, evaluation=False)
                optimizer.zero_grad()
                dataset_size += len(data)
                output = models(data)
                loss = nn.functional.cross_entropy(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.data
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size

            main.logger.info(
                '___Train {},  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.4f}%)'.format(models.name, start_epoch, '99', internal_epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))
        # update the weight and bias
        for name, data in models.state_dict().items():
            server_update[name] = torch.zeros_like(data)
            server_update[name] = (data - last_params_variabless[name])
            last_params_variabless[name] = copy.deepcopy(data)

    return epochs_submit_update_dict, num_samples_dict, user_grads, server_update, tuned_trigger  # 返回本轮训练得到的用户模型以及各个用户的数据量
