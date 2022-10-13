import argparse
import datetime
import logging
import torch
import train
import test
from loan_helper import LoanHelper
from image_helper import ImageHelper
from ddos_helper import DDosHelper
from mnist_helper import MnistHelper
from cifar100_helper import Cifar100Helper
from fmnist_helper import FMnistHelper
import utils.csv_record as csv_record
from smooth import certificate_over_model, certificate_over_model_lodd
import yaml
import time
import numpy as np
import random
import config
import copy
from torch.autograd import Variable

CUDA_DEVICE = 0

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger("logger")

criterion = torch.nn.CrossEntropyLoss()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)
np.random.seed(1)


def cuda(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


if __name__ == '__main__':
    print('Start training')
    np.random.seed(1)
    time_start_load_everything = time.time()
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()

 
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f, Loader=yaml.FullLoader)

    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
   
    if params_loaded['type'] == config.TYPE_CIFAR:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'cifar'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_CIFAR100:
        helper = Cifar100Helper(current_time=current_time, params=params_loaded,
                                name=params_loaded.get('name', 'cifar100'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_MNIST:
        helper = MnistHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'mnist'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_FMNIST:
        helper = FMnistHelper(current_time=current_time, params=params_loaded,
                              name=params_loaded.get('name', 'fmnist'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_DDOS:
        helper = DDosHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'ddos'))
        helper.load_data(params_loaded)
    elif params_loaded['type'] == config.TYPE_LOAN:
        helper = LoanHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'loan'))
        helper.load_data(params_loaded)
    else:
        helper = None

    logger.info(f'load data done')
    helper.create_model()
    logger.info(f'create model done')
    ### Create models
    if helper.params['is_poison']:
        logger.info(f"Poisoned following participants: {(helper.params['adversary_list'])}")
    else:
        helper.params['adversary_list'] = list()

    best_loss = float('inf')
    logger.info(f"We use following environment for graphs:  {helper.params['environment_name']}")

   
    weight_accumulator = helper.init_weight_accumulator(helper.target_model)

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)

    submit_update_dict = None
    num_no_progress = 0

    # load trigger
    if params_loaded['type'] == config.TYPE_FMNIST or params_loaded['type'] == config.TYPE_MNIST:
        intinal_trigger = helper.params['trigger_value']  
        noise_trigger = copy.deepcopy(intinal_trigger)
    elif params_loaded['type'] == config.TYPE_LOAN or params_loaded['type'] == config.TYPE_DDOS:
        intinal_trigger = helper.params['trigger_value']  
        noise_trigger = copy.deepcopy(intinal_trigger)
    elif params_loaded['type'] == config.TYPE_CIFAR or params_loaded['type'] == config.TYPE_CIFAR100:
        data_iterator = helper.test_data
        for batch_id, (datas, labels) in enumerate(data_iterator):
            x = Variable(cuda(datas, True))
            sz = x.size()[1:]
            intinal_trigger = torch.zeros(sz)
            break
        poison_patterns = []
        poison_patterns = poison_patterns + helper.params['sum_poison_pattern']
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            intinal_trigger[0][pos[0]][pos[1]] = 1  # +delta i  #？
            intinal_trigger[1][pos[0]][pos[1]] = 1  # +delta i
            intinal_trigger[2][pos[0]][pos[1]] = 1  # +delta i

        noise_trigger = copy.deepcopy(intinal_trigger)

   
    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1, helper.params['aggr_epoch_interval']):
        start_time = time.time()
        t = time.time()
        
        agent_name_keys = helper.participants_list
        adversarial_name_keys = []
        if helper.params['is_random_namelist']:
            if helper.params['is_random_adversary']:  # random choose , maybe don't have advasarial
                agent_name_keys = random.sample(helper.participants_list, helper.params['no_models'])
                for _name_keys in agent_name_keys:
                    if _name_keys in helper.params['adversary_list']:
                        adversarial_name_keys.append(_name_keys)
            else:  # must have advasarial if this epoch is in their poison epoch
                ongoing_epochs = list(range(epoch, epoch + helper.params['aggr_epoch_interval']))
                for idx in range(0, len(helper.params['adversary_list'])):
                    for ongoing_epoch in ongoing_epochs:
                        if ongoing_epoch in helper.params['sum_poison_epochs']:
                            if helper.params['adversary_list'][idx] not in adversarial_name_keys:
                                adversarial_name_keys.append(helper.params['adversary_list'][idx])

                nonattacker = []
                for adv in helper.params['adversary_list']:
                    if adv not in adversarial_name_keys:
                        nonattacker.append(copy.deepcopy(adv))
                benign_num = helper.params['no_models'] - len(adversarial_name_keys)
                random_agent_name_keys = random.sample(helper.benign_namelist + nonattacker, benign_num)
                agent_name_keys = adversarial_name_keys + random_agent_name_keys
        else:
            if helper.params['is_random_adversary'] == False:
                adversarial_name_keys = copy.deepcopy(helper.params['adversary_list'])
        logger.info(f'Server Epoch:{epoch} choose agents : {agent_name_keys}.')

        current_number_of_adversaries = 0
        for temp_name in agent_name_keys: 
            if temp_name in helper.params['adversary_list']:
                current_number_of_adversaries += 1
        logger.info(f'current malicious clients={current_number_of_adversaries}')

      
        epochs_submit_update_dict, num_samples_dict, user_grad, server_update, tuned_trigger = train.train(
            helper=helper,
            start_epoch=epoch,
            local_model=helper.local_model,
            target_model=helper.target_model,
            is_poison=helper.params['is_poison'],
            agent_name_keys=agent_name_keys,
            noise_trigger=noise_trigger,
            intinal_trigger=intinal_trigger)

      
        noise_trigger = tuned_trigger

        logger.info(f'time spent on training: {time.time() - t}')
        weight_accumulator, updates = helper.accumulate_weight(weight_accumulator, epochs_submit_update_dict,
                                                               agent_name_keys, num_samples_dict)
        is_updated = True

      
        print(helper.params['aggregation_methods'])
        if helper.params['aggregation_methods'] == config.AGGR_MEAN:
            # Average the models
            is_updated = helper.average_shrink_models(weight_accumulator=weight_accumulator,
                                                      target_model=helper.target_model,
                                                      epoch_interval=helper.params['aggr_epoch_interval'])
            num_oracle_calls = 1
        elif helper.params['aggregation_methods'] == config.AGGR_FEDAVGLR:
            # Average the models
            is_updated = helper.fedavglr(weight_accumulator=weight_accumulator,
                                         target_model=helper.target_model,
                                         epocht=epoch,
                                         epoch_interval=helper.params['aggr_epoch_interval'])
            num_oracle_calls = 1
            logger.info(1 / epoch)
        elif helper.params['aggregation_methods'] == config.AGGR_GEO_MED:
            maxiter = helper.params['geom_median_maxiter']
            num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(helper.target_model,
                                                                                                  updates,
                                                                                                  maxiter=maxiter)


        elif helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
            is_updated, names, weights, alphas = helper.foolsgold_update(helper.target_model, updates)

            num_oracle_calls = 1

        elif helper.params['aggregation_methods'] == config.AGGR_KRUM:
            is_updated = helper.krum(target_model=helper.target_model, updates=updates,
                                     users_count=int(helper.params['no_models']),
                                     corrupted_count=current_number_of_adversaries, agent_name_keys=agent_name_keys)

            num_oracle_calls = 1
        elif helper.params['aggregation_methods'] == config.AGGR_TRIMMED_MEAN:

            is_updated = helper.trimmed_mean(target_model=helper.target_model, updates=updates,
                                             users_count=int(helper.params['no_models']),
                                             corrupted_count=current_number_of_adversaries)

            num_oracle_calls = 1

        elif helper.params['aggregation_methods'] == config.AGGR_BULYAN:
            is_updated = helper.bulyan(target_model=helper.target_model, updates=updates,
                                       users_count=int(helper.params['no_models']),
                                       corrupted_count=current_number_of_adversaries)

            num_oracle_calls = 1
        elif helper.params['aggregation_methods'] == config.AGGR_CRFL:
            is_updated = helper.CRFL(weight_accumulator=weight_accumulator, target_model=helper.target_model,
                                     epoch=epoch)


        elif helper.params['aggregation_methods'] == config.AGGR_MEDIAN:
            is_updated = helper.median(target_model=helper.target_model, updates=updates,
                                       users_count=int(helper.params['no_models']))


        elif helper.params['aggregation_methods'] == config.AGGR_MKRUM:
            is_updated = helper.mkrum(target_model=helper.target_model, updates=updates,
                                      corrupted_count=current_number_of_adversaries,
                                      users_count=int(helper.params['no_models']))

        elif helper.params['aggregation_methods'] == config.AGGR_FLTRUST:
            is_updated = helper.fltrust(target_model=helper.target_model, updates=updates,
                                        server_update=server_update, agent_name_keys=agent_name_keys)

        elif helper.params['aggregation_methods'] == config.AGGR_FEDLDP:
            is_updated = helper.fedLDP(target_model=helper.target_model, updates=updates,
                                       agent_name_keys=agent_name_keys)

        elif helper.params['aggregation_methods'] == config.AGGR_FEDCDP:
            is_updated = helper.fedCDP(target_model=helper.target_model, updates=updates,
                                       agent_name_keys=agent_name_keys, epoch=epoch)

        elif helper.params['aggregation_methods'] == config.AGGR_DNC:
           
            is_updated = helper.DnC(target_model=helper.target_model, updates=updates,
                                    users_count=int(helper.params["no_models"]), m=current_number_of_adversaries)

        elif helper.params['aggregation_methods'] == config.AGGR_FLAME:
            is_updated = helper.flame(target_model=helper.target_model, updates=updates,
                                      agent_name_keys=agent_name_keys)

        # clear the weight_accumulator
        weight_accumulator = helper.init_weight_accumulator(helper.target_model)
       
        temp_global_epoch = epoch + helper.params['aggr_epoch_interval'] - 1

        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=temp_global_epoch,
                                                                       model=helper.target_model, is_poison=False,
                                                                       visualize=False, agent_name_key="global")
        csv_record.test_result.append(["global", temp_global_epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])
        if len(csv_record.scale_temp_one_row) > 0:
            csv_record.scale_temp_one_row.append(round(epoch_acc, 4))

        # if helper.params['is_poison']:
        epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                    epoch=temp_global_epoch,
                                                                                    model=helper.target_model,
                                                                                    noise_trigger=tuned_trigger,
                                                                                    is_poison=True,
                                                                                    visualize=False,
                                                                                    agent_name_key="global")

        csv_record.posiontest_result.append(
                ["global", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])

      
        helper.save_model(epoch=epoch, val_loss=epoch_loss)
        logger.info(f'Done in {time.time() - start_time} sec.')
        csv_record.save_result_csv(epoch, helper.params['is_poison'], helper.folder_path)

    if helper.params['aggregation_methods'] == config.AGGR_CRFL:
        if params_loaded['type'] == config.TYPE_DDOS or params_loaded['type'] == config.TYPE_LOAN:
            pa_exp, pb_exp, is_acc, pa_exp_poison, pb_exp_poison, is_acc_poison = certificate_over_model_lodd(
                models=helper.target_model, helper=helper, N_m=helper.params['N_m'],
                sigma=helper.params['test_sigma'], tuned_trigger=noise_trigger)
        else:
            pa_exp, pb_exp, is_acc, pa_exp_poison, pb_exp_poison, is_acc_poison = certificate_over_model(
                models=helper.target_model, helper=helper, N_m=helper.params['N_m'], sigma=helper.params['test_sigma'],
                tuned_trigger=noise_trigger)

        foldername = helper.params['smoothed_fname'].split('/')
        epoch = int(foldername[-1].split('_')[-1])
        foldername = helper.folder_path
        # prepare output file

        output_fname = os.path.join(foldername, "pred_clean_Epoch%dM%dSigma%.4f.txt" % (
            epoch, helper.params['N_m'], helper.params['test_sigma']))
        f = open(output_fname, 'w')
        print("idx\tpa_exp\tpb_exp\tis_acc", file=f, flush=True)

        for i in range(len(pa_exp)):  # len of test data set
            print("{}\t{}\t{}\t{}".format(i, pa_exp[i], pb_exp[i], is_acc[i]), file=f, flush=True)

        logger.info("is_acc for clean data-clean label %.4f " % (float(sum(is_acc)) / len(is_acc)))
        f.close()
        logger.info("save to %s" % output_fname)

        # prepare output file
        output_fname = os.path.join(foldername, "pred_poison_Epoch%dM%dSigma%.4f.txt" % (
            epoch, helper.params['N_m'], helper.params['test_sigma']))
        f = open(output_fname, 'w')
        print("idx\tpa_exp\tpb_exp\tis_acc", file=f, flush=True)

        for i in range(len(pa_exp_poison)):  # len of test data set
            print("{}\t{}\t{}\t{}".format(i, pa_exp_poison[i], pb_exp_poison[i], is_acc_poison[i]), file=f,
                  flush=True)

        logger.info("is_acc for poison data-clean label %.4f " % (float(sum(is_acc_poison)) / len(is_acc_poison)))
        f.close()
        logger.info("save to %s" % output_fname)

    logger.info('Saving all the graphs.')
    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")
