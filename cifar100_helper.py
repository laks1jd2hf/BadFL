from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np
from model.resnet_cifar import ResNet18
from model.resnet_cifar100 import ResNet18100
from model.MNISTnet import MnistNet
from model.resnet_tinyimage import resnet18

logger = logging.getLogger("logger")
import config
from config import device
import copy
import yaml
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import datetime
import json


class Cifar100Helper(Helper):

    def create_model(self):
        local_model = None
        target_model = None
        if self.params['type'] == config.TYPE_CIFAR:
            local_model = ResNet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = ResNet18(name='Target',
                                    created_time=self.params['current_time'])

        elif self.params['type'] == config.TYPE_CIFAR100:
            local_model = ResNet18100(name='Local',
                                      created_time=self.params['current_time'])
            target_model = ResNet18100(name='Target',
                                       created_time=self.params['current_time'])

        elif self.params['type'] == config.TYPE_MNIST:
            local_model = MnistNet(name='Local',
                                   created_time=self.params['current_time'])
            target_model = MnistNet(name='Target',
                                    created_time=self.params['current_time'])

        elif self.params['type'] == config.TYPE_TINYIMAGENET:

            local_model = resnet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = resnet18(name='Target',
                                    created_time=self.params['current_time'])

        local_model = local_model.to(device)  # 将模型加载到指定设备上。
        target_model = target_model.to(device)
        if self.params['resumed_model']:
            if torch.cuda.is_available():
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}", map_location=device)
            else:
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}", map_location='cpu')
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch'] + 1
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model

    def build_classes_dict(self):  # 创建类别列表
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):  # for cifar: 50000; for tinyimagenet: 100000
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        return cifar_classes

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.数据索引列表
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.#每一维代表着每一个类别里抽取多少数据
        """

        cifar_classes = self.classes_dict  # 类别
        class_size = len(cifar_classes[0])  # for cifar: 5000
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())  # for cifar: 10

        image_nums = []
        scalenum = 3
        if self.params['aggregation_methods'] == config.AGGR_BULYAN or self.params[
            'aggregation_methods'] == config.AGGR_KRUM or self.params[
            'aggregation_methods'] == config.AGGR_MKRUM or self.params[
            'aggregation_methods'] == config.AGGR_TRIMMED_MEAN:
            scalenum = 10
        elif self.params['aggregation_methods'] == config.AGGR_GEO_MED:
            scalenum = 1
        else:
            scalenum = 3
        for n in range(no_classes):
            image_num = []
            random.shuffle(cifar_classes[n])  # 将列表随机排序
            sampled_probabilities = int(scalenum) * class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                image_num.append(len(sampled_list))
                per_participant_list[user].extend(sampled_list)
                random.shuffle(cifar_classes[n])  # 将列表随机排序
                # cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
            image_nums.append(image_num)

        # self.draw_dirichlet_plot(no_classes,no_participants,image_nums,alpha)
        return per_participant_list

    def draw_dirichlet_plot(self, no_classes, no_participants, image_nums, alpha):
        fig = plt.figure(figsize=(10, 5))
        s = np.empty([no_classes, no_participants])
        for i in range(0, len(image_nums)):
            for j in range(0, len(image_nums[0])):
                s[i][j] = image_nums[i][j]
        s = s.transpose()
        left = 0
        y_labels = []
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, no_participants))
        for k in range(no_classes):
            y_labels.append('Label ' + str(k))
        vis_par = [0, 10, 20, 30]
        for k in range(no_participants):
            # for k in vis_par:
            color = category_colors[k]
            plt.barh(y_labels, s[k], left=left, label=str(k), color=color)
            widths = s[k]
            xcenters = left + widths / 2
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            # for y, (x, c) in enumerate(zip(xcenters, widths)):
            #     plt.text(x, y, str(int(c)), ha='center', va='center',
            #              color=text_color,fontsize='small')
            left += s[k]
        plt.legend(ncol=20, loc='lower left', bbox_to_anchor=(0, 1), fontsize=4)  #
        # plt.legend(ncol=len(vis_par), bbox_to_anchor=(0, 1),
        #            loc='lower left', fontsize='small')
        plt.xlabel("Number of Images", fontsize=16)
        # plt.ylabel("Label 0 ~ 199", fontsize=16)
        # plt.yticks([])
        fig.tight_layout(pad=0.1)
        # plt.ylabel("Label",fontsize='small')
        fig.savefig(self.folder_path + '/Num_Img_Dirichlet_Alpha{}.pdf'.format(alpha))

    def poison_test_dataset(self):
        logger.info('get poison test loader')
        # delete the test data with target label
        test_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(self.test_dataset)))
        for image_ind in test_classes[self.params['poison_label_swap']]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind)
        poison_label_inds = test_classes[self.params['poison_label_swap']]

        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               range_no_id)), \
               torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               poison_label_inds))

    def load_data(self):
        logger.info('Loading data')
        dataPath = './data'
        # 加载数据
        if self.params['type'] == config.TYPE_CIFAR:
            ### data load
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

            self.train_dataset = datasets.CIFAR10(dataPath, train=True, download=True,
                                                  transform=transform_train)

            self.test_dataset = datasets.CIFAR10(dataPath, train=False, transform=transform_test)

        if self.params['type'] == config.TYPE_CIFAR100:
            ### data load
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
                transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            self.train_dataset = datasets.CIFAR100(dataPath, train=True, download=True,
                                                   transform=transform_train)

            self.test_dataset = datasets.CIFAR100(dataPath, train=False, transform=transform_test)

        elif self.params['type'] == config.TYPE_MNIST:

            self.train_dataset = datasets.MNIST('./data', train=True, download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    # transforms.Normalize((0.1307,), (0.3081,))
                                                ]))
            self.test_dataset = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]))
        elif self.params['type'] == config.TYPE_TINYIMAGENET:

            _data_transforms = {
                'train': transforms.Compose([
                    # transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                    # transforms.Resize(224),
                    transforms.ToTensor(),
                ]),
            }
            _data_dir = './data/tiny-imagenet-200/'
            self.train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                                      _data_transforms['train'])
            self.test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'),
                                                     _data_transforms['val'])
            logger.info('reading data done')

        self.classes_dict = self.build_classes_dict()  # 创建类别列表
        logger.info('build_classes_dict done')
        # 根据用户划分数据，数据分布
        if self.params['sampling_dirichlet']:
            ## sample indices for participants using Dirichlet distribution
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params['number_of_total_participants'],  # 100
                alpha=self.params['dirichlet_alpha'])
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]
        else:
            ## sample indices for participants that are equally
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [(pos, self.get_train_old(all_range, pos))
                             for pos in range(self.params['number_of_total_participants'])]

        logger.info('train loaders done')
        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.test_data_poison, self.test_targetlabel_data = self.poison_test_dataset()  # 加载投毒后的测试集，去掉目标样本

        self.advasarial_namelist = self.params['adversary_list']  # 加载攻击用户列表

        if self.params['is_random_namelist'] == False:  # 加载参与方用户列表
            self.participants_list = self.params['participants_namelist']
        else:
            self.participants_list = list(range(self.params['number_of_total_participants']))
        # random.shuffle(self.participants_list)
        self.benign_namelist = list(set(self.participants_list) - set(self.advasarial_namelist))

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       indices), pin_memory=True, num_workers=8)
        return train_loader

    def get_train_old(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param params:
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(len(self.train_dataset) / self.params['number_of_total_participants'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       sub_indices))
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=False)
        return test_loader

    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

    def get_poison_batch(self, bptt, noise_trigger, evaluation=False):

        images, targets = bptt

        poison_count = 0
        new_images = images
        new_targets = targets

        # 得到训练样本：每一个local worker对图像加global trigger
        for index in range(0, len(images)):
            if evaluation:  # poison all data when testing
                new_targets[index] = self.params['poison_label_swap']
                new_images[index] = self.add_pixel_pattern(images[index], noise_trigger)
                poison_count += 1

            else:  # poison part of data when training
                if index < self.params['poisoning_per_batch']:
                    new_targets[index] = self.params['poison_label_swap']
                    new_images[index] = self.add_pixel_pattern(images[index], noise_trigger)  # 对图片加噪声
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index] = targets[index]

        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images, new_targets, poison_count

    # 对图片进行投毒操作，加噪声
    def add_pixel_pattern(self, ori_image, noise_trigger):
        image = copy.deepcopy(ori_image)
        noise = torch.tensor(noise_trigger).cpu()
        poison_patterns = []
        poison_patterns = poison_patterns + self.params['sum_poison_pattern']
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = noise[0][pos[0]][pos[1]]  # +delta i  #？
            image[1][pos[0]][pos[1]] = noise[1][pos[0]][pos[1]]  # +delta i
            image[2][pos[0]][pos[1]] = noise[2][pos[0]][pos[1]]  # +delta i

        image = torch.clamp(image, -1, 1)

        return image


if __name__ == '__main__':
    np.random.seed(1)
    with open(f'./utils/cifar_params.yaml', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    helper = Cifar100Helper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'mnist'))
    helper.load_data()

    pars = list(range(100))
    # show the data distribution among all participants.
    count_all = 0
    for par in pars:
        cifar_class_count = dict()
        for i in range(10):
            cifar_class_count[i] = 0
        count = 0
        _, data_iterator = helper.train_data[par]
        for batch_id, batch in enumerate(data_iterator):
            data, targets = batch
            for t in targets:
                cifar_class_count[t.item()] += 1
            count += len(targets)
        count_all += count
        print(par, cifar_class_count, count, max(zip(cifar_class_count.values(), cifar_class_count.keys())))

    print('avg', count_all * 1.0 / 100)
