import os, torch
import torch.nn as nn


class configs(object):
    def __init__(self):
        self.T = 10
        self.dt = 0.004

        self.epoch = 0
        self.num_epochs = 100
        self.batch_size = 128
        self.batch_size_test = 128

        #  None  'kaiming'  'xavier'
        self.init_method = None

        # input
        self.ds = 1
        self.in_channels = 2
        self.im_width, self.im_height = (128 // self.ds, 128 // self.ds)

        # output
        self.target_size = 20

        # Data
        self.clip = 1
        self.is_train_Enhanced = False
        self.is_spike = False
        self.interval_scaling = False

        # network
        self.beta = 0
        self.alpha = 0.3
        self.Vreset = 0
        self.Vthres = 0.3
        self.reduction = 5
        self.T_extend_Conv = False
        self.T_extend_BN = False
        self.h_conv = False
        self.mem_act = torch.relu
        self.mode_select = 'spike'
        self.TR_model = 'NTR'

        # BatchNorm
        self.track_running_stats = True

        self.a = 0.25
        self.lens = self.a / 2

        self.lr = 1e-3
        self.betas = [0.9, 0.999]
        self.eps = 1e-8
        self.weight_decay = 0

        self.lr_scheduler = True
        self.lr_scheduler_epoch = 25

        self.name = None
        self.modelPath = os.path.dirname(
            os.path.abspath(__file__)) + os.sep + 'Result'
        self.modelNames = None
        self.recordPath = self.modelPath

        self.recordNames = None

        # dataset
        self.savePath = os.path.dirname(os.path.dirname(__file__)) + os.sep + 'data'

        # Datasets
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None

        # Dataloader
        self.drop_last = False
        self.pip_memory = False
        self.num_work = 0

        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.device = None
        self.device_ids = None

        self.best_acc = 0
        self.best_epoch = 0

        self.epoch_list = []
        self.loss_train_list = []
        self.loss_test_list = []
        self.acc_train_list = []
        self.acc_test_list = []

        self.train_loss = 0
        self.train_correct = 0
        self.train_acc = 0

        self.test_loss = 0
        self.test_correct = 0
        self.test_acc = 0

        self.state = None

    def __str__(self):
        for item in self.__dict__:
            print(item + "==" + str(self.__dict__[item]))
        return ""


if __name__ == '__main__':
    config = configs()
    print(config)
