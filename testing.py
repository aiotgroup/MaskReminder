import logging
import os

import scipy.io as scio
import torch
from torch.utils.data.dataloader import DataLoader

import utils as init_utils
from data_process import SensorDataset
from pipeline import Tester

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    basic_config = init_utils.init_configs()
    os.environ["CUDA_VISIBLE_DEVICES"] = basic_config.gpu_device

    left_test_mat = scio.loadmat(os.path.join(basic_config.dataset_path,
                                              '%s-%s-%d' %
                                              (basic_config.preprocess_method,
                                               basic_config.preprocess_strategy,
                                               basic_config.seq_len),
                                              'left_test.mat'))
    right_test_mat = scio.loadmat(os.path.join(basic_config.dataset_path,
                                               '%s-%s-%d' %
                                               (basic_config.preprocess_method,
                                                basic_config.preprocess_strategy,
                                                basic_config.seq_len),
                                               'right_test.mat'))
    left_test_dataset = SensorDataset(left_test_mat)
    right_test_dataset = SensorDataset(right_test_mat)

    strategy = init_utils.init_strategy(basic_config)
    strategy.load_state_dict(torch.load(basic_config.model_path))

    tester = Tester(strategy,
                    eval_data_loader=DataLoader(left_test_dataset, batch_size=basic_config.test_batch_size, shuffle=False),
                    n_classes=basic_config.n_classes,
                    output_path=basic_config.check_point_path,
                    hand="left",
                    use_gpu=True)
    tester.testing()

    tester = Tester(strategy,
                    eval_data_loader=DataLoader(right_test_dataset, batch_size=basic_config.test_batch_size, shuffle=False),
                    n_classes=basic_config.n_classes,
                    output_path=basic_config.check_point_path,
                    hand="right",
                    use_gpu=True)
    tester.testing()
