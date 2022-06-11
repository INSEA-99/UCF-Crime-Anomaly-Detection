import argparse
import os
from os import path

import torch
import torch.backends.cudnn as cudnn
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from features_loader import FeaturesLoader, FeaturesLoaderVal
from network.TorchUtils import TorchModel
from network.anomaly_detector_model import AnomalyDetector, custom_objective, original_objective, RegularizedLoss
from utils.callbacks import DefaultModelCallback, TensorBoardCallback
from utils.utils import register_logger, get_torch_device


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")

    # io
    parser.add_argument('--features_path', default='.\\features\\download_features\\download_r3d152_features',
                        help="추출된 feature가 저장되어 있는 경로")
    parser.add_argument('--annotation_path', default=".\\Train_Annotation.txt",
                        help="학습에 사용할 Dataset의 annotation이 적힌 txt 파일")
    parser.add_argument('--log_file', type=str, default="log.log",
                        help="set logging file.")
    parser.add_argument('--exps_dir', type=str, default=".\\exps\\Using_download_features\\r3d152_CustomLoss_top6_WAvg_val",
                        help="학습된 모델이 저장될 경로.")
    
    parser.add_argument('--checkpoint', type=str,
                        help="load a model for resume training")

    # optimization
    parser.add_argument('--batch_size', type=int, default=60,
                        help="batch size")
    parser.add_argument('--feature_dim', type=int, default=4096,
                        help="feature dimension")
    parser.add_argument('--save_every', type=int, default=1,
                        help="epochs interval for saving the model checkpoints")
    parser.add_argument('--lr_base', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--iterations_per_epoch', type=int, default=20000,
                        help="number of training iterations")
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of training epochs")
    parser.add_argument('--loss_type', type=str, default="original", 
                        choices=['original', 'custom'], help="number of training epochs")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # Register directories
    register_logger(log_file=args.log_file)
    os.makedirs(args.exps_dir, exist_ok=True)
    models_dir = path.join(args.exps_dir, 'models')
    tb_dir = path.join(args.exps_dir, 'tensorboard')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # Optimizations
    device = get_torch_device()
    cudnn.benchmark = True  # enable cudnn tune

    # Data loader
    train_loader = FeaturesLoader(features_path=args.features_path,
                                  feature_dim=args.feature_dim,
                                  annotation_path=args.annotation_path,
                                  iterations=args.iterations_per_epoch)
    # Model
    if args.checkpoint is not None and path.exists(args.checkpoint):
        model = TorchModel.load_model(args.checkpoint)
    else:
        network = AnomalyDetector(args.feature_dim)
        model = TorchModel(network)
   
   
    model = model.to(device).train()
    
    print(device)
   
    # Training parameters
    """
    In the original paper:
        lr = 0.01
        epsilon = 1e-8
    """
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr_base, eps=1e-8)
    
    if args.loss_type == "original":
        criterion = RegularizedLoss(network, original_objective).to(device)
    else:
        criterion = RegularizedLoss(network, custom_objective).to(device)


    # Callbacks
    tb_writer = SummaryWriter(log_dir=tb_dir) #  "/root/default/sw_capstone/exps.tensorboard"
    model.register_callback(DefaultModelCallback(visualization_dir=args.exps_dir)) # "/root/default/sw_capstone/exps"
    model.register_callback(TensorBoardCallback(tb_writer=tb_writer)) # SummaryWriter(log_dir=tb_dir) ->

    # Training
    model.fit(train_iter=train_loader,
              criterion=criterion,
              optimizer=optimizer,
              epochs=args.epochs,
              network_model_path_base=models_dir,
              save_every=args.save_every,
              evaluate_every=True)
