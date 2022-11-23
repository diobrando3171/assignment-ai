import argparse
import os
from utils import util
import torch

basic_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class BaseConfig:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument("--basic_dir", type=str, default=basic_dir)
        parser.add_argument(
            "--is_train", type=bool, default=True, help="train or test"
        )
        # parser.add_argument(
        #     "--gan_mode", type=str, default="hinge", help="(ls|original|hinge)"
        # )
        parser.add_argument(
            "--gt_root",
            type=str,
            default="/Users/dio/Desktop/assignment/assignment-ai",
            help="path to detail images",
        )
        parser.add_argument(
            "--log_dir", type=str, default="logs", help="the path to record log"
        )
        parser.add_argument("--batchSize", type=int,
                            default=5, help="input batch size")
        parser.add_argument(
            "--name",
            type=str,
            default="VGG-Training",
            help="name of the experiment. It decides where to store samples and models",
        )
        parser.add_argument(
            "--train_image_size",
            type=int,
            default=224,
            help="image size of training process",
        )
        parser.add_argument(
            "--num_epochs",
            type=int,
            default=15,
            help="image size of training process",
        )
        parser.add_argument(
            "--print_freq",
            type=int,
            default=10,
            help="frequency of showing training results on console",
        )
        
        parser.add_argument(
            # set gup_ids to -1 for local cpu training
            "--gpu_ids", type=str, default="-1", help="gpu ids"
        )
        parser.add_argument(
            "--model",
            type=str,
            default="VGG",
            help="select the type of model VGG or CNN",
        )
        
        parser.add_argument(
            "--checkpoints_dir",
            type=str,
            default="checkpoints",
            help="models and logs are saved here",
        )
        
        parser.add_argument(
            "--use_dropout", action="store_true", help="use dropout for the generator"
        )
        
        self.initialized = True
        return parser

    def gather_config(self):
        # initialize parser with basic cfgions
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def print_config(self, cfg):
        message = ""
        message += "----------------- Config ---------------\n"
        for k, v in sorted(vars(cfg).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = os.path.join(basic_dir, cfg.checkpoints_dir, cfg.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "cfg.txt")
        with open(file_name, "wt") as cfg_file:
            cfg_file.write(message)
            cfg_file.write("\n")

    def create_config(self):
        cfg = self.gather_config()

        self.print_config(cfg)
        # set gpu ids
        str_ids = cfg.gpu_ids.split(",")
        cfg.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                cfg.gpu_ids.append(id)
        if len(cfg.gpu_ids) > 0:
            torch.cuda.set_device(cfg.gpu_ids[0])

        self.cfg = cfg
        return self.cfg
