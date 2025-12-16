"""日志记录与 TensorBoard 工具

封装训练过程中的文本日志输出与 TensorBoard 标量记录，
统一写入每个实验对应的 logs/<exp_name>/ 目录。
"""

import os
import logging
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        
        # 1. 设置 Python Logging (输出到控制台和 txt 文件)
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        
        # File Handler
        fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Stream Handler (Console)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        
        # 2. 设置 TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)

    def info(self, msg):
        self.logger.info(msg)

    def log_metrics(self, metrics_dict, step, mode='train'):
        """
        metrics_dict: {'loss': 0.5, 'cc': 0.8}
        mode: 'train' or 'val'
        """
        for k, v in metrics_dict.items():
            self.writer.add_scalar(f"{mode}/{k}", v, step)

    def log_image(self, tag, img_tensor, step):
        """Log a single image or grid to TensorBoard."""
        self.writer.add_image(tag, img_tensor, step)

    def log_histogram(self, tag, values, step, bins=30):
        """Log histogram for monitoring distribution (e.g., logits/probabilities)."""
        self.writer.add_histogram(tag, values, step, bins=bins)

    def close(self):
        self.writer.close()