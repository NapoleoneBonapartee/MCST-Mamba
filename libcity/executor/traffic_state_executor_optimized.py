import os
import time
import numpy as np
import random
import torch
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from libcity.executor.traffic_state_executor import TrafficStateExecutor
from libcity.utils import get_evaluator, ensure_dir
from libcity.model import loss
from functools import partial




class TrafficStateExecutorOptimized(TrafficStateExecutor):
    """Optimized TrafficStateExecutor with gradient accumulation and mixed precision"""
    
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)


    def save_model_with_epoch(self, epoch):
        """
        Save model with additional scaler state if using FP16
        """
        ensure_dir(self.cache_dir)
        config = dict()
        config['model_state_dict'] = self.model.state_dict()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['epoch'] = epoch

        # Store training state
        if self.lr_scheduler is not None:
            config['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        if hasattr(self, 'best_epoch'):
            config['best_epoch'] = self.best_epoch
        if hasattr(self, 'min_val_loss'):
            config['min_val_loss'] = self.min_val_loss
        if hasattr(self, 'wait'):
            config['wait'] = self.wait
        
        # Store gradient scaler state if using FP16
        fp16 = self.config.get('fp16', False)
        if fp16 and hasattr(self, 'scaler') and self.scaler is not None:
            config['scaler_state_dict'] = self.scaler.state_dict()
        
        # Store random state
        config['torch_rng_state'] = torch.get_rng_state()
        if torch.cuda.is_available():
            config['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        config['numpy_rng_state'] = np.random.get_state()
        config['random_rng_state'] = random.getstate()

        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        torch.save(config, model_path)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path


    def load_model_with_epoch(self, epoch):
        """
        Load model with gradient scaler state if using FP16
        """
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location='cpu')

        # Load model and optimizer
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            # 处理 _orig_mod. 前缀问题（通常由 torch.compile() 导致）
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                # 去除 _orig_mod. 前缀
                if k.startswith('_orig_mod.'):
                    new_key = k[10:]  # len('_orig_mod.') == 10
                else:
                    new_key = k
                new_state_dict[new_key] = v
            self.model.load_state_dict(new_state_dict)
            self._logger.info("已自动转换带 _orig_mod. 前缀的 state_dict")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore gradient scaler state if using FP16
        fp16 = self.config.get('fp16', False)
        if fp16 and hasattr(self, 'scaler') and self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self._logger.info("已恢复梯度缩放器状态")

        # Restore learning rate
        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            self._logger.info("已恢复学习率调度器状态 (当前学习率: {:.6f})".format(
                self.optimizer.param_groups[0]['lr']))
        # Restore training state
        if 'best_epoch' in checkpoint:
            self.best_epoch = checkpoint['best_epoch']
            self._logger.info("已恢复最佳epoch记录: {}".format(self.best_epoch))
        if 'min_val_loss' in checkpoint:
            self.min_val_loss = checkpoint['min_val_loss']
            self._logger.info("已恢复最小验证损失: {:.4f}".format(self.min_val_loss))
        if 'wait' in checkpoint:
            self.wait = checkpoint['wait']
            self._logger.info("已恢复early stopping等待计数: {}".format(self.wait))
        # Restore random state
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        
        self._logger.info("Loaded model at {}".format(epoch))

