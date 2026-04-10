"""
训练并评估单一模型的脚本（优化版本）
通过覆盖关键模块来实现优化
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from libcity.pipeline import run_model
from libcity.utils import str2bool, add_general_args


def apply_optimizations():
    """Apply optimizations by monkey-patching"""
    # Import original modules
    from libcity import data
    from libcity.data import utils as data_utils
    
    # Import optimized modules
    from libcity.data import utils_optimized
    
    # Monkey-patch the data utils
    data_utils.generate_dataloader = utils_optimized.generate_dataloader
    data_utils.generate_dataloader_pad = utils_optimized.generate_dataloader_pad
    
    print("[优化] 已启用优化的 DataLoader (移除 deepcopy, 启用 persistent_workers, prefetch)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 增加指定的参数
    parser.add_argument('--task', type=str,
                        default='traffic_state_pred', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='GRU', help='the name of model')
    parser.add_argument('--dataset', type=str,
                        default='METR_LA', help='the name of dataset')
    parser.add_argument('--config_file', type=str,
                        default=None, help='the file name of config file')
    parser.add_argument('--saved_model', type=str2bool,
                        default=True, help='whether save the trained model')
    parser.add_argument('--train', type=str2bool, default=True,
                        help='whether re-train model if the model is trained before')
    parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # 添加接续训练参数
    parser.add_argument('--resume', type=str, default=None,
                        help='resume training')
    
    # 添加优化选项
    parser.add_argument('--optimized', type=str2bool, default=True,
                        help='whether to use optimized version')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='gradient accumulation steps for larger effective batch size')
    parser.add_argument('--fp16', type=str2bool, default=True,
                        help='whether to use mixed precision training (FP16)')

    # 增加其他可选的参数
    add_general_args(parser)
    # 解析参数
    args = parser.parse_args()
    
    # Apply optimizations if requested
    if args.optimized:
        apply_optimizations()
    
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
        val is not None}
    
    run_model(task=args.task, model_name=args.model, dataset_name=args.dataset,
              config_file=args.config_file, saved_model=args.saved_model,
              train=args.train, other_args=other_args)
