#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
import argparse
import datetime
import torch
from thop import profile  # 用于计算FLOPs

# ===== 关键修改：确保优先加载本地 ultralytics 代码 =====
LOCAL_YOLO_PATH = Path("E:/TBH/ultralytics-main/ultralytics-main")
sys.path.insert(0, str(LOCAL_YOLO_PATH))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info  # 修改了导入路径


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv8 Training Script')
    parser.add_argument('--data', type=str,
                        default='E:/TBH/ultralytics-main/ultralytics-main/ultralytics/cfg/datasets/dataset.yaml',
                        help='Path to dataset config file')
    #parser.add_argument('--model', type=str, default='ultralytics-main/ultralytics/cfg/models/v8/my_yolov8.yaml',
    parser.add_argument('--model', type=str, default='ultralytics-main/ultralytics/cfg/models/v8/yolov8m.yaml',
                        help='Path to model config file')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    parser.add_argument('--name', type=str, default=None, help='Experiment name (default: timestamp)')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision (AMP) training')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (e.g. cpu, cuda:0)')
    return parser.parse_args()


def setup_training_directory(project, name=None):
    """设置训练目录，确保每次训练都有独立目录"""
    os.makedirs(project, exist_ok=True)

    if name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"exp_{timestamp}"

    exp_dir = os.path.join(project, name)

    if os.path.exists(exp_dir):
        random_suffix = datetime.datetime.now().strftime("%f")
        name = f"{name}_{random_suffix}"
        exp_dir = os.path.join(project, name)

    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir, name


def calculate_metrics(model, input_size=(1, 3, 640, 640), device='cuda'):
    """计算模型的GFLOPs和Params"""
    model = model.to(device).eval()
    dummy_input = torch.randn(input_size).to(device)

    try:
        # 计算FLOPs和Params
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9

        # 计算FPS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 100
        timings = []

        # Warm-up
        for _ in range(10):
            _ = model(dummy_input)

        # 测量推理时间
        with torch.no_grad():
            for _ in range(repetitions):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))

        avg_time = sum(timings) / repetitions
        fps = 1000 / avg_time  # 转换为FPS

        return gflops, params, fps
    except Exception as e:
        print(f"无法计算模型指标: {str(e)}")
        return 0, 0, 0


def print_model_info(model, device):
    """打印模型信息"""
    print("\n===== 模型信息 ======")

    # 使用YOLO内置的info方法
    model.info(verbose=False)

    # 计算GFLOPs和Params
    try:
        gflops, params, fps = calculate_metrics(model.model, device=device)
        print(f"GFLOPs: {gflops:.2f}")
        print(f"Params: {params / 1e6:.2f}M")
        print(f"FPS: {fps:.2f} (on {device})")
    except Exception as e:
        print(f"无法获取完整模型信息: {str(e)}")

    print("====================\n")


def main():
    # 解析参数
    args = parse_arguments()

    # 设置训练目录
    exp_dir, exp_name = setup_training_directory(args.project, args.name)
    print(f"Training results will be saved in: {exp_dir}")

    # 加载模型
    model = YOLO(args.model).to(args.device)

    # 打印模型信息
    print_model_info(model, args.device)

    # 训练配置
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'lr0': args.lr0,
        'resume': args.resume,
        'project': args.project,
        'name': exp_name,
        'exist_ok': False,
        'verbose': True,
        'amp': args.amp,
        'device': args.device,
    }

    try:
        # 打印关键配置信息
        print("\n===== 训练配置 ======")
        print(f"Model: {args.model}")
        print(f"Data: {args.data}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch}")
        print(f"AMP enabled: {train_args['amp']}")
        print(f"Device: {args.device}")
        print(f"Experiment name: {exp_name}")
        print("====================\n")

        # 开始训练
        results = model.train(**train_args)

        # 训练完成后保存最终模型
        model.save(os.path.join(exp_dir, 'trained_model.pt'))
        print(f"Training completed successfully! Results saved to: {exp_dir}")

    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        if "expected scalar type Float but found Half" in str(e):
            print("\n💡 解决方案建议：")
            print("1. 禁用AMP训练：添加 --amp=False 参数")
            print("2. 修改MAMBA模块代码，确保其支持float16计算")
            print("3. 检查rs_mamba_cd.py中的LayerNorm操作是否兼容半精度")
        raise


if __name__ == '__main__':
    main()