import os
import argparse
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
import torchvision
from pytorch_fid import fid_score
from pyiqa import create_metric
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")

class IQADataset(Dataset):
    def __init__(self, ori_dir, sr_dir, file_list):
        self.ori_dir = ori_dir
        self.sr_dir = sr_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        ori_path = os.path.join(self.ori_dir, img_name)
        sr_path = os.path.join(self.sr_dir, img_name)

        ori_img = Image.open(ori_path).convert('RGB')
        sr_img = Image.open(sr_path).convert('RGB')

        ori_tensor = torch.from_numpy(np.array(ori_img)).float().permute(2, 0, 1) / 255.0
        sr_tensor = torch.from_numpy(np.array(sr_img)).float().permute(2, 0, 1) / 255.0

        return ori_tensor, sr_tensor, img_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_ori', type=str, required=True)
    parser.add_argument('--path_to_sr', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing (default: 8)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading (default: 4)')
    parser.add_argument('--all', type=str, default='false', help='Calculate all metrics, including NR metrics that cannot be batched')
    parser.add_argument('--fidonly', type=str, default='false', help='Only calculate FID')
    args = parser.parse_args()

    # 计算 FID
    fid_value = fid_score.calculate_fid_given_paths([args.path_to_ori, args.path_to_sr],
                                                    batch_size=args.batch_size,
                                                    device='cuda' if torch.cuda.is_available() else 'cpu',
                                                    dims=2048,
                                                    num_workers=args.num_workers)
    print(f"FID value: {fid_value}")
    # 释放FID计算占用的资源
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    if args.fidonly.lower() == 'true':
        return

    # 获取图像文件
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    ori_files = sorted([f for f in os.listdir(args.path_to_ori) 
                       if os.path.splitext(f)[1].lower() in image_extensions])
    sr_files = sorted([f for f in os.listdir(args.path_to_sr) 
                      if os.path.splitext(f)[1].lower() in image_extensions])
    
    common_files = sorted(list(set(ori_files) & set(sr_files)))
    print(f"Processing {len(common_files)} images...")

    if not common_files:
        print("No common image files found!")
        return

    # 构建数据集和 DataLoader
    dataset = IQADataset(args.path_to_ori, args.path_to_sr, common_files)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )

    # 初始化设备
    device = torch.device(args.device)

    # 获取图像总数和批次信息
    total_images = len(common_files)
    print(f"Total images: {total_images}")

    # 初始化结果列表
    results = []

    # 首先处理可批处理的指标（FR指标）
    fr_metrics = ['psnr', 'ssim', 'lpips', 'dists', 'clipiqa']

    for metric_name in fr_metrics:
        print(f"Processing {metric_name}...")
        metric_model = None

        try:
            # 为当前指标创建模型
            try:
                metric_model = create_metric(metric_name, device=device)
            except Exception as e:
                print(f"Warning: {metric_name} not available: {e}")
                # 为所有图像设置None值
                for i in range(total_images):
                    if i >= len(results):
                        results.append({'filename': common_files[i]})
                    results[i][metric_name] = None
                continue

            # 重新创建数据加载器以确保从开始迭代
            dataset = IQADataset(args.path_to_ori, args.path_to_sr, common_files)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True if args.device == 'cuda' else False
            )

            # 处理所有批次
            batch_idx = 0
            for ori_batch, sr_batch, filenames in tqdm(dataloader, desc=f"Calculating {metric_name}"):
                ori_batch = ori_batch.to(device, non_blocking=True)
                sr_batch = sr_batch.to(device, non_blocking=True)
                batch_size = ori_batch.shape[0]

                try:
                    scores = metric_model(sr_batch, ori_batch)
                    if isinstance(scores, torch.Tensor):
                        scores = scores.detach().cpu().numpy().flatten()

                    # 存储结果
                    for i in range(batch_size):
                        idx = batch_idx * args.batch_size + i
                        if idx >= total_images:
                            continue

                        if idx >= len(results):
                            results.append({'filename': filenames[i]})

                        results[idx][metric_name] = float(scores[i]) if not np.isnan(scores[i]) else None
                except Exception as e:
                    print(f"Error calculating {metric_name} for batch {batch_idx}: {e}")
                    for i in range(batch_size):
                        idx = batch_idx * args.batch_size + i
                        if idx >= total_images:
                            continue

                        if idx >= len(results):
                            results.append({'filename': filenames[i]})

                        results[idx][metric_name] = None

                batch_idx += 1
        finally:
            # 释放当前指标模型占用的资源
            if metric_model is not None:
                del metric_model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            print(f"  {metric_name} calculation completed, resources released.")

    # 处理慢速指标（NR指标，需--all控制）
    if args.all.lower() == 'true':
        nr_metrics = ['niqe', 'maniqa', 'musiq']
    else:
        nr_metrics = []

    for metric_name in nr_metrics:
        print(f"Processing {metric_name}...")
        metric_model = None

        try:
            # 为当前指标创建模型
            try:
                metric_model = create_metric(metric_name, device=device)
            except Exception as e:
                print(f"Warning: {metric_name} not available: {e}")
                # 为所有图像设置None值
                for i in range(total_images):
                    if i < len(results):
                        results[i][metric_name] = None
                    else:
                        results.append({'filename': common_files[i], metric_name: None})
                continue

            # 重新创建数据加载器
            dataset = IQADataset(args.path_to_ori, args.path_to_sr, common_files)
            dataloader = DataLoader(
                dataset,
                batch_size=1,  # NR指标通常逐个图像处理
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True if args.device == 'cuda' else False
            )

            # 处理所有图像
            img_idx = 0
            for _, sr_batch, filenames in tqdm(dataloader, desc=f"Calculating {metric_name}"):
                sr_batch = sr_batch.to(device, non_blocking=True)

                try:
                    score = metric_model(sr_batch)
                    if isinstance(score, torch.Tensor):
                        score = score.item()

                    # 存储结果
                    if img_idx < len(results):
                        results[img_idx][metric_name] = float(score)
                    else:
                        results.append({'filename': filenames[0], metric_name: float(score)})
                except Exception as e:
                    print(f"Error calculating {metric_name} for image {filenames[0]}: {e}")
                    if img_idx < len(results):
                        results[img_idx][metric_name] = None
                    else:
                        results.append({'filename': filenames[0], metric_name: None})

                img_idx += 1
        finally:
            # 释放当前指标模型占用的资源
            if metric_model is not None:
                del metric_model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            print(f"  {metric_name} calculation completed, resources released.")

    # 保存结果
    if results:
        df = pd.DataFrame(results)
        
        # 添加平均值行
        avg_dict = {'filename': 'AVERAGE'}
        for col in df.columns:
            if col != 'filename':
                valid_values = pd.to_numeric(df[col], errors='coerce').dropna()
                avg_dict[col] = valid_values.mean() if len(valid_values) > 0 else None
        
        df = pd.concat([df, pd.DataFrame([avg_dict])], ignore_index=True)
        df.to_csv(args.output, index=False)
        
        print(f"\nSaved results to {args.output}")
        print("\nAverage values:")
        for key, value in avg_dict.items():
            if key != 'filename':
                if value is not None:
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: N/A")
    else:
        print("No results to save!")


if __name__ == '__main__':
    main()