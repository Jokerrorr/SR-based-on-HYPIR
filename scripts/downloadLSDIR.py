from huggingface_hub import hf_hub_download, list_repo_files
import polars as pl
import os
from tqdm import tqdm
from PIL import Image

def download_all_images(save_dir="LSDIR_images"):
    """下载LSDIR数据集"""
    
    files = list_repo_files(
        repo_id="danjacobellis/LSDIR",
        repo_type="dataset"
    )
    
    parquet_files = [f for f in files if f.endswith('.parquet') and 'train' in f][:50]

    df = None
    for parquet_file in tqdm(parquet_files, desc="处理parquet文件"):
        print("正在处理：" + parquet_file.split('/')[-1])

        # 下载parquet文件
        local_path = hf_hub_download(
                repo_id="danjacobellis/LSDIR",
                filename=parquet_file,
                repo_type="dataset"
            )
        df_LSDIR = pl.read_parquet(local_path)

        # 创建并保存符合HYPIR要求的parquet文件
        if df is None:
            df = pl.from_dict({
                "image_path": ["data/" + path for path in df_LSDIR["path"]],
                "prompt": [""] * len(df_LSDIR["path"]),
            })
        else:
            df = pl.concat([
                df,
                pl.from_dict({
                    "image_path": ["data/" + path for path in df_LSDIR["path"]],
                    "prompt": [""] * len(df_LSDIR["path"]),
                })
            ], how="vertical")
        
        # 保存图片
        for i in tqdm(range(len(df_LSDIR["path"])), desc="保存图片"):
            img_dir = 'data/train/' + df_LSDIR["path"][i]
            if not os.path.exists(img_dir):
                os.makedirs(os.path.dirname(img_dir), exist_ok=True)
                with open(img_dir, 'wb') as f:
                    f.write(df_LSDIR["image"][i]['bytes'])

    df.write_parquet("data/train/LSDIR/lsdir_hypir.parquet")

if __name__ == "__main__":
    download_all_images()