import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import colors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from PIL import Image
import warnings
import matplotlib
warnings.filterwarnings("ignore")

matplotlib.use('Agg')  # Use non-GUI backend

def create_gif(image_list, gif_name, duration, loop=0):
    images = [Image.open(image_name) for image_name in image_list]
    images[0].save(
        gif_name, save_all=True, append_images=images[1:], 
        duration=duration, loop=loop
    )

def visualize_results(result_dir, output_dir, sample_indices=None, time_indices=None, lower_threshold=None):
    os.makedirs(output_dir, exist_ok=True)
    batch_dirs = sorted([d for d in os.listdir(result_dir) if d.startswith('batch_')])

    mycolors_rain = ("#FFFFFF", "#83DF74", "#3DBA3D", "#61BBFF", "#0000FF", "#FA00FA", "#7A0036")
    cmap_rain = colors.ListedColormap(mycolors_rain)
    bounds_rain = [0, 0.1, 2, 4, 8, 20, 50, 200]
    norm_rain = colors.BoundaryNorm(bounds_rain, cmap_rain.N)

    try:
        shandong = cfeature.ShapelyFeature(
            Reader('/home/jianhao/methods/single-source/data/山东省_省/山东省_省.shp').geometries(),
            ccrs.PlateCarree(), edgecolor='k', facecolor='none')
        shandong2 = cfeature.ShapelyFeature(
            Reader('/home/jianhao/methods/single-source/data/山东省_市/山东省_市.shp').geometries(),
            ccrs.PlateCarree(), edgecolor='k', facecolor='none')
        use_cartopy = True
    except Exception:
        print("Warning: Unable to load shapefiles, using basic visualization")
        use_cartopy = False

    for batch_dir in batch_dirs:
        batch_path = os.path.join(result_dir, batch_dir)
        sample_dirs = sorted([d for d in os.listdir(batch_path) if d.startswith('sample_')])
        if sample_indices is not None:
            sample_dirs = [f'sample_{i}' for i in sample_indices if f'sample_{i}' in sample_dirs]

        for sample_dir in sample_dirs:
            sample_path = os.path.join(batch_path, sample_dir)
            # 按照文件名中的数字进行排序，而不是字母序
            pred_files = [f for f in os.listdir(sample_path) if f.startswith('pred_')]
            pred_files = sorted(pred_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            if time_indices is not None:
                pred_files = [f'pred_{i}.npy' for i in time_indices if f'pred_{i}.npy' in pred_files]

            # 读取时间信息
            time_info_file = os.path.join(sample_path, "time_info.json")
            target_times = None
            if os.path.exists(time_info_file):
                import json
                with open(time_info_file, 'r') as f:
                    time_info = json.load(f)
                    target_times = time_info.get('target_times', [])
                    # 调试信息
                    print(f"Debug: {sample_dir}, target_times length: {len(target_times)}")
                    if len(target_times) < 12:
                        print(f"Debug: target_times content: {target_times}")
            else:
                print(f"Debug: time_info.json not found in {sample_path}")

            gt_frames, pred_frames = [], []
            for i, pred_file in enumerate(pred_files):
                time_idx = int(pred_file.split('_')[1].split('.')[0])
                gt_file = f'gt_{time_idx}.npy'
                pred = np.load(os.path.join(sample_path, pred_file))
                gt = np.load(os.path.join(sample_path, gt_file))

                gt_frames.append(gt)
                pred_enhanced = pred.copy()
                if lower_threshold is not None:
                    pred_enhanced[pred_enhanced < lower_threshold] = 0
                pred_frames.append(pred_enhanced)

                pred = pred_enhanced
                gt = gt

                # 确定时间标题
                if target_times and i < len(target_times):
                    # 解析具体时间
                    time_str = target_times[i]  # 格式: "20230101_072"
                    date_part = time_str.split('_')[0]  # "20230101"
                    time_idx_part = int(time_str.split('_')[1])  # 72
                    # 将时间索引转换为具体时间
                    hours = time_idx_part * 10 // 60
                    minutes = time_idx_part * 10 % 60
                    time_title = f'{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {hours:02d}:{minutes:02d}'
                elif target_times and len(target_times) > 0:
                    # 如果索引超出范围，但有时间信息，尝试推算
                    print(f"Debug: Index {i} out of range, target_times length={len(target_times)}, trying to extrapolate")
                    base_time_str = target_times[0]
                    date_part = base_time_str.split('_')[0]
                    base_time_idx = int(base_time_str.split('_')[1])
                    # 推算当前时间点
                    current_time_idx = base_time_idx + i
                    hours = current_time_idx * 10 // 60
                    minutes = current_time_idx * 10 % 60
                    time_title = f'{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {hours:02d}:{minutes:02d}'
                else:
                    # 备用方案：使用相对时间
                    print(f"Debug: Using fallback time for i={i}, target_times length={len(target_times) if target_times else 0}")
                    time_title = f't+{time_idx*10}min'

                fig = plt.figure(figsize=(16, 7))
                if use_cartopy:
                    extent = [114, 123, 34, 39]
                    lon_labels = np.arange(114, 124, 2)
                    lat_labels = np.arange(34, 40, 1)
                    ax1 = plt.subplot(121, projection=ccrs.PlateCarree())
                    ax1.imshow(gt, cmap=cmap_rain, norm=norm_rain, aspect='auto', extent=extent)
                    ax1.add_feature(shandong, lw=0.5)
                    ax1.add_feature(shandong2, lw=0.8)
                    ax1.set_xticks(lon_labels)
                    ax1.set_yticks(lat_labels)
                    ax1.set_title(f'Ground Truth ({time_title})')
                    ax1.set_xlabel('Longitude')
                    ax1.set_ylabel('Latitude')

                    ax2 = plt.subplot(122, projection=ccrs.PlateCarree())
                    img2 = ax2.imshow(pred, cmap=cmap_rain, norm=norm_rain, aspect='auto', extent=extent)
                    ax2.add_feature(shandong, lw=0.5)
                    ax2.add_feature(shandong2, lw=0.8)
                    ax2.set_xticks(lon_labels)
                    ax2.set_yticks(lat_labels)
                    ax2.set_title(f'Prediction ({time_title})')
                    ax2.set_xlabel('Longitude')

                    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                    cbar = fig.colorbar(img2, cax=cbar_ax)
                    cbar.set_label('Precipitation (mm/h)')
                    cbar.set_ticks(bounds_rain)
                else:
                    ax1 = plt.subplot(121)
                    ax1.imshow(gt, cmap=cmap_rain, norm=norm_rain)
                    ax1.set_title(f'Ground Truth ({time_title})')
                    ax1.axis('off')
                    ax2 = plt.subplot(122)
                    img2 = ax2.imshow(pred, cmap=cmap_rain, norm=norm_rain)
                    ax2.set_title(f'Prediction ({time_title})')
                    ax2.axis('off')
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                    cbar = fig.colorbar(img2, cax=cbar_ax)
                    cbar.set_label('Precipitation (mm/h)')
                    cbar.set_ticks(bounds_rain)

                plt.tight_layout(rect=[0, 0, 0.9, 1])
                output_subdir = os.path.join(output_dir, batch_dir, sample_dir)
                os.makedirs(output_subdir, exist_ok=True)
                plt.savefig(os.path.join(output_subdir, f'comparison_{time_idx}.png'), dpi=300, bbox_inches='tight')
                plt.close()

                print(f'Saved {batch_dir}/{sample_dir}/comparison_{time_idx}.png')

            if len(gt_frames) > 0 and len(pred_frames) > 0:
                create_stamp_montage(gt_frames, pred_frames, output_dir, batch_dir, sample_dir, norm_rain, cmap_rain, target_times)

def create_stamp_montage(gt_frames, pred_frames, output_dir, batch_dir, sample_dir, norm, cmap, target_times=None):
    output_subdir = os.path.join(output_dir, batch_dir, sample_dir)
    os.makedirs(output_subdir, exist_ok=True)
    gt_frames = [frame for frame in gt_frames]
    pred_frames = [frame for frame in pred_frames]

    fig_gt, axes_gt = plt.subplots(3, 4, figsize=(16, 10))
    fig_pred, axes_pred = plt.subplots(3, 4, figsize=(16, 10))
    axes_gt = axes_gt.flatten()
    axes_pred = axes_pred.flatten()

    for i in range(min(12, len(gt_frames))):
        # 确定时间标题
        if target_times and i < len(target_times):
            time_str = target_times[i]
            date_part = time_str.split('_')[0]
            time_idx_part = int(time_str.split('_')[1])
            hours = time_idx_part * 10 // 60
            minutes = time_idx_part * 10 % 60
            time_title = f'{hours:02d}:{minutes:02d}'
        else:
            time_title = f't+{i*10}min'
        
        axes_gt[i].imshow(gt_frames[i], cmap=cmap, norm=norm)
        axes_gt[i].set_title(time_title)
        axes_gt[i].axis('off')

        axes_pred[i].imshow(pred_frames[i], cmap=cmap, norm=norm)
        axes_pred[i].set_title(time_title)
        axes_pred[i].axis('off')

    fig_gt.subplots_adjust(right=0.9)
    fig_pred.subplots_adjust(right=0.9)
    im_gt = axes_gt[0].imshow(gt_frames[0], cmap=cmap, norm=norm)
    im_pred = axes_pred[0].imshow(pred_frames[0], cmap=cmap, norm=norm)
    fig_gt.colorbar(im_gt, ax=axes_gt, fraction=0.015, pad=0.04).set_label('Precipitation (mm/h)')
    fig_pred.colorbar(im_pred, ax=axes_pred, fraction=0.015, pad=0.04).set_label('Precipitation (mm/h)')
    fig_gt.suptitle('Ground Truth - 12 frames', fontsize=16)
    fig_pred.suptitle('Prediction - 12 frames', fontsize=16)
    fig_gt.savefig(os.path.join(output_subdir, 'gt_stamp_montage.png'), dpi=300, bbox_inches='tight')
    fig_pred.savefig(os.path.join(output_subdir, 'pred_stamp_montage.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_gt)
    plt.close(fig_pred)

    def save_horizontal_concat(frames, filename):
        fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4))
        if len(frames) == 1:
            axes = [axes]
        for i, (ax, frame) in enumerate(zip(axes, frames)):
            ax.imshow(frame, cmap=cmap, norm=norm)
            # 使用相同的时间标题逻辑
            if target_times and i < len(target_times):
                time_str = target_times[i]
                date_part = time_str.split('_')[0]
                time_idx_part = int(time_str.split('_')[1])
                hours = time_idx_part * 10 // 60
                minutes = time_idx_part * 10 % 60
                time_title = f'{hours:02d}:{minutes:02d}'
            else:
                time_title = f't+{i*10}min'
            ax.set_title(time_title)
            ax.axis('off')
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        im = axes[-1].imshow(frames[-1], cmap=cmap, norm=norm)
        fig.colorbar(im, cax=cbar_ax).set_label('Precipitation (mm/h)')
        fig.savefig(os.path.join(output_subdir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)

    save_horizontal_concat(gt_frames[:12], 'gt_horizontal_concat.png')
    save_horizontal_concat(pred_frames[:12], 'pred_horizontal_concat.png')

    def save_gif(frames, gif_path):
        temp_images = []
        for i, frame in enumerate(frames[:12]):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(frame, cmap=cmap, norm=norm)
            ax.axis('off')
            # 使用相同的时间标题逻辑
            if target_times and i < len(target_times):
                time_str = target_times[i]
                date_part = time_str.split('_')[0]
                time_idx_part = int(time_str.split('_')[1])
                hours = time_idx_part * 10 // 60
                minutes = time_idx_part * 10 % 60
                time_title = f'{hours:02d}:{minutes:02d}'
            else:
                time_title = f't+{i*10}min'
            ax.set_title(time_title)
            tmp_path = os.path.join(output_subdir, f'_tmp_frame_{i}.png')
            plt.savefig(tmp_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            temp_images.append(tmp_path)
        create_gif(temp_images, gif_path, duration=500)
        for path in temp_images:
            os.remove(path)

    save_gif(gt_frames, os.path.join(output_subdir, 'gt_animation.gif'))
    save_gif(pred_frames, os.path.join(output_subdir, 'pred_animation.gif'))
    
    # 生成左右对比的GIF动图
    def save_comparison_gif(gt_frames, pred_frames, gif_path):
        temp_images = []
        for i, (gt_frame, pred_frame) in enumerate(zip(gt_frames[:12], pred_frames[:12])):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            
            # 左侧显示Ground Truth
            ax1.imshow(gt_frame, cmap=cmap, norm=norm)
            ax1.axis('off')
            ax1.set_title('Ground Truth')
            
            # 右侧显示Prediction
            ax2.imshow(pred_frame, cmap=cmap, norm=norm)
            ax2.axis('off')
            ax2.set_title('Prediction')
            
            # 使用相同的时间标题逻辑
            if target_times and i < len(target_times):
                time_str = target_times[i]
                date_part = time_str.split('_')[0]
                time_idx_part = int(time_str.split('_')[1])
                hours = time_idx_part * 10 // 60
                minutes = time_idx_part * 10 % 60
                time_title = f'{hours:02d}:{minutes:02d}'
            else:
                time_title = f't+{i*10}min'
            
            # 添加整体标题
            fig.suptitle(f'Comparison at {time_title}', fontsize=14, y=0.95)
            
            # 添加颜色条
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            im = ax2.imshow(pred_frame, cmap=cmap, norm=norm)
            fig.colorbar(im, cax=cbar_ax).set_label('Precipitation (mm/h)')
            
            tmp_path = os.path.join(output_subdir, f'_tmp_comparison_{i}.png')
            plt.savefig(tmp_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            temp_images.append(tmp_path)
        
        create_gif(temp_images, gif_path, duration=500)
        for path in temp_images:
            os.remove(path)
    
    save_comparison_gif(gt_frames, pred_frames, os.path.join(output_subdir, 'comparison_animation.gif'))
    
    print(f"Saved horizontal concatenation and GIFs for {batch_dir}/{sample_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize prediction results')
    parser.add_argument('--result_dir', type=str, default='./pred_results')
    parser.add_argument('--output_dir', type=str, default='./visualization')
    parser.add_argument('--sample_indices', type=str, default=None)
    parser.add_argument('--time_indices', type=str, default=None)
    parser.add_argument('--no_cartopy', action='store_true')
    parser.add_argument('--lower_threshold', type=float, default=0.005)
    args = parser.parse_args()

    sample_indices = [int(i) for i in args.sample_indices.split(',')] if args.sample_indices else None
    time_indices = [int(i) for i in args.time_indices.split(',')] if args.time_indices else None

    visualize_results(args.result_dir, args.output_dir, sample_indices, time_indices, args.lower_threshold)

