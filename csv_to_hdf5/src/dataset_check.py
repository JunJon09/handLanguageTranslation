import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_dataset(dataset_path, output_dir="dataset_analysis"):
    """
    HDF5データセットを分析し、問題点を検出するプログラム
    
    Parameters:
    - dataset_path: HDF5ファイルが含まれるディレクトリパス
    - output_dir: 分析結果を保存するディレクトリ
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 結果保存用の辞書
    results = defaultdict(list)
    problematic_files = []
    
    # データセット内の各ファイルをチェック
    for filename in sorted(os.listdir(dataset_path)):
        if not filename.endswith('.hdf5'):
            continue
            
        file_path = os.path.join(dataset_path, filename)
        print(f"Analyzing {filename}...")
        
        try:
            with h5py.File(file_path, 'r') as f:
                # ファイル内のデータセットを調査
                file_has_problem = False
                file_stats = {}
                
                # 各データセットのキーを取得
                for key in f.keys():
                    data = f[key][()]
                    
                    # 基本統計情報の計算
                    stats = {
                        'key': key,
                        'shape': data.shape,
                        'min': np.min(data) if not np.any(np.isnan(data)) else "Contains NaN",
                        'max': np.max(data) if not np.any(np.isnan(data)) else "Contains NaN",
                        'mean': np.mean(data) if not np.any(np.isnan(data)) else "Contains NaN",
                        'std': np.std(data) if not np.any(np.isnan(data)) else "Contains NaN",
                        'has_nan': np.any(np.isnan(data)),
                        'has_inf': np.any(np.isinf(data)),
                        'zeros_percent': 100 * np.sum(data == 0) / data.size if data.size > 0 else 0,
                        'extreme_values_count': np.sum(np.abs(data) > 1000) if not np.any(np.isnan(data)) else "N/A"
                    }
                    
                    # 問題の検出
                    if stats['has_nan'] or stats['has_inf']:
                        file_has_problem = True
                        if stats['has_nan']:
                            print(f"  WARNING: NaN values detected in {key}")
                        if stats['has_inf']:
                            print(f"  WARNING: Infinite values detected in {key}")
                    
                    # 極端な値や偏った分布をチェック
                    if not np.any(np.isnan(data)) and not np.any(np.isinf(data)):
                        if np.abs(stats['std']) < 1e-7:
                            print(f"  WARNING: Very small standard deviation in {key}: {stats['std']}")
                            file_has_problem = True
                        
                        if stats['extreme_values_count'] > 0:
                            print(f"  WARNING: {stats['extreme_values_count']} extreme values in {key}")
                            file_has_problem = True
                            
                        if stats['zeros_percent'] > 95:
                            print(f"  WARNING: {stats['zeros_percent']:.2f}% zeros in {key}")
                            file_has_problem = True
                        
                        # データの分布を可視化
                        plt.figure(figsize=(10, 6))
                        plt.hist(data.flatten(), bins=50)
                        plt.title(f"Distribution of {filename} - {key}")
                        plt.xlabel("Value")
                        plt.ylabel("Count")
                        plt.savefig(os.path.join(output_dir, f"{filename}_{key}_hist.png"))
                        plt.close()
                    
                    file_stats[key] = stats
                
                # 問題のあるファイルを記録
                if file_has_problem:
                    problematic_files.append({
                        'filename': filename,
                        'stats': file_stats
                    })
                
                # 結果を収集
                for key, stats in file_stats.items():
                    for stat_name, stat_value in stats.items():
                        if stat_name != 'key' and stat_name != 'shape':
                            results[f"{key}_{stat_name}"].append(stat_value)
        
        except Exception as e:
            print(f"Error analyzing {filename}: {str(e)}")
            problematic_files.append({
                'filename': filename,
                'error': str(e)
            })
    
    # 分析サマリーレポートの生成
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
        f.write("Dataset Analysis Report\n")
        f.write("=====================\n\n")
        
        f.write(f"Total files analyzed: {len(os.listdir(dataset_path))}\n")
        f.write(f"Problematic files detected: {len(problematic_files)}\n\n")
        
        for i, prob_file in enumerate(problematic_files):
            f.write(f"Problem #{i+1}: {prob_file['filename']}\n")
            if 'error' in prob_file:
                f.write(f"  Error: {prob_file['error']}\n")
            else:
                for key, stats in prob_file['stats'].items():
                    if stats['has_nan'] or stats['has_inf'] or (
                        isinstance(stats['extreme_values_count'], int) and stats['extreme_values_count'] > 0
                    ) or (
                        isinstance(stats['zeros_percent'], float) and stats['zeros_percent'] > 95
                    ):
                        f.write(f"  Dataset: {key}\n")
                        for stat_name, stat_value in stats.items():
                            f.write(f"    {stat_name}: {stat_value}\n")
            f.write("\n")
    
    # NANを引き起こす可能性のある具体的な修正提案
    with open(os.path.join(output_dir, 'recommendations.txt'), 'w') as f:
        f.write("Recommendations for fixing dataset issues\n")
        f.write("======================================\n\n")
        
        if len(problematic_files) > 0:
            f.write("1. Data Normalization:\n")
            f.write("   Consider normalizing all data within a reasonable range (e.g., [-1, 1] or [0, 1]).\n\n")
            
            f.write("2. Handle Extreme Values:\n")
            f.write("   Use clipping to limit extreme values: data = np.clip(data, -threshold, threshold)\n\n")
            
            f.write("3. Fix Zero-dominant Features:\n")
            f.write("   Features with >95% zeros may cause instability. Consider removing these features\n")
            f.write("   or using alternative representations.\n\n")
            
            f.write("4. Specific File Fixes:\n")
            for prob_file in problematic_files:
                f.write(f"   - {prob_file['filename']}:\n")
                if 'error' in prob_file:
                    f.write(f"     File is corrupted or cannot be read. Consider removing or fixing it.\n")
                else:
                    for key, stats in prob_file['stats'].items():
                        if stats['has_nan']:
                            f.write(f"     Replace NaN values in '{key}' with mean or median.\n")
                        if stats['has_inf']:
                            f.write(f"     Replace infinite values in '{key}' with large finite values.\n")
                        if isinstance(stats['extreme_values_count'], int) and stats['extreme_values_count'] > 0:
                            f.write(f"     Clip extreme values in '{key}'.\n")
        else:
            f.write("No major issues detected in the dataset files.\n")
            f.write("Check your CNN implementation for numerical stability issues instead.\n")

    return problematic_files, results

def compare_datasets(dataset_a_path, dataset_b_path, output_dir="dataset_comparison"):
    """
    データセットAとBを比較し、主な違いを分析する
    
    Parameters:
    - dataset_a_path: 正常に動作するデータセットAのパス
    - dataset_b_path: NANが発生するデータセットBのパス
    - output_dir: 分析結果を保存するディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Analyzing Dataset A...")
    _, results_a = analyze_dataset(dataset_a_path, os.path.join(output_dir, "dataset_a"))
    
    print("\nAnalyzing Dataset B...")
    problematic_files_b, results_b = analyze_dataset(dataset_b_path, os.path.join(output_dir, "dataset_b"))
    
    # データセット間の分布の違いを可視化
    with open(os.path.join(output_dir, 'comparison_report.txt'), 'w') as f:
        f.write("Dataset Comparison Report\n")
        f.write("=======================\n\n")
        
        # 共通する統計情報のキーを検索
        common_keys = set(results_a.keys()).intersection(set(results_b.keys()))
        
        for key in common_keys:
            # 数値データのみを処理
            values_a = [v for v in results_a[key] if not isinstance(v, str)]
            values_b = [v for v in results_b[key] if not isinstance(v, str)]
            
            if len(values_a) > 0 and len(values_b) > 0:
                mean_a = np.mean(values_a)
                mean_b = np.mean(values_b)
                std_a = np.std(values_a)
                std_b = np.std(values_b)
                
                f.write(f"Feature: {key}\n")
                f.write(f"  Dataset A - Mean: {mean_a:.6f}, Std: {std_a:.6f}\n")
                f.write(f"  Dataset B - Mean: {mean_b:.6f}, Std: {std_b:.6f}\n")
                
                # 大きな違いを検出
                if abs(mean_a - mean_b) > max(std_a, std_b):
                    f.write(f"  WARNING: Significant difference in means detected!\n")
                
                if abs(std_a - std_b) / max(std_a, std_b) > 0.5:
                    f.write(f"  WARNING: Significant difference in standard deviations detected!\n")
                
                f.write("\n")
                
                # 分布の可視化
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.hist(values_a, bins=30, alpha=0.7, label='Dataset A')
                plt.title(f"{key} - Dataset A")
                plt.xlabel("Value")
                plt.ylabel("Count")
                
                plt.subplot(1, 2, 2)
                plt.hist(values_b, bins=30, alpha=0.7, label='Dataset B')
                plt.title(f"{key} - Dataset B")
                plt.xlabel("Value")
                plt.ylabel("Count")
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{key}_comparison.png"))
                plt.close()
    
    # 具体的な修正方法の提案
    with open(os.path.join(output_dir, 'fix_recommendations.txt'), 'w') as f:
        f.write("CNN NaN Issue Fix Recommendations\n")
        f.write("===============================\n\n")
        
        f.write("Based on the dataset comparison, here are recommended fixes for your CNN:\n\n")
        
        if len(problematic_files_b) > 0:
            f.write("1. Data Preprocessing Changes:\n")
            f.write("   - Add robust normalization before feeding data to the CNN\n")
            f.write("   - Handle outliers in Dataset B\n")
            f.write("   - Code example:\n")
            f.write("```python\n")
            f.write("def preprocess_data(data):\n")
            f.write("    # Replace NaN values with mean\n")
            f.write("    if np.any(np.isnan(data)):\n")
            f.write("        data = np.nan_to_num(data, nan=np.nanmean(data))\n")
            f.write("    \n")
            f.write("    # Replace infinite values\n") 
            f.write("    if np.any(np.isinf(data)):\n")
            f.write("        data = np.clip(data, -1e6, 1e6)\n")
            f.write("    \n")
            f.write("    # Robust normalization (less affected by outliers)\n")
            f.write("    q1, q3 = np.percentile(data, [25, 75])\n")
            f.write("    iqr = q3 - q1\n")
            f.write("    lower_bound = q1 - 1.5 * iqr\n")
            f.write("    upper_bound = q3 + 1.5 * iqr\n")
            f.write("    \n")
            f.write("    # Clip outliers\n")
            f.write("    data = np.clip(data, lower_bound, upper_bound)\n")
            f.write("    \n")
            f.write("    # Z-score normalization\n")
            f.write("    mean = np.mean(data)\n")
            f.write("    std = np.std(data)\n")
            f.write("    # Prevent division by zero\n")
            f.write("    if std < 1e-10:\n")
            f.write("        std = 1e-10\n")
            f.write("    normalized_data = (data - mean) / std\n")
            f.write("    \n")
            f.write("    return normalized_data\n")
            f.write("```\n\n")
            
            f.write("2. CNN Architecture Modifications:\n")
            f.write("   - Adjust your UltraStableCNN1Layer class:\n")
            f.write("```python\n")
            f.write("def numerically_stable_normalize(self, x, eps=1e-5):\n")
            f.write("    # Compute statistics along appropriate dimensions\n")
            f.write("    # For (batch, channel, length) data:\n")
            f.write("    x_mean = x.mean(dim=(0, 2), keepdim=True)\n")

if __name__ == "__main__":
    dataset_a_path = read_dataset_dir = "../../csv/one_word"
    dataset_b_path = "../../csv/minimum_continuous_hand_language"
    
    # データセットAとBの比較
    compare_datasets(dataset_a_path, dataset_b_path)
    
    # データセットの分析
    analyze_dataset(dataset_a_path)
    analyze_dataset(dataset_b_path)