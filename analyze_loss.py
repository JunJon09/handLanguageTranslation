#!/usr/bin/env python3
"""
Loss分析スクリプト
ログファイルからConv, Seq, Distの各損失成分を抽出して分析
"""
import re
import pandas as pd
import matplotlib.pyplot as plt
import sys

def parse_loss_log(log_file_path):
    """
    ログファイルから損失情報を抽出
    """
    loss_pattern = r"損失: Conv=(\d+\.\d+), Seq=(\d+\.\d+), Dist=(\d+\.\d+), 総計=(\d+\.\d+)"
    
    losses = {
        'Conv': [],
        'Seq': [],
        'Dist': [],
        'Total': []
    }
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(loss_pattern, line)
                if match:
                    conv_loss = float(match.group(1))
                    seq_loss = float(match.group(2))
                    dist_loss = float(match.group(3))
                    total_loss = float(match.group(4))
                    
                    losses['Conv'].append(conv_loss)
                    losses['Seq'].append(seq_loss)
                    losses['Dist'].append(dist_loss)
                    losses['Total'].append(total_loss)
    
    except FileNotFoundError:
        print(f"ログファイルが見つかりません: {log_file_path}")
        return None
    
    return losses

def analyze_losses(losses):
    """
    損失成分の分析
    """
    if not losses or not losses['Conv']:
        print("損失データが見つかりません")
        return
    
    df = pd.DataFrame(losses)
    
    print("=== Loss成分分析 ===")
    print(f"Total samples: {len(df)}")
    print("\n各成分の統計:")
    print(df.describe())
    
    print("\n各成分の割合 (最新100サンプル):")
    recent_df = df.tail(100)
    total_avg = recent_df['Total'].mean()
    
    conv_ratio = recent_df['Conv'].mean() / total_avg * 100
    seq_ratio = recent_df['Seq'].mean() / total_avg * 100
    dist_ratio = recent_df['Dist'].mean() / total_avg * 100
    
    print(f"Conv: {conv_ratio:.1f}%")
    print(f"Seq:  {seq_ratio:.1f}%")
    print(f"Dist: {dist_ratio:.1f}%")
    
    # 異常値検出
    print("\n異常値検出:")
    for component in ['Conv', 'Seq', 'Dist']:
        q75, q25 = df[component].quantile([0.75, 0.25])
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr
        outliers = df[df[component] > outlier_threshold]
        if len(outliers) > 0:
            print(f"{component}: {len(outliers)}個の異常値 (閾値: {outlier_threshold:.3f})")
    
    return df

def plot_losses(df):
    """
    損失の可視化
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 各成分の推移
    axes[0, 0].plot(df['Conv'], label='Conv', alpha=0.7)
    axes[0, 0].plot(df['Seq'], label='Seq', alpha=0.7)
    axes[0, 0].plot(df['Dist'], label='Dist', alpha=0.7)
    axes[0, 0].set_title('Loss Components Over Time')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # 総損失
    axes[0, 1].plot(df['Total'], color='red')
    axes[0, 1].set_title('Total Loss Over Time')
    axes[0, 1].set_yscale('log')
    
    # 各成分の分布
    axes[1, 0].hist([df['Conv'], df['Seq'], df['Dist']], 
                   bins=50, alpha=0.7, label=['Conv', 'Seq', 'Dist'])
    axes[1, 0].set_title('Loss Distribution')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    
    # 成分比率の推移
    window_size = 100
    if len(df) > window_size:
        rolling_total = df['Total'].rolling(window=window_size).mean()
        conv_ratio = df['Conv'].rolling(window=window_size).mean() / rolling_total * 100
        seq_ratio = df['Seq'].rolling(window=window_size).mean() / rolling_total * 100
        dist_ratio = df['Dist'].rolling(window=window_size).mean() / rolling_total * 100
        
        axes[1, 1].plot(conv_ratio, label='Conv %')
        axes[1, 1].plot(seq_ratio, label='Seq %')
        axes[1, 1].plot(dist_ratio, label='Dist %')
        axes[1, 1].set_title(f'Component Ratios (Rolling {window_size})')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('loss_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用法: python analyze_loss.py <ログファイルパス>")
        print("例: python analyze_loss.py training.log")
        sys.exit(1)
    
    log_file = sys.argv[1]
    losses = parse_loss_log(log_file)
    
    if losses:
        df = analyze_losses(losses)
        if df is not None:
            plot_losses(df)
