import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# データの再読み込み
data = []
with open(r"D:\Phonenix-2014-translate\handLanguageTranslation\ContinuousSignLanguage\CNN_BiLSTM\logs\training_20250908_164346.log", 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if "損失: Conv=" in line:
            try:
                parts = line.split("損失: Conv=")[1]
                conv = float(parts.split(", Seq=")[0])
                seq = float(parts.split(", Seq=")[1].split(", Dist=")[0])
                dist = float(parts.split(", Dist=")[1].split(", 総計=")[0])
                total = float(parts.split(", 総計=")[1].split()[0])
                data.append([conv, seq, dist, total])
            except:
                continue

df = pd.DataFrame(data, columns=['Conv', 'Seq', 'Dist', 'Total'])

# Train/Val分離
train_mask = df['Total'] > 5
val_mask = df['Total'] <= 5
train_df = df[train_mask]
val_df = df[val_mask]

# 図のサイズを大きくして文字の重なりを防ぐ
plt.figure(figsize=(20, 15))  # さらに大きく
plt.rcParams.update({'font.size': 12})  # フォントサイズを調整

# サブプロット1: 分布比較（ヒストグラム）
plt.subplot(2, 3, 1)
plt.hist(train_df['Total'], bins=50, alpha=0.7, label=f'Train Loss (n={len(train_df)})', color='blue', density=True)
plt.hist(val_df['Total'], bins=30, alpha=0.7, label=f'Val Loss (n={len(val_df)})', color='orange', density=True)
plt.xlabel('Total Loss', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Loss Distribution Comparison\n(Normalized)', fontsize=16, pad=20)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# サブプロット2: 成分別比較（バープロット）
plt.subplot(2, 3, 2)
components = ['Conv', 'Seq', 'Dist']
train_means = [train_df[comp].mean() for comp in components]
val_means = [val_df[comp].mean() for comp in components]

x = np.arange(len(components))
width = 0.35

bars1 = plt.bar(x - width/2, train_means, width, label='Train', color='blue', alpha=0.7)
bars2 = plt.bar(x + width/2, val_means, width, label='Val', color='orange', alpha=0.7)

# 値をバーの上に表示
for i, (train_val, val_val) in enumerate(zip(train_means, val_means)):
    plt.text(i - width/2, train_val + 0.5, f'{train_val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.text(i + width/2, val_val + 0.1, f'{val_val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xlabel('Loss Components', fontsize=14)
plt.ylabel('Average Loss Value', fontsize=14)
plt.title('Loss Components: Train vs Val\n(Average Values)', fontsize=16, pad=20)
plt.xticks(x, components, fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# サブプロット3: 時系列トレンド（サンプリング）
plt.subplot(2, 3, 3)
# データが多すぎるので間引いてプロット
sample_rate = max(1, len(df) // 1000)  # 最大1000ポイント
sampled_indices = range(0, len(df), sample_rate)

plt.plot([i for i in sampled_indices if train_mask.iloc[i]], 
         [df.iloc[i]['Total'] for i in sampled_indices if train_mask.iloc[i]], 
         'b.', alpha=0.6, markersize=3, label=f'Train (every {sample_rate}th point)')
plt.plot([i for i in sampled_indices if val_mask.iloc[i]], 
         [df.iloc[i]['Total'] for i in sampled_indices if val_mask.iloc[i]], 
         'ro', alpha=0.8, markersize=4, label=f'Val (every {sample_rate}th point)')

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Total Loss', fontsize=14)
plt.title('Loss Trends Over Time\n(Sampled Data)', fontsize=16, pad=20)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# サブプロット4: 比率分析
plt.subplot(2, 3, 4)
ratios = []
labels = []
colors = []

for comp in components:
    if val_df[comp].mean() > 0:  # ゼロ除算回避
        ratio = train_df[comp].mean() / val_df[comp].mean()
        ratios.append(ratio)
        labels.append(f'{comp}\n({ratio:.1f}x)')
        
        # 比率に応じて色を変更
        if ratio > 5:
            colors.append('red')
        elif ratio > 2:
            colors.append('orange')
        else:
            colors.append('green')

bars = plt.bar(range(len(ratios)), ratios, color=colors, alpha=0.7)
plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal (1.0x)')
plt.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Warning (2.0x)')
plt.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Critical (5.0x)')

# 値をバーの上に表示
for i, (bar, ratio) in enumerate(zip(bars, ratios)):
    plt.text(i, ratio + 0.2, f'{ratio:.1f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.xlabel('Loss Components', fontsize=14)
plt.ylabel('Train/Val Ratio', fontsize=14)
plt.title('Overfitting Analysis\n(Train/Val Loss Ratios)', fontsize=16, pad=20)
plt.xticks(range(len(labels)), [comp for comp in components], fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')

# サブプロット5: 相関分析
plt.subplot(2, 3, 5)
plt.scatter(train_df['Conv'], train_df['Seq'], alpha=0.5, s=10, label='Train: Conv vs Seq', color='blue')
plt.scatter(val_df['Conv'], val_df['Seq'], alpha=0.7, s=15, label='Val: Conv vs Seq', color='orange')
plt.xlabel('Conv Loss', fontsize=14)
plt.ylabel('Seq Loss', fontsize=14)
plt.title('Conv vs Seq Loss Correlation\n(Scatter Plot)', fontsize=16, pad=20)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# サブプロット6: 統計サマリー
plt.subplot(2, 3, 6)
plt.axis('off')

# 統計テーブル
stats_text = f"""
COMPREHENSIVE LOSS ANALYSIS SUMMARY

📊 Dataset Statistics:
• Total Entries: {len(df):,}
• Train Entries: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)
• Val Entries: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)

🔍 Average Losses:
           Train    Val     Ratio
Conv:     {train_df['Conv'].mean():.2f}    {val_df['Conv'].mean():.2f}    {train_df['Conv'].mean()/val_df['Conv'].mean():.1f}x
Seq:      {train_df['Seq'].mean():.2f}   {val_df['Seq'].mean():.2f}    {train_df['Seq'].mean()/val_df['Seq'].mean():.1f}x
Dist:     {train_df['Dist'].mean():.3f}   {val_df['Dist'].mean():.3f}   {train_df['Dist'].mean()/val_df['Dist'].mean():.1f}x
Total:    {train_df['Total'].mean():.2f}   {val_df['Total'].mean():.2f}    {train_df['Total'].mean()/val_df['Total'].mean():.1f}x

⚠️  Critical Issues:
• Seq Loss ratio: {train_df['Seq'].mean()/val_df['Seq'].mean():.1f}x (SEVERE)
• Overall ratio: {train_df['Total'].mean()/val_df['Total'].mean():.1f}x (HIGH)
• Dist contribution: {train_df['Dist'].mean()/train_df['Total'].mean()*100:.1f}% (LOW)

🎯 Recommendations:
1. Increase BiLSTM Dropout (0.3→0.5)
2. Add CNN Dropout layers
3. Increase Distillation weight (0.15→0.3)
4. Investigate data preprocessing differences
"""

plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

plt.tight_layout(pad=3.0)  # パディングを増やして文字の重なりを防ぐ
plt.savefig('comprehensive_loss_analysis_improved.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 改良版分析図が生成されました!")
print("📁 ファイル名: comprehensive_loss_analysis_improved.png")