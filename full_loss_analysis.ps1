# PowerShell Loss Visualization and Data Bias Analysis
param(
    [string]$OutputDir = "loss_analysis_output"
)

# Create output directory
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir
}

Write-Host "=== Loss Data Analysis and Visualization ===" -ForegroundColor Green

# Load and parse loss data
$lossLines = Get-Content temp_losses.txt
$trainData = @()
$valData = @()
$allData = @()
$timeIndex = 0

foreach ($line in $lossLines) {
    if ($line -match "Conv=(\d+\.?\d*), Seq=(\d+\.?\d*), Dist=(\d+\.?\d*).*?(\d+\.?\d*)$") {
        $conv = [float]$matches[1]
        $seq = [float]$matches[2]  
        $dist = [float]$matches[3]
        $total = [float]$matches[4]
        
        $dataPoint = [PSCustomObject]@{
            TimeIndex = $timeIndex
            Conv = $conv
            Seq = $seq
            Dist = $dist
            Total = $total
            Type = if ($total -ge 1.0 -and $total -le 5.0) { "Val" } else { "Train" }
        }
        
        $allData += $dataPoint
        
        if ($dataPoint.Type -eq "Val") {
            $valData += $dataPoint
        } else {
            $trainData += $dataPoint
        }
        
        $timeIndex++
    }
}

Write-Host "Data loaded: $($trainData.Count) train, $($valData.Count) val entries"

# Export data for plotting
$trainData | Export-Csv -Path "$OutputDir\train_losses.csv" -NoTypeInformation -Encoding UTF8
$valData | Export-Csv -Path "$OutputDir\val_losses.csv" -NoTypeInformation -Encoding UTF8
$allData | Export-Csv -Path "$OutputDir\all_losses.csv" -NoTypeInformation -Encoding UTF8

Write-Host "CSV files exported to $OutputDir"

# Temporal analysis - check for patterns
Write-Host "`n=== Temporal Pattern Analysis ===" -ForegroundColor Yellow

# Data distribution analysis
Write-Host "`n=== Data Distribution Analysis ===" -ForegroundColor Cyan

# Check train/val alternation pattern
$consecutiveTrainCount = 0
$consecutiveValCount = 0
$maxConsecutiveTrain = 0
$maxConsecutiveVal = 0
$switches = 0
$lastType = ""

foreach ($entry in $allData) {
    if ($entry.Type -eq "Train") {
        $consecutiveTrainCount++
        if ($consecutiveValCount -gt 0) {
            $maxConsecutiveVal = [Math]::Max($maxConsecutiveVal, $consecutiveValCount)
            $consecutiveValCount = 0
            if ($lastType -eq "Val") { $switches++ }
        }
    } else {
        $consecutiveValCount++
        if ($consecutiveTrainCount -gt 0) {
            $maxConsecutiveTrain = [Math]::Max($maxConsecutiveTrain, $consecutiveTrainCount)
            $consecutiveTrainCount = 0
            if ($lastType -eq "Train") { $switches++ }
        }
    }
    $lastType = $entry.Type
}

Write-Host "Train/Val Pattern Analysis:"
Write-Host "  Max consecutive train entries: $maxConsecutiveTrain"
Write-Host "  Max consecutive val entries: $maxConsecutiveVal"
Write-Host "  Number of train/val switches: $switches"
Write-Host "  Average entries per switch: $([Math]::Round($allData.Count / $switches, 1))"

# Generate Python plotting script
$pythonScript = @"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load data
train_df = pd.read_csv('train_losses.csv')
val_df = pd.read_csv('val_losses.csv')
all_df = pd.read_csv('all_losses.csv')

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Train/Val Loss Analysis - Phoenix 2014 Sign Language', fontsize=16, fontweight='bold')

# 1. Loss components over time
ax1 = axes[0, 0]
ax1.plot(train_df['TimeIndex'], train_df['Conv'], label='Train Conv', alpha=0.7, linewidth=0.5)
ax1.plot(train_df['TimeIndex'], train_df['Seq'], label='Train Seq', alpha=0.7, linewidth=0.5)
ax1.plot(train_df['TimeIndex'], train_df['Dist'], label='Train Dist', alpha=0.7, linewidth=0.5)
ax1.set_title('Train Loss Components Over Time')
ax1.set_xlabel('Time Index')
ax1.set_ylabel('Loss Value')
ax1.legend()
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# 2. Val loss components
ax2 = axes[0, 1]
ax2.plot(val_df['TimeIndex'], val_df['Conv'], label='Val Conv', alpha=0.8, linewidth=1)
ax2.plot(val_df['TimeIndex'], val_df['Seq'], label='Val Seq', alpha=0.8, linewidth=1)
ax2.plot(val_df['TimeIndex'], val_df['Dist'], label='Val Dist', alpha=0.8, linewidth=1)
ax2.set_title('Validation Loss Components Over Time')
ax2.set_xlabel('Time Index')
ax2.set_ylabel('Loss Value')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Train vs Val total loss
ax3 = axes[0, 2]
ax3.plot(train_df['TimeIndex'], train_df['Total'], label='Train Total', alpha=0.6, linewidth=0.5)
ax3.plot(val_df['TimeIndex'], val_df['Total'], label='Val Total', alpha=0.8, linewidth=1, color='red')
ax3.set_title('Train vs Val Total Loss')
ax3.set_xlabel('Time Index')
ax3.set_ylabel('Total Loss')
ax3.legend()
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# 4. Loss distribution comparison
ax4 = axes[1, 0]
components = ['Conv', 'Seq', 'Dist']
train_avgs = [train_df['Conv'].mean(), train_df['Seq'].mean(), train_df['Dist'].mean()]
val_avgs = [val_df['Conv'].mean(), val_df['Seq'].mean(), val_df['Dist'].mean()]

x = np.arange(len(components))
width = 0.35

ax4.bar(x - width/2, train_avgs, width, label='Train', alpha=0.8)
ax4.bar(x + width/2, val_avgs, width, label='Val', alpha=0.8)
ax4.set_title('Average Loss Components Comparison')
ax4.set_xlabel('Loss Component')
ax4.set_ylabel('Average Loss')
ax4.set_xticks(x)
ax4.set_xticklabels(components)
ax4.legend()
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)

# 5. Train/Val ratio analysis
ax5 = axes[1, 1]
ratios = []
ratio_labels = []
for comp in components:
    if val_df[comp].mean() > 0:
        ratio = train_df[comp].mean() / val_df[comp].mean()
        ratios.append(ratio)
        ratio_labels.append(f'{comp}\n({ratio:.1f}x)')
    else:
        ratios.append(0)
        ratio_labels.append(f'{comp}\n(N/A)')

bars = ax5.bar(range(len(ratios)), ratios, alpha=0.8)
ax5.set_title('Train/Val Loss Ratios')
ax5.set_xlabel('Loss Component')
ax5.set_ylabel('Train/Val Ratio')
ax5.set_xticks(range(len(ratios)))
ax5.set_xticklabels(ratio_labels)
ax5.grid(True, alpha=0.3)

# Highlight problematic ratios
for i, (bar, ratio) in enumerate(zip(bars, ratios)):
    if ratio > 5:
        bar.set_color('red')
        bar.set_alpha(0.9)

# 6. Data pattern analysis
ax6 = axes[1, 2]
# Plot train/val occurrence pattern
train_val_pattern = [1 if x['Type'] == 'Train' else 0 for _, x in all_df.iterrows()]
ax6.plot(all_df['TimeIndex'], train_val_pattern, alpha=0.7, linewidth=0.5)
ax6.set_title('Train/Val Data Pattern')
ax6.set_xlabel('Time Index')
ax6.set_ylabel('Type (0=Val, 1=Train)')
ax6.set_ylim(-0.1, 1.1)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('loss_analysis.pdf', bbox_inches='tight')
print("Plots saved as loss_analysis.png and loss_analysis.pdf")

# Generate summary statistics
print("\n=== DETAILED ANALYSIS SUMMARY ===")
print(f"Train entries: {len(train_df)}")
print(f"Val entries: {len(val_df)}")
print(f"Train/Val ratio: {len(train_df)/len(val_df):.1f}")

print(f"\nTrain loss averages:")
print(f"  Conv: {train_df['Conv'].mean():.3f}")
print(f"  Seq:  {train_df['Seq'].mean():.3f}")
print(f"  Dist: {train_df['Dist'].mean():.3f}")
print(f"  Total: {train_df['Total'].mean():.3f}")

print(f"\nVal loss averages:")
print(f"  Conv: {val_df['Conv'].mean():.3f}")
print(f"  Seq:  {val_df['Seq'].mean():.3f}")
print(f"  Dist: {val_df['Dist'].mean():.3f}")
print(f"  Total: {val_df['Total'].mean():.3f}")

print(f"\nTrain/Val ratios:")
for comp in ['Conv', 'Seq', 'Dist', 'Total']:
    if val_df[comp].mean() > 0:
        ratio = train_df[comp].mean() / val_df[comp].mean()
        print(f"  {comp}: {ratio:.1f}x")

plt.show()
"@

# Save Python script
$pythonScript | Out-File -FilePath "$OutputDir\plot_losses.py" -Encoding UTF8

Write-Host "`n=== Files Generated ===" -ForegroundColor Green
Write-Host "Data files:"
Write-Host "  - train_losses.csv"
Write-Host "  - val_losses.csv" 
Write-Host "  - all_losses.csv"
Write-Host ""
Write-Host "Visualization:"
Write-Host "  - plot_losses.py (Python script)"
Write-Host ""
Write-Host "To generate plots, run: python $OutputDir\plot_losses.py"

# Data bias summary
Write-Host "`n=== DATA BIAS ANALYSIS SUMMARY ===" -ForegroundColor Red

Write-Host "Potential biases detected:"

if ($maxConsecutiveTrain -gt 1000) {
    Write-Host "  - CRITICAL: Very long consecutive training sequences ($maxConsecutiveTrain)" -ForegroundColor Red
}

if ($switches -lt 50) {
    Write-Host "  - WARNING: Few train/val switches ($switches) - possible batch processing" -ForegroundColor Yellow
}

$trainValRatio = $trainData.Count / $valData.Count
if ($trainValRatio -gt 20) {
    Write-Host "  - WARNING: Highly imbalanced train/val ratio ($($trainValRatio.ToString('F1')))" -ForegroundColor Yellow
}

Write-Host "`nRecommendations:"
Write-Host "  1. Check if validation is run after every epoch or multiple epochs"
Write-Host "  2. Verify train/val data splitting methodology" 
Write-Host "  3. Consider data preprocessing differences between train/val"

Write-Host "`nAnalysis complete! Check $OutputDir for detailed data files." -ForegroundColor Green
