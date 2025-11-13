# PowerShell Advanced Loss Analysis
$lossLines = Get-Content temp_losses.txt

Write-Host "=== Detailed Loss Component Analysis ===" -ForegroundColor Green

# 正規表現でより正確に抽出
$lossData = @()
$count = 0

foreach ($line in $lossLines) {
    if ($line -match "Conv=(\d+\.?\d*), Seq=(\d+\.?\d*), Dist=(\d+\.?\d*).*?(\d+\.?\d*)$") {
        $conv = [float]$matches[1]
        $seq = [float]$matches[2]  
        $dist = [float]$matches[3]
        $total = [float]$matches[4]
        
        $lossData += [PSCustomObject]@{
            Index = $count
            Conv = $conv
            Seq = $seq
            Dist = $dist
            Total = $total
        }
        $count++
    }
}

Write-Host "Successfully parsed $($lossData.Count) loss entries" -ForegroundColor Green

# 初期、中期、最終の比較
$first1000 = $lossData | Select-Object -First 1000
$last1000 = $lossData | Select-Object -Last 1000
$middle1000 = $lossData | Select-Object -Skip ([Math]::Floor($lossData.Count/2) - 500) -First 1000

Write-Host "`n=== Training Phase Analysis ===" -ForegroundColor Yellow

function AnalyzePhase($data, $phaseName) {
    $convAvg = ($data.Conv | Measure-Object -Average).Average
    $seqAvg = ($data.Seq | Measure-Object -Average).Average
    $distAvg = ($data.Dist | Measure-Object -Average).Average
    $totalAvg = ($data.Total | Measure-Object -Average).Average
    
    $convRatio = ($convAvg / $totalAvg) * 100
    $seqRatio = ($seqAvg / $totalAvg) * 100
    $distRatio = ($distAvg / $totalAvg) * 100
    
    Write-Host "$phaseName Phase:"
    Write-Host "  Conv: $($convAvg.ToString('F2')) ($($convRatio.ToString('F1'))%)"
    Write-Host "  Seq:  $($seqAvg.ToString('F2')) ($($seqRatio.ToString('F1'))%)"
    Write-Host "  Dist: $($distAvg.ToString('F2')) ($($distRatio.ToString('F1'))%)"
    Write-Host "  Total: $($totalAvg.ToString('F2'))"
    Write-Host ""
}

AnalyzePhase $first1000 "Early"
AnalyzePhase $middle1000 "Middle"  
AnalyzePhase $last1000 "Late"

# Distillation Loss の異常検出
Write-Host "=== Distillation Loss Analysis ===" -ForegroundColor Red
$distStats = $lossData.Dist | Measure-Object -Average -Minimum -Maximum
Write-Host "Distillation Loss Statistics:"
Write-Host "  Average: $($distStats.Average.ToString('F4'))"
Write-Host "  Min: $($distStats.Minimum.ToString('F4'))"
Write-Host "  Max: $($distStats.Maximum.ToString('F4'))"

$lowDistCount = ($lossData.Dist | Where-Object { $_ -lt 0.01 }).Count
Write-Host "  Entries with Dist < 0.01: $lowDistCount ($([Math]::Round($lowDistCount/$lossData.Count*100, 1))%)"

# Conv vs Seq 比率の推移
Write-Host "`n=== Conv vs Seq Ratio Analysis ===" -ForegroundColor Cyan
$ratios = $lossData | ForEach-Object { $_.Conv / $_.Seq }
$avgRatio = ($ratios | Measure-Object -Average).Average
Write-Host "Average Conv/Seq ratio: $($avgRatio.ToString('F3'))"

if ($avgRatio -lt 1.0) {
    Write-Host "⚠️  Conv loss is consistently lower than Seq loss" -ForegroundColor Yellow
} else {
    Write-Host "✅ Conv loss is higher than Seq loss" -ForegroundColor Green
}

Write-Host "`n=== Potential Issues Identified ===" -ForegroundColor Red

if ($distStats.Average -lt 0.1) {
    Write-Host "🔴 CRITICAL: Distillation loss is extremely low (avg: $($distStats.Average.ToString('F4')))"
    Write-Host "   This suggests the distillation mechanism is not working properly."
}

if ($avgRatio -lt 0.8) {
    Write-Host "🔴 CRITICAL: Conv loss much lower than Seq loss"
    Write-Host "   This suggests CNN is not learning effectively compared to RNN."
}

Write-Host "`nAnalysis saved. Check the temp_losses.txt file for raw data." -ForegroundColor Green
