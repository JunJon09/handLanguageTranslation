# PowerShell Train/Val Separated Loss Analysis
$lossLines = Get-Content temp_losses.txt

Write-Host "=== Train/Val Separated Loss Analysis ===" -ForegroundColor Green

# 正規表現でより正確に抽出
$trainData = @()
$valData = @()
$count = 0

foreach ($line in $lossLines) {
    if ($line -match "Conv=(\d+\.?\d*), Seq=(\d+\.?\d*), Dist=(\d+\.?\d*).*?(\d+\.?\d*)$") {
        $conv = [float]$matches[1]
        $seq = [float]$matches[2]  
        $dist = [float]$matches[3]
        $total = [float]$matches[4]
        
        $dataPoint = [PSCustomObject]@{
            Index = $count
            Conv = $conv
            Seq = $seq
            Dist = $dist
            Total = $total
        }
        
        # 総計の値でtrain/valを判別 (1-5の間がval、それ以外はtrain)
        if ($total -ge 1.0 -and $total -le 5.0) {
            $valData += $dataPoint
        } else {
            $trainData += $dataPoint
        }
        
        $count++
    }
}

Write-Host "解析結果:" -ForegroundColor Yellow
Write-Host "  総データ数: $count"
Write-Host "  Train データ数: $($trainData.Count)"
Write-Host "  Val データ数: $($valData.Count)"
Write-Host ""

# Train データの分析
if ($trainData.Count -gt 0) {
    Write-Host "=== TRAIN Loss Analysis ===" -ForegroundColor Red
    
    $trainFirst = $trainData | Select-Object -First 100
    $trainLast = $trainData | Select-Object -Last 100
    
    function AnalyzeLossData($data, $dataName) {
        $convAvg = ($data.Conv | Measure-Object -Average).Average
        $seqAvg = ($data.Seq | Measure-Object -Average).Average
        $distAvg = ($data.Dist | Measure-Object -Average).Average
        $totalAvg = ($data.Total | Measure-Object -Average).Average
        
        $convRatio = ($convAvg / $totalAvg) * 100
        $seqRatio = ($seqAvg / $totalAvg) * 100
        $distRatio = ($distAvg / $totalAvg) * 100
        
        Write-Host "${dataName}:"
        Write-Host "  Conv: $($convAvg.ToString('F2')) ($($convRatio.ToString('F1'))%)"
        Write-Host "  Seq:  $($seqAvg.ToString('F2')) ($($seqRatio.ToString('F1'))%)"
        Write-Host "  Dist: $($distAvg.ToString('F2')) ($($distRatio.ToString('F1'))%)"
        Write-Host "  Total: $($totalAvg.ToString('F2'))"
        Write-Host ""
        
        return @{
            Conv = $convAvg
            Seq = $seqAvg
            Dist = $distAvg
            Total = $totalAvg
        }
    }
    
    $trainFirstStats = AnalyzeLossData $trainFirst "Train Early (first 100)"
    $trainLastStats = AnalyzeLossData $trainLast "Train Late (last 100)"
    
    # Train の改善度計算
    $trainImprovement = (($trainFirstStats.Total - $trainLastStats.Total) / $trainFirstStats.Total) * 100
    Write-Host "Train 改善度: $($trainImprovement.ToString('F1'))%" -ForegroundColor Green
}

# Val データの分析
if ($valData.Count -gt 0) {
    Write-Host "`n=== VALIDATION Loss Analysis ===" -ForegroundColor Blue
    
    $valFirst = $valData | Select-Object -First 50
    $valLast = $valData | Select-Object -Last 50
    
    $valFirstStats = AnalyzeLossData $valFirst "Val Early (first 50)"
    $valLastStats = AnalyzeLossData $valLast "Val Late (last 50)"
    
    # Val の改善度計算
    $valImprovement = (($valFirstStats.Total - $valLastStats.Total) / $valFirstStats.Total) * 100
    Write-Host "Val 改善度: $($valImprovement.ToString('F1'))%" -ForegroundColor Green
}

# Train vs Val の比較
if ($trainData.Count -gt 0 -and $valData.Count -gt 0) {
    Write-Host "`n=== TRAIN vs VAL Comparison ===" -ForegroundColor Magenta
    
    $trainAvgTotal = ($trainData.Total | Measure-Object -Average).Average
    $valAvgTotal = ($valData.Total | Measure-Object -Average).Average
    
    Write-Host "平均総損失比較:"
    Write-Host "  Train: $($trainAvgTotal.ToString('F2'))"
    Write-Host "  Val:   $($valAvgTotal.ToString('F2'))"
    Write-Host "  比率:  Train/Val = $(($trainAvgTotal / $valAvgTotal).ToString('F1'))"
    
    if ($trainAvgTotal / $valAvgTotal -gt 5.0) {
        Write-Host "🔴 CRITICAL: Train/Val 損失の差が異常に大きい (overfitting疑い)" -ForegroundColor Red
    }
    
    # 各成分の比較
    $trainConvAvg = ($trainData.Conv | Measure-Object -Average).Average
    $trainSeqAvg = ($trainData.Seq | Measure-Object -Average).Average
    $trainDistAvg = ($trainData.Dist | Measure-Object -Average).Average
    
    $valConvAvg = ($valData.Conv | Measure-Object -Average).Average
    $valSeqAvg = ($valData.Seq | Measure-Object -Average).Average
    $valDistAvg = ($valData.Dist | Measure-Object -Average).Average
    
    Write-Host "`n成分別比較 (Train vs Val):"
    Write-Host "  Conv: $($trainConvAvg.ToString('F2')) vs $($valConvAvg.ToString('F2')) (比率: $($trainConvAvg / $valConvAvg | ForEach-Object {$_.ToString('F1')}))"
    Write-Host "  Seq:  $($trainSeqAvg.ToString('F2')) vs $($valSeqAvg.ToString('F2')) (比率: $($trainSeqAvg / $valSeqAvg | ForEach-Object {$_.ToString('F1')}))"
    Write-Host "  Dist: $($trainDistAvg.ToString('F2')) vs $($valDistAvg.ToString('F2')) (比率: $($trainDistAvg / $valDistAvg | ForEach-Object {$_.ToString('F1')}))"
}

# Distillation Loss の特別分析
Write-Host "`n=== Distillation Loss Deep Analysis ===" -ForegroundColor Yellow

if ($trainData.Count -gt 0) {
    $trainDistStats = $trainData.Dist | Measure-Object -Average -Minimum -Maximum
    Write-Host "Train Distillation:"
    Write-Host "  Average: $($trainDistStats.Average.ToString('F4'))"
    Write-Host "  Min: $($trainDistStats.Minimum.ToString('F4'))"
    Write-Host "  Max: $($trainDistStats.Maximum.ToString('F4'))"
    
    $trainLowDistCount = ($trainData.Dist | Where-Object { $_ -lt 0.01 }).Count
    Write-Host "  Dist < 0.01: $trainLowDistCount ($([Math]::Round($trainLowDistCount/$trainData.Count*100, 1))%)"
}

if ($valData.Count -gt 0) {
    $valDistStats = $valData.Dist | Measure-Object -Average -Minimum -Maximum
    Write-Host "Val Distillation:"
    Write-Host "  Average: $($valDistStats.Average.ToString('F4'))"
    Write-Host "  Min: $($valDistStats.Minimum.ToString('F4'))"
    Write-Host "  Max: $($valDistStats.Maximum.ToString('F4'))"
    
    $valLowDistCount = ($valData.Dist | Where-Object { $_ -lt 0.01 }).Count
    Write-Host "  Dist < 0.01: $valLowDistCount ($([Math]::Round($valLowDistCount/$valData.Count*100, 1))%)"
}

Write-Host "`n=== 問題診断 ===" -ForegroundColor Red

# 1. Overfitting 診断
if ($trainData.Count -gt 0 -and $valData.Count -gt 0) {
    $overfitRatio = $trainAvgTotal / $valAvgTotal
    if ($overfitRatio -gt 10.0) {
        Write-Host "🔴 SEVERE OVERFITTING: Train/Val比率 = $($overfitRatio.ToString('F1'))" -ForegroundColor Red
    } elseif ($overfitRatio -gt 5.0) {
        Write-Host "🟠 MODERATE OVERFITTING: Train/Val比率 = $($overfitRatio.ToString('F1'))" -ForegroundColor Yellow
    }
}

# 2. Distillation 機能不全診断
if ($trainData.Count -gt 0) {
    if ($trainDistStats.Average -lt 0.05) {
        Write-Host "🔴 DISTILLATION FAILURE: Train平均Dist損失 = $($trainDistStats.Average.ToString('F4'))" -ForegroundColor Red
    }
}

# 3. CNN vs LSTM 不均衡診断
if ($trainData.Count -gt 0) {
    $convSeqRatio = $trainConvAvg / $trainSeqAvg
    if ($convSeqRatio -lt 0.5) {
        Write-Host "🔴 CNN UNDERPERFORMING: Conv/Seq比率 = $($convSeqRatio.ToString('F2'))" -ForegroundColor Red
    }
}

Write-Host "分析完了。詳細は上記の診断結果を参照してください。" -ForegroundColor Green
