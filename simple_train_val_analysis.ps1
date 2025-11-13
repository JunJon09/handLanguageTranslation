# PowerShell Train/Val Separated Loss Analysis - Simple Version
$lossLines = Get-Content temp_losses.txt

Write-Host "=== Train/Val Separated Loss Analysis ===" -ForegroundColor Green

# Extract loss data
$trainData = @()
$valData = @()

foreach ($line in $lossLines) {
    if ($line -match "Conv=(\d+\.?\d*), Seq=(\d+\.?\d*), Dist=(\d+\.?\d*).*?(\d+\.?\d*)$") {
        $conv = [float]$matches[1]
        $seq = [float]$matches[2]  
        $dist = [float]$matches[3]
        $total = [float]$matches[4]
        
        $dataPoint = [PSCustomObject]@{
            Conv = $conv
            Seq = $seq
            Dist = $dist
            Total = $total
        }
        
        # Separate train/val based on total loss (1-5 = val, others = train)
        if ($total -ge 1.0 -and $total -le 5.0) {
            $valData += $dataPoint
        } else {
            $trainData += $dataPoint
        }
    }
}

Write-Host "Data Summary:"
Write-Host "  Total entries: $($trainData.Count + $valData.Count)"
Write-Host "  Train entries: $($trainData.Count)"
Write-Host "  Val entries: $($valData.Count)"
Write-Host ""

# Train Analysis
if ($trainData.Count -gt 0) {
    Write-Host "=== TRAIN Analysis ===" -ForegroundColor Red
    
    $trainConv = ($trainData.Conv | Measure-Object -Average -Min -Max)
    $trainSeq = ($trainData.Seq | Measure-Object -Average -Min -Max)
    $trainDist = ($trainData.Dist | Measure-Object -Average -Min -Max)
    $trainTotal = ($trainData.Total | Measure-Object -Average -Min -Max)
    
    Write-Host "Conv Loss - Avg: $($trainConv.Average.ToString('F2')), Min: $($trainConv.Minimum.ToString('F2')), Max: $($trainConv.Maximum.ToString('F2'))"
    Write-Host "Seq Loss  - Avg: $($trainSeq.Average.ToString('F2')), Min: $($trainSeq.Minimum.ToString('F2')), Max: $($trainSeq.Maximum.ToString('F2'))"
    Write-Host "Dist Loss - Avg: $($trainDist.Average.ToString('F3')), Min: $($trainDist.Minimum.ToString('F3')), Max: $($trainDist.Maximum.ToString('F3'))"
    Write-Host "Total     - Avg: $($trainTotal.Average.ToString('F2')), Min: $($trainTotal.Minimum.ToString('F2')), Max: $($trainTotal.Maximum.ToString('F2'))"
    
    # Component ratios for train
    $trainConvRatio = ($trainConv.Average / $trainTotal.Average) * 100
    $trainSeqRatio = ($trainSeq.Average / $trainTotal.Average) * 100
    $trainDistRatio = ($trainDist.Average / $trainTotal.Average) * 100
    
    Write-Host "Ratios - Conv: $($trainConvRatio.ToString('F1'))%, Seq: $($trainSeqRatio.ToString('F1'))%, Dist: $($trainDistRatio.ToString('F1'))%"
    Write-Host ""
}

# Val Analysis  
if ($valData.Count -gt 0) {
    Write-Host "=== VALIDATION Analysis ===" -ForegroundColor Blue
    
    $valConv = ($valData.Conv | Measure-Object -Average -Min -Max)
    $valSeq = ($valData.Seq | Measure-Object -Average -Min -Max)
    $valDist = ($valData.Dist | Measure-Object -Average -Min -Max)
    $valTotal = ($valData.Total | Measure-Object -Average -Min -Max)
    
    Write-Host "Conv Loss - Avg: $($valConv.Average.ToString('F2')), Min: $($valConv.Minimum.ToString('F2')), Max: $($valConv.Maximum.ToString('F2'))"
    Write-Host "Seq Loss  - Avg: $($valSeq.Average.ToString('F2')), Min: $($valSeq.Minimum.ToString('F2')), Max: $($valSeq.Maximum.ToString('F2'))"
    Write-Host "Dist Loss - Avg: $($valDist.Average.ToString('F3')), Min: $($valDist.Minimum.ToString('F3')), Max: $($valDist.Maximum.ToString('F3'))"
    Write-Host "Total     - Avg: $($valTotal.Average.ToString('F2')), Min: $($valTotal.Minimum.ToString('F2')), Max: $($valTotal.Maximum.ToString('F2'))"
    
    # Component ratios for val
    $valConvRatio = ($valConv.Average / $valTotal.Average) * 100
    $valSeqRatio = ($valSeq.Average / $valTotal.Average) * 100
    $valDistRatio = ($valDist.Average / $valTotal.Average) * 100
    
    Write-Host "Ratios - Conv: $($valConvRatio.ToString('F1'))%, Seq: $($valSeqRatio.ToString('F1'))%, Dist: $($valDistRatio.ToString('F1'))%"
    Write-Host ""
}

# Train vs Val Comparison
if ($trainData.Count -gt 0 -and $valData.Count -gt 0) {
    Write-Host "=== TRAIN vs VAL Comparison ===" -ForegroundColor Magenta
    
    $trainValRatio = $trainTotal.Average / $valTotal.Average
    Write-Host "Train/Val Total Loss Ratio: $($trainValRatio.ToString('F1'))"
    
    Write-Host "Component Ratios (Train/Val):"
    Write-Host "  Conv: $($trainConv.Average.ToString('F2')) / $($valConv.Average.ToString('F2')) = $($trainConv.Average / $valConv.Average | ForEach-Object {$_.ToString('F1')})"
    Write-Host "  Seq:  $($trainSeq.Average.ToString('F2')) / $($valSeq.Average.ToString('F2')) = $($trainSeq.Average / $valSeq.Average | ForEach-Object {$_.ToString('F1')})"
    Write-Host "  Dist: $($trainDist.Average.ToString('F3')) / $($valDist.Average.ToString('F3')) = $($trainDist.Average / $valDist.Average | ForEach-Object {$_.ToString('F1')})"
    Write-Host ""
}

# Problem Diagnosis
Write-Host "=== Problem Diagnosis ===" -ForegroundColor Red

if ($trainData.Count -gt 0 -and $valData.Count -gt 0) {
    if ($trainValRatio -gt 10.0) {
        Write-Host "CRITICAL: Severe overfitting detected (Train/Val = $($trainValRatio.ToString('F1')))" -ForegroundColor Red
    } elseif ($trainValRatio -gt 5.0) {
        Write-Host "WARNING: Moderate overfitting detected (Train/Val = $($trainValRatio.ToString('F1')))" -ForegroundColor Yellow
    }
}

if ($trainData.Count -gt 0) {
    if ($trainDist.Average -lt 0.05) {
        Write-Host "CRITICAL: Distillation loss too low (avg = $($trainDist.Average.ToString('F4')))" -ForegroundColor Red
    }
    
    $convSeqRatio = $trainConv.Average / $trainSeq.Average
    if ($convSeqRatio -lt 0.7) {
        Write-Host "WARNING: Conv loss much lower than Seq loss (ratio = $($convSeqRatio.ToString('F2')))" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Analysis complete!" -ForegroundColor Green
