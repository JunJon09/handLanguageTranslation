# PowerShell Loss Analysis Script
param(
    [string]$LogFile = "D:\Phonenix-2014-translate\handLanguageTranslation\ContinuousSignLanguage\CNN_BiLSTM\logs\training_20250908_164346.log"
)

Write-Host "=== Loss Analysis Script ===" -ForegroundColor Green
Write-Host "Analyzing file: $LogFile" -ForegroundColor Yellow

# ログファイルから損失データを抽出
$lossLines = Get-Content $LogFile | Select-String "損失:"

if ($lossLines.Count -eq 0) {
    Write-Host "No loss data found in the log file." -ForegroundColor Red
    exit
}

Write-Host "Found $($lossLines.Count) loss entries" -ForegroundColor Green

# 損失データを解析
$lossData = @()

foreach ($line in $lossLines) {
    if ($line -match "Conv=(\d+\.?\d*), Seq=(\d+\.?\d*), Dist=(\d+\.?\d*), 総計=(\d+\.?\d*)") {
        $conv = [float]$matches[1]
        $seq = [float]$matches[2]
        $dist = [float]$matches[3]
        $total = [float]$matches[4]
        
        $lossData += [PSCustomObject]@{
            Conv = $conv
            Seq = $seq
            Dist = $dist
            Total = $total
        }
    }
}

if ($lossData.Count -eq 0) {
    Write-Host "Could not parse loss data." -ForegroundColor Red
    exit
}

Write-Host "`n=== Loss Statistics ===" -ForegroundColor Green

# 統計計算
$convStats = $lossData.Conv | Measure-Object -Average -Minimum -Maximum
$seqStats = $lossData.Seq | Measure-Object -Average -Minimum -Maximum  
$distStats = $lossData.Dist | Measure-Object -Average -Minimum -Maximum
$totalStats = $lossData.Total | Measure-Object -Average -Minimum -Maximum

Write-Host "Conv Loss:"
Write-Host "  Average: $($convStats.Average.ToString('F3'))"
Write-Host "  Min: $($convStats.Minimum.ToString('F3'))"
Write-Host "  Max: $($convStats.Maximum.ToString('F3'))"

Write-Host "Seq Loss:"
Write-Host "  Average: $($seqStats.Average.ToString('F3'))"
Write-Host "  Min: $($seqStats.Minimum.ToString('F3'))"
Write-Host "  Max: $($seqStats.Maximum.ToString('F3'))"

Write-Host "Dist Loss:"
Write-Host "  Average: $($distStats.Average.ToString('F3'))"
Write-Host "  Min: $($distStats.Minimum.ToString('F3'))"
Write-Host "  Max: $($distStats.Maximum.ToString('F3'))"

Write-Host "Total Loss:"
Write-Host "  Average: $($totalStats.Average.ToString('F3'))"
Write-Host "  Min: $($totalStats.Minimum.ToString('F3'))"
Write-Host "  Max: $($totalStats.Maximum.ToString('F3'))"

# 最新100エントリの割合分析
Write-Host "`n=== Component Ratios (Latest 100 entries) ===" -ForegroundColor Green

$recent = $lossData | Select-Object -Last 100
$recentAvgTotal = ($recent.Total | Measure-Object -Average).Average

if ($recentAvgTotal -gt 0) {
    $convRatio = (($recent.Conv | Measure-Object -Average).Average / $recentAvgTotal) * 100
    $seqRatio = (($recent.Seq | Measure-Object -Average).Average / $recentAvgTotal) * 100
    $distRatio = (($recent.Dist | Measure-Object -Average).Average / $recentAvgTotal) * 100
    
    Write-Host "Conv: $($convRatio.ToString('F1'))%"
    Write-Host "Seq:  $($seqRatio.ToString('F1'))%"
    Write-Host "Dist: $($distRatio.ToString('F1'))%"
}

# 初期と最終の比較
Write-Host "`n=== Training Progress ===" -ForegroundColor Green

$first10 = $lossData | Select-Object -First 10
$last10 = $lossData | Select-Object -Last 10

$firstAvg = ($first10.Total | Measure-Object -Average).Average
$lastAvg = ($last10.Total | Measure-Object -Average).Average

Write-Host "First 10 entries average total loss: $($firstAvg.ToString('F3'))"
Write-Host "Last 10 entries average total loss: $($lastAvg.ToString('F3'))"
Write-Host "Improvement: $((($firstAvg - $lastAvg) / $firstAvg * 100).ToString('F1'))%"

# 各成分の初期vs最終
$firstConv = ($first10.Conv | Measure-Object -Average).Average
$lastConv = ($last10.Conv | Measure-Object -Average).Average
$firstSeq = ($first10.Seq | Measure-Object -Average).Average
$lastSeq = ($last10.Seq | Measure-Object -Average).Average
$firstDist = ($first10.Dist | Measure-Object -Average).Average
$lastDist = ($last10.Dist | Measure-Object -Average).Average

Write-Host "`nComponent Changes:"
Write-Host "Conv: $($firstConv.ToString('F3')) -> $($lastConv.ToString('F3')) ($((($firstConv - $lastConv) / $firstConv * 100).ToString('F1'))% improvement)"
Write-Host "Seq:  $($firstSeq.ToString('F3')) -> $($lastSeq.ToString('F3')) ($((($firstSeq - $lastSeq) / $firstSeq * 100).ToString('F1'))% improvement)"
Write-Host "Dist: $($firstDist.ToString('F3')) -> $($lastDist.ToString('F3')) ($((($firstDist - $lastDist) / $firstDist * 100).ToString('F1'))% improvement)"

# 異常値検出
Write-Host "`n=== Anomaly Detection ===" -ForegroundColor Yellow

foreach ($component in @('Conv', 'Seq', 'Dist')) {
    $values = $lossData.$component
    $sorted = $values | Sort-Object
    $q1Index = [Math]::Floor($sorted.Count * 0.25)
    $q3Index = [Math]::Floor($sorted.Count * 0.75)
    $q1 = $sorted[$q1Index]
    $q3 = $sorted[$q3Index]
    $iqr = $q3 - $q1
    $outlierThreshold = $q3 + 1.5 * $iqr
    
    $outliers = $values | Where-Object { $_ -gt $outlierThreshold }
    
    if ($outliers.Count -gt 0) {
        Write-Host "${component}: $($outliers.Count) outliers detected (threshold: $($outlierThreshold.ToString('F3')))"
    } else {
        Write-Host "${component}: No significant outliers"
    }
}

Write-Host "`n=== Analysis Complete ===" -ForegroundColor Green
