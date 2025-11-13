@echo off
echo === Quick Loss Analysis ===
echo Extracting loss data from log file...

findstr "損失:" "D:\Phonenix-2014-translate\handLanguageTranslation\ContinuousSignLanguage\CNN_BiLSTM\logs\training_20250908_164346.log" > temp_losses.txt

echo.
echo First 5 loss entries:
head -5 temp_losses.txt

echo.
echo Last 5 loss entries:
tail -5 temp_losses.txt

echo.
echo Total loss entries:
find /c "Conv=" temp_losses.txt

echo.
echo Extracting numerical values...
rem Extract just the numerical parts for analysis
findstr "Conv=" temp_losses.txt | findstr /R "Conv=[0-9][0-9]*\.[0-9]*" > temp_conv.txt

if exist temp_conv.txt (
    echo Analysis complete. Check temp_losses.txt for full data.
) else (
    echo Failed to extract loss data.
)

pause
