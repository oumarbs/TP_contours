@echo off
set "CURRENT_DIR=%~dp0"
set "CURRENT_DIR=%CURRENT_DIR:~0,-1%"
for /f "usebackq delims=" %%i in (`wsl wslpath -u "%CURRENT_DIR%"`) do set "WSL_DIR=%%i"

wsl -e bash -c "cd '%WSL_DIR%' && g++ -std=c++17 -O2 main.cpp -o tp_contours -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs && ./tp_contours %*"
