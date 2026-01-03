@echo off
echo Installing requirements...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 exit /b %errorlevel%

echo Uninstalling CPU torch...
.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
if %errorlevel% neq 0 exit /b %errorlevel%

echo Installing CUDA torch...
.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 exit /b %errorlevel%

echo Installing fixes...
.venv\Scripts\python.exe -m pip install "numpy<2" "portalocker>=2.0.0"
if %errorlevel% neq 0 exit /b %errorlevel%

echo Setup complete.
