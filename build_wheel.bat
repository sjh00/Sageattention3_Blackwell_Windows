@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem =============================================================================
rem  build_wheel.bat - Build sageattn3 wheel for Windows (Blackwell / RTX 50xx)
rem
rem  Requirements:
rem    - Visual Studio 2026 Build Tools (MSVC) with C++ workload
rem    - CUDA Toolkit >= 12.8 (13.x recommended for RTX 50-series)
rem    - Python with matching torch CUDA build installed
rem    - Git (used once to clone NVIDIA CUTLASS)
rem    - ninja, packaging, wheel, build (pip packages)
rem
rem  Usage:
rem    build_wheel.bat
rem    build_wheel.bat --clean
rem    build_wheel.bat --python "D:\path\to\python.exe"
rem    build_wheel.bat --cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
rem =============================================================================

cd /d "%~dp0"

set "PY_EXE=python"
set "CUDA_HOME_OVERRIDE="
set "DO_CLEAN=0"
set "MAX_JOBS=2"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--clean" (
  set "DO_CLEAN=1"
  shift
  goto parse_args
)
if /I "%~1"=="--python" (
  set "PY_EXE=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--cuda" (
  set "CUDA_HOME_OVERRIDE=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--jobs" (
  set "MAX_JOBS=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--help" (
  call :print_help
  exit /b 0
)
if /I "%~1"=="-h" (
  call :print_help
  exit /b 0
)
echo [ERROR] Unknown argument: %~1
call :print_help
exit /b 1

:print_help
echo.
echo Usage: build_wheel.bat [options]
echo.
echo Options:
echo   --clean                 Remove build\ and dist\ before compiling
echo   --python PATH           Python executable to use
echo   --cuda PATH             CUDA Toolkit root (sets CUDA_HOME)
echo   --jobs N                Parallel compile jobs (default: 2)
echo   --help                  Show this help
echo.
goto :eof

:args_done

echo.
echo ============================================================
echo  SageAttention3 Blackwell - Windows wheel build
echo ============================================================
echo.

rem ----- Locate Visual Studio / MSVC -----
rem Primary: VS 2026 (install path often under ...\18\BuildTools). Fallback: vswhere / VS 2022.
set "VCVARS="
if defined VSINSTALLDIR (
  if exist "%VSINSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=%VSINSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat"
  )
)

rem Prefer VS 2026 by product year folder and by internal major "18".
if not defined VCVARS (
  for %%E in (BuildTools Community Professional Enterprise) do (
    if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\18\%%E\VC\Auxiliary\Build\vcvars64.bat" (
      set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\18\%%E\VC\Auxiliary\Build\vcvars64.bat"
    )
    if exist "%ProgramFiles%\Microsoft Visual Studio\18\%%E\VC\Auxiliary\Build\vcvars64.bat" (
      set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\18\%%E\VC\Auxiliary\Build\vcvars64.bat"
    )
    if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2026\%%E\VC\Auxiliary\Build\vcvars64.bat" (
      set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2026\%%E\VC\Auxiliary\Build\vcvars64.bat"
    )
    if exist "%ProgramFiles%\Microsoft Visual Studio\2026\%%E\VC\Auxiliary\Build\vcvars64.bat" (
      set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2026\%%E\VC\Auxiliary\Build\vcvars64.bat"
    )
  )
)

if not defined VCVARS if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
  for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
  )
)

if not defined VCVARS (
  for %%Y in (2026 18 2025 2022) do (
    for %%E in (BuildTools Community Professional Enterprise) do (
      if exist "%ProgramFiles%\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat"
      )
      if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat"
      )
    )
  )
)

if not defined VCVARS (
  echo [ERROR] Could not find vcvars64.bat.
  echo         Install Visual Studio 2026 Build Tools with "Desktop development with C++".
  exit /b 1
)

echo [INFO] Loading MSVC environment:
echo        %VCVARS%
call "%VCVARS%"
if errorlevel 1 (
  echo [ERROR] vcvars64.bat failed.
  exit /b 1
)

where cl >nul 2>&1
if errorlevel 1 (
  echo [ERROR] cl.exe not found after vcvars64. Check your MSVC install.
  exit /b 1
)

rem ----- Python -----
rem Normalize to an absolute path so nested for /f + quoting stays reliable.
for %%P in ("%PY_EXE%") do set "PY_EXE=%%~fP"
if not exist "%PY_EXE%" (
  where "%PY_EXE%" >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] Python not found: %PY_EXE%
    exit /b 1
  )
)

echo [INFO] Python:
"%PY_EXE%" -c "import sys; print(sys.executable); print(sys.version)"
if errorlevel 1 exit /b 1

rem Put this interpreter's Scripts dir on PATH (ninja.exe from `uv pip install ninja`).
rem Write to a temp file to avoid nested-quote breakage in for /f.
set "PY_SCRIPTS="
"%PY_EXE%" -c "import sysconfig; print(sysconfig.get_path('scripts'))" > "%TEMP%\sageattn3_scripts.txt" 2>nul
if exist "%TEMP%\sageattn3_scripts.txt" (
  set /p PY_SCRIPTS=<"%TEMP%\sageattn3_scripts.txt"
  del "%TEMP%\sageattn3_scripts.txt" >nul 2>&1
)
if defined PY_SCRIPTS (
  set "PATH=%PY_SCRIPTS%;%PATH%"
  echo [INFO] Python Scripts: %PY_SCRIPTS%
)

"%PY_EXE%" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
if errorlevel 1 (
  echo [ERROR] PyTorch is required. Install a CUDA-enabled torch first ^(e.g. uv pip install torch^).
  exit /b 1
)

"%PY_EXE%" -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU not visible to torch'; c=torch.cuda.get_device_capability(); print('GPU capability', c); assert c in ((10,0),(12,0),(12,1)), 'Need Blackwell sm_100 / sm_120 / sm_121, got %%s' %% (c,)"
if errorlevel 1 (
  echo [ERROR] GPU check failed. SageAttention3 targets Blackwell GPUs only.
  exit /b 1
)

rem ----- CUDA Toolkit -----
if defined CUDA_HOME_OVERRIDE set "CUDA_HOME=%CUDA_HOME_OVERRIDE%"

if not defined CUDA_HOME (
  if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
)
if not defined CUDA_HOME (
  if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
)
if not defined CUDA_HOME (
  if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
)
if not defined CUDA_HOME (
  if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
)
if not defined CUDA_HOME (
  if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
)

if not defined CUDA_HOME (
  echo [ERROR] CUDA_HOME not set and no CUDA Toolkit was found under
  echo         "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\".
  echo         Install CUDA Toolkit 12.8+ and re-run, or pass --cuda "PATH".
  exit /b 1
)

if not exist "%CUDA_HOME%\bin\nvcc.exe" (
  echo [ERROR] nvcc not found at: %CUDA_HOME%\bin\nvcc.exe
  exit /b 1
)

set "PATH=%CUDA_HOME%\bin;%PATH%"
echo [INFO] CUDA_HOME=%CUDA_HOME%
"%CUDA_HOME%\bin\nvcc.exe" --version

where git >nul 2>&1
if errorlevel 1 (
  echo [ERROR] git is required to clone NVIDIA CUTLASS on first build.
  exit /b 1
)

rem ----- Python build deps -----
rem uv venvs have no pip module by default; prefer "uv pip" when uv is available.
echo [INFO] Ensuring build dependencies...
where uv >nul 2>&1
if errorlevel 1 (
  echo [WARN] uv not found; falling back to "%PY_EXE% -m pip".
  "%PY_EXE%" -m pip install -U setuptools wheel ninja packaging build
) else (
  echo [INFO] Using: uv pip install --python "%PY_EXE%"
  uv pip install --python "%PY_EXE%" -U setuptools wheel ninja packaging build
)
if errorlevel 1 (
  echo [ERROR] Failed to install Python build dependencies.
  exit /b 1
)

rem ----- Clean optional / stale ABI -----
rem Switching Python (e.g. 3.14t -> 3.13) leaves incompatible build\ tree; wipe when ABI changes.
set "PY_ABI_TAG="
"%PY_EXE%" -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}{getattr(sys, \"abiflags\", \"\")}')" > "%TEMP%\sageattn3_abi.txt" 2>nul
if exist "%TEMP%\sageattn3_abi.txt" (
  set /p PY_ABI_TAG=<"%TEMP%\sageattn3_abi.txt"
  del "%TEMP%\sageattn3_abi.txt" >nul 2>&1
)
set "STALE_ABI=0"
if exist build if defined PY_ABI_TAG (
  dir /b /ad build\lib.win-amd64-cpython-* 2>nul | findstr /I /V "cpython-%PY_ABI_TAG%" >nul 2>&1
  if not errorlevel 1 set "STALE_ABI=1"
  dir /b /ad build\temp.win-amd64-cpython-* 2>nul | findstr /I /V "cpython-%PY_ABI_TAG%" >nul 2>&1
  if not errorlevel 1 set "STALE_ABI=1"
)
if "%DO_CLEAN%"=="1" set "STALE_ABI=1"
if "%STALE_ABI%"=="1" (
  echo [INFO] Cleaning build\ dist\ and egg-info ^(clean/ABI switch^)...
  if exist build rmdir /s /q build
  if exist dist rmdir /s /q dist
  for /d %%D in (*.egg-info) do rmdir /s /q "%%D"
)

rem ----- Build -----
set "DISTUTILS_USE_SDK=1"
set "MAX_JOBS=%MAX_JOBS%"
set "CMAKE_BUILD_PARALLEL_LEVEL=%MAX_JOBS%"
rem Quiet torch/extension chatter that is not actionable on Windows release builds.
set "TORCH_CUDA_ARCH_LIST="

echo.
echo [INFO] Building wheel (this can take a long time)...
echo        Log file: build.log
echo.

"%PY_EXE%" -m build --wheel --no-isolation > build.log 2>&1
if errorlevel 1 (
  echo [ERROR] Build failed. Last 80 lines of build.log:
  echo ------------------------------------------------------------
  powershell -NoProfile -Command "Get-Content -Path 'build.log' -Tail 80"
  echo ------------------------------------------------------------
  exit /b 1
)

rem Scan log for remaining process issues (not just final success).
powershell -NoProfile -Command ^
  "$log = Get-Content -Path 'build.log' -Raw -ErrorAction SilentlyContinue; if (-not $log) { exit 0 };" ^
  "$hard = [regex]::Matches($log, '(?im)^(error:|nvcc fatal|fatal error|FAILED:|undefined reference|LNK[0-9]{4})');" ^
  "$warn = [regex]::Matches($log, '(?im)(warning (D9025|#221|#68)\b|cl:.*warning D9025)');" ^
  "if ($hard.Count -gt 0) { Write-Host ('[ERROR] Hard build issues in log: ' + $hard.Count); $hard | Select-Object -First 10 | ForEach-Object { Write-Host $_.Value }; exit 2 };" ^
  "if ($warn.Count -gt 0) { Write-Host ('[WARN] Residual compile warnings of interest: ' + $warn.Count); $warn | Select-Object -First 10 | ForEach-Object { Write-Host $_.Value }; exit 0 };" ^
  "Write-Host '[OK] No hard errors; key warning classes (D9025/#221/#68) not found in build.log'"
if errorlevel 2 (
  echo [ERROR] Build reported success but log still has hard errors.
  exit /b 1
)

echo.
echo [OK] Build succeeded. Wheels in dist\:
dir /b dist\*.whl 2>nul
echo.
echo Install example (adjust python path for ComfyUI portable):
where uv >nul 2>&1
if errorlevel 1 (
  echo   "%PY_EXE%" -m pip install --force-reinstall --no-deps dist\sageattn3-*.whl
) else (
  echo   uv pip install --python "%PY_EXE%" --force-reinstall --no-deps dist\sageattn3-*.whl
)
echo.
echo Quick smoke test:
echo   "%PY_EXE%" -c "import torch; from sageattn3 import sageattn3_blackwell; q=torch.randn(1,8,128,128,device='cuda',dtype=torch.bfloat16); print(sageattn3_blackwell(q,q,q,per_block_mean=False).shape)"
echo.
exit /b 0
