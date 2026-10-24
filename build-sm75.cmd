@echo off
cd /d "%~dp0"
rem Local dev build of stable-diffusion.cpp for Spellcast3D experiments: the official cuda12 release's flags, compute_75 only (the Turing kernel paths; runs on newer cards through the PTX JIT with GGML_CUDA_FAKE_CC=750),
rem no WebP/WebM (Spellcast3D's launcher reads/writes pictures itself).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;C:\_myDrive\repos\spellcast3d\code\venvs\pixal3d\Scripts;%PATH%"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist build-sm75\CMakeCache.txt (
  "%CMAKE%" -S . -B build-sm75 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DSD_BUILD_SHARED_LIBS=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS="-Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=550" ^
    -DGGML_NATIVE=OFF -DSD_BUILD_SHARED_GGML_LIB=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON ^
    -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_BUILD_EXAMPLES=ON || exit /b 1
)
"%CMAKE%" --build build-sm75 --config Release -j 12 || exit /b 1
echo BUILD OK
