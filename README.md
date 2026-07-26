# ClarityFix

单张图片画质修复工具。基于 ffmpeg 经典滤镜、realesrgan-ncnn-vulkan(细节增强)、
GFPGAN(人脸修复)拼成的开源方案,自用。本地 Mac 版用 Apple Silicon 的 Metal
(经 MoltenVK)加速;云端部署版走纯 PyTorch CPU 兜底路径。

## 无需命令行:网页界面(本地 Mac 版)

桌面 `ClarityFix` 文件夹里双击 **`启动界面.command`**,会自动打开终端窗口跑起服务
并弹出浏览器页面。上传图片、调 8 个滑块 + 人脸修复开关、点「开始修复」,
修复前后对比图直接看,满意就下载。

首次双击如果被 macOS 拦截,右键选"打开"确认一次即可。

## 参数说明

| 参数 | 实现方式 |
|---|---|
| 修复压缩感 | ffmpeg deblock 滤镜,减少压缩块状/马赛克感 |
| 细节增强 | Real-ESRGAN 超分后按比例与原图混合,注入 AI 细节 |
| 锐化 | ffmpeg unsharp,细粒度锐化 |
| 降噪 | ffmpeg hqdn3d 降噪 |
| 去光晕 | ffmpeg smartblur,压制锐化产生的光晕/振铃 |
| 去锯齿/去模糊 | ffmpeg unsharp(较大半径),整体去模糊/去锯齿 |
| 加颗粒 | ffmpeg noise 滤镜,处理完太"塑料感"时加回颗粒 |
| 恢复细节 | ffmpeg unsharp(中等半径),弥补降噪损失的细节 |
| 人脸修复 | GFPGAN 人脸修复模型,单独一次通道,可调混合强度 |

默认值取自常见调参经验,效果因图而异,可以按需调滑块比对。

## 命令行用法(可选,本地 Mac 版)

```bash
cd ~/Desktop/ClarityFix
source venv/bin/activate
python3 clarityfix.py restore 输入.jpg 输出.jpg \
  --fix-compression 90 --improve-detail 70 --sharpen 50 --reduce-noise 10 \
  --dehalo 20 --anti-alias 100 --add-noise 0 --recover-detail 25 \
  --face-restore --face-strength 80
```

## 目录结构
- `models/GFPGANv1.4.pth` — 人脸修复模型权重
- `tools/realesrgan-ncnn-vulkan/` — 细节增强引擎 + 权重(仅本地 Mac 版,云端部署走 PyTorch 兜底)
- `gfpgan/weights/` — 人脸检测/解析辅助模型,首次用人脸修复时自动下载
- `work/` — 临时处理目录,处理时自动清理
- `venv/` — Python 虚拟环境(本地 Mac 版)
- `app.py` — 网页界面
- `clarityfix.py` — 处理逻辑 / 命令行入口
- `启动界面.command` — 双击启动网页界面(本地 Mac 版)
- `requirements.txt` / `packages.txt` / `runtime.txt` — 云端部署(Streamlit Community Cloud)用的依赖清单
