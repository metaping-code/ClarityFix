#!/usr/bin/env python3
"""ClarityFix:单张图片画质修复工具(细节增强 + 人脸修复)。

依赖: ffmpeg, realesrgan-ncnn-vulkan (tools/, 可选), GFPGAN (models/GFPGANv1.4.pth)
用法示例见 README.md
"""
import argparse
import shutil
import subprocess
import sys
import types
import urllib.request
from pathlib import Path

# basicsr 1.4.2 依赖已被新版 torchvision 移除的 functional_tensor 模块,
# 云端全新 pip install 时不会像本地这样手动打过补丁,这里注册一个垫片兼容。
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    import torchvision.transforms.functional as _F
    _shim = types.ModuleType("torchvision.transforms.functional_tensor")
    _shim.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _shim

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
TOOLS = ROOT / "tools"
WORK = ROOT / "work"

GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
REALESRGAN_PT_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)


def ensure_model(url: str, dest: Path):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载模型: {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def find_realesrgan_bin():
    """本地 Mac 版专用的快速引擎;云端部署时不存在,返回 None 走 PyTorch 兜底路径。"""
    cands = list(TOOLS.glob("realesrgan-ncnn-vulkan*/realesrgan-ncnn-vulkan"))
    return cands[0] if cands else None


def get_pytorch_upsampler():
    """没有 realesrgan-ncnn-vulkan 时(比如云端 Linux 主机)的纯 PyTorch 兜底实现,CPU 也能跑。"""
    import torch
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    weight_path = ensure_model(REALESRGAN_PT_URL, MODELS / "realesr-general-x4v3.pth")
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return RealESRGANer(scale=4, model_path=str(weight_path), model=model,
                         tile=0, tile_pad=10, pre_pad=0, half=False, device=device)


# ---------- restore (图片画质修复,仅支持单张图片) ----------

def cmd_restore(args):
    import cv2

    src = Path(args.input).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    filters = []
    if args.reduce_noise > 0:
        s = args.reduce_noise / 100
        filters.append(f"hqdn3d={12*s:.2f}:{9*s:.2f}:{18*s:.2f}:{13.5*s:.2f}")
    if args.fix_compression > 0:
        s = args.fix_compression / 100
        th = 0.45 * s
        filters.append(f"deblock=filter=strong:block=8:alpha={th:.3f}:beta={th:.3f}:gamma={th:.3f}:delta={th:.3f}")
    if args.anti_alias > 0:
        s = args.anti_alias / 100
        filters.append(f"unsharp=9:9:{2.8*s:.2f}:9:9:0")
    if args.recover_detail > 0:
        s = args.recover_detail / 100
        filters.append(f"unsharp=5:5:{2.0*s:.2f}:5:5:0")
    if args.dehalo > 0:
        s = args.dehalo / 100
        filters.append(f"smartblur=lr=2.0:ls={0.9*s:.2f}:lt=6.0:cr=0.0:cs=0.0:ct=0.0")
    if args.sharpen > 0:
        s = args.sharpen / 100
        filters.append(f"unsharp=3:3:{3.5*s:.2f}:3:3:0")
    if args.add_noise > 0:
        s = args.add_noise / 100
        filters.append(f"noise=alls={40*s:.1f}:allf=t+u")

    current = out.with_name("_stage_" + out.name)
    if filters:
        run(["ffmpeg", "-y", "-i", str(src), "-vf", ",".join(filters), str(current)])
    else:
        shutil.copy(src, current)

    if args.improve_detail > 0:
        bin_path = find_realesrgan_bin()
        base = cv2.imread(str(current), cv2.IMREAD_COLOR)
        h, w = base.shape[:2]
        if bin_path is not None:
            models_dir = bin_path.parent / "models"
            sr_out = out.with_name("_sr_" + out.name)
            run([str(bin_path), "-i", str(current), "-o", str(sr_out),
                 "-m", str(models_dir), "-n", "realesr-animevideov3", "-s", "2", "-t", "0"])
            sr = cv2.imread(str(sr_out), cv2.IMREAD_COLOR)
            sr_out.unlink(missing_ok=True)
        else:
            upsampler = get_pytorch_upsampler()
            sr, _ = upsampler.enhance(base, outscale=2)
        sr_down = cv2.resize(sr, (w, h), interpolation=cv2.INTER_LANCZOS4)
        alpha = args.improve_detail / 100
        blended = cv2.addWeighted(sr_down, alpha, base, 1 - alpha, 0)
        cv2.imwrite(str(current), blended)

    if args.face_restore:
        from gfpgan import GFPGANer
        ensure_model(GFPGAN_URL, MODELS / "GFPGANv1.4.pth")
        restorer = GFPGANer(model_path=str(MODELS / "GFPGANv1.4.pth"), upscale=1,
                             arch="clean", channel_multiplier=2, bg_upsampler=None)
        img = cv2.imread(str(current), cv2.IMREAD_COLOR)
        _, _, restored = restorer.enhance(img, has_aligned=False, only_center_face=False,
                                           paste_back=True)
        if restored is not None:
            fs = args.face_strength / 100
            blended = cv2.addWeighted(restored, fs, img, 1 - fs, 0)
            cv2.imwrite(str(current), blended)

    shutil.move(str(current), str(out))
    print(f"done -> {out}")


def main():
    p = argparse.ArgumentParser(description="ClarityFix 画质修复 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    rs = sub.add_parser("restore", help="图片画质修复(仅图片)")
    rs.add_argument("input")
    rs.add_argument("output")
    rs.add_argument("--fix-compression", type=float, default=90)
    rs.add_argument("--improve-detail", type=float, default=70)
    rs.add_argument("--sharpen", type=float, default=50)
    rs.add_argument("--reduce-noise", type=float, default=10)
    rs.add_argument("--dehalo", type=float, default=20)
    rs.add_argument("--anti-alias", type=float, default=100)
    rs.add_argument("--add-noise", type=float, default=0)
    rs.add_argument("--recover-detail", type=float, default=25)
    rs.add_argument("--face-restore", action="store_true", help="启用人脸修复")
    rs.add_argument("--face-strength", type=float, default=80, help="人脸修复效果与原图的混合强度 0~100")
    rs.set_defaults(func=cmd_restore)

    args = p.parse_args()
    WORK.mkdir(exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
