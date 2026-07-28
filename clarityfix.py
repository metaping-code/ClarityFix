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
    """没有 realesrgan-ncnn-vulkan 时(比如云端 Linux 主机)的纯 PyTorch 兜底实现,CPU 也能跑。
    加载耗时(读权重+初始化网络),建议调用方缓存复用。"""
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


def get_gfpgan_restorer():
    """加载 GFPGAN 人脸修复模型(耗时操作:读取约330MB权重+初始化网络)。
    建议调用方缓存复用,而不是每次处理都重新调用。"""
    from gfpgan import GFPGANer
    ensure_model(GFPGAN_URL, MODELS / "GFPGANv1.4.pth")
    return GFPGANer(model_path=str(MODELS / "GFPGANv1.4.pth"), upscale=1,
                     arch="clean", channel_multiplier=2, bg_upsampler=None)


# ---------- restore (图片画质修复,仅支持单张图片) ----------

DEFAULT_PARAMS = dict(
    fix_compression=90, improve_detail=70, sharpen=50, reduce_noise=10,
    dehalo=20, anti_alias=100, add_noise=0, recover_detail=25,
    face_restore=False, face_strength=80,
)


def restore_image(src, out, params=None, upsampler=None, face_restorer=None):
    """执行画质修复流水线。

    params: 覆盖 DEFAULT_PARAMS 的参数字典。
    upsampler / face_restorer: 可选的预加载模型实例(用于预览等需要重复调用的
        场景,避免每次都重新加载模型);不传时按需临时创建,适合一次性 CLI 调用。
    """
    import cv2

    p = {**DEFAULT_PARAMS, **(params or {})}
    src = Path(src).resolve()
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # 第一步:降噪/去块/去锯齿/恢复细节/去光晕 —— 在 AI 步骤之前做,避免
    # AI 模型把压缩伪影/噪点误判成"细节"并放大。
    pre_filters = []
    if p["reduce_noise"] > 0:
        s = p["reduce_noise"] / 100
        pre_filters.append(f"hqdn3d={4*s:.2f}:{3*s:.2f}:{6*s:.2f}:{4.5*s:.2f}")
    if p["fix_compression"] > 0:
        s = p["fix_compression"] / 100
        th = 0.3 * s
        pre_filters.append(f"deblock=filter=strong:block=8:alpha={th:.3f}:beta={th:.3f}:gamma={th:.3f}:delta={th:.3f}")
    if p["anti_alias"] > 0:
        s = p["anti_alias"] / 100
        pre_filters.append(f"unsharp=9:9:{1.6*s:.2f}:9:9:0")
    if p["recover_detail"] > 0:
        s = p["recover_detail"] / 100
        pre_filters.append(f"unsharp=5:5:{1.3*s:.2f}:5:5:0")
    if p["dehalo"] > 0:
        s = p["dehalo"] / 100
        pre_filters.append(f"smartblur=lr=2.0:ls={0.9*s:.2f}:lt=6.0:cr=0.0:cs=0.0:ct=0.0")

    current = out.with_name("_stage_" + out.name)
    if pre_filters:
        run(["ffmpeg", "-y", "-i", str(src), "-vf", ",".join(pre_filters), "-update", "1", str(current)])
    else:
        shutil.copy(src, current)

    # 第二步:AI 细节增强(超分)
    if p["improve_detail"] > 0:
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
            up = upsampler or get_pytorch_upsampler()
            sr, _ = up.enhance(base, outscale=2)
        sr_down = cv2.resize(sr, (w, h), interpolation=cv2.INTER_LANCZOS4)
        alpha = p["improve_detail"] / 100
        blended = cv2.addWeighted(sr_down, alpha, base, 1 - alpha, 0)
        cv2.imwrite(str(current), blended)

    # 第三步:锐化 —— 放在超分之后、人脸修复之前。这样能让超分产生的细节
    # 保留清晰度,同时不会对着 GFPGAN 磨光滑的人脸再做一遍锐化(会把假脸的
    # 边缘描得更明显、更塑料)。
    if p["sharpen"] > 0:
        s = p["sharpen"] / 100
        sharp_out = out.with_name("_sharp_" + out.name)
        run(["ffmpeg", "-y", "-i", str(current), "-vf", f"unsharp=3:3:{2.2*s:.2f}:3:3:0",
             "-update", "1", str(sharp_out)])
        sharp_out.replace(current)

    # 第四步:人脸修复(放最后,不再被后续滤镜处理)
    if p["face_restore"]:
        restorer = face_restorer or get_gfpgan_restorer()
        img = cv2.imread(str(current), cv2.IMREAD_COLOR)
        _, _, restored = restorer.enhance(img, has_aligned=False, only_center_face=False,
                                           paste_back=True)
        if restored is not None:
            fs = p["face_strength"] / 100
            blended = cv2.addWeighted(restored, fs, img, 1 - fs, 0)
            cv2.imwrite(str(current), blended)

    # 第五步:加颗粒 —— 放最后,顺带能盖住人脸修复过于光滑的塑料感。
    if p["add_noise"] > 0:
        s = p["add_noise"] / 100
        run(["ffmpeg", "-y", "-i", str(current), "-vf", f"noise=alls={30*s:.1f}:allf=t+u",
             "-update", "1", str(out)])
    else:
        shutil.move(str(current), str(out))
    print(f"done -> {out}")


def cmd_restore(args):
    params = dict(
        fix_compression=args.fix_compression, improve_detail=args.improve_detail,
        sharpen=args.sharpen, reduce_noise=args.reduce_noise, dehalo=args.dehalo,
        anti_alias=args.anti_alias, add_noise=args.add_noise,
        recover_detail=args.recover_detail, face_restore=args.face_restore,
        face_strength=args.face_strength,
    )
    restore_image(args.input, args.output, params)


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
