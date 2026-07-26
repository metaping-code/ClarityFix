"""ClarityFix 网页界面:单张图片的画质修复(细节增强 + 人脸修复)。"""
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
PY = sys.executable
SCRIPT = ROOT / "clarityfix.py"
RUN_DIR = ROOT / "work" / "ui_run"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

st.set_page_config(page_title="ClarityFix", page_icon="🖼️", layout="centered")
st.title("🖼️ ClarityFix 图片画质修复")
st.caption("AI 细节增强 + 人脸修复,仅支持单张图片")


def run_with_log(cmd, label):
    with st.status(f"正在执行:{label}", expanded=True) as status:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            status.write(line.rstrip())
        ret = proc.wait()
        if ret != 0:
            status.update(label=f"❌ {label} 失败", state="error")
            st.error(f"{label} 处理失败,退出码 {ret},详情见上方日志")
            st.stop()
        status.update(label=f"✅ {label} 完成", state="complete")


uploaded = st.file_uploader("上传图片", type=list(v.strip(".") for v in IMAGE_EXT))

if uploaded is not None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    src_path = RUN_DIR / ("restore_input" + Path(uploaded.name).suffix.lower())
    src_path.write_bytes(uploaded.getvalue())

    st.image(str(src_path), caption="原图")

    st.subheader("修复参数")
    c1, c2 = st.columns(2)
    with c1:
        fix_compression = st.slider("修复压缩感", 0, 100, 90)
        improve_detail = st.slider("细节增强(AI超分混合)", 0, 100, 70)
        sharpen = st.slider("锐化", 0, 100, 50)
        reduce_noise = st.slider("降噪", 0, 100, 10)
    with c2:
        dehalo = st.slider("去光晕", 0, 100, 20)
        anti_alias = st.slider("去锯齿/去模糊", 0, 100, 100)
        add_noise = st.slider("加颗粒", 0, 100, 0)
        recover_detail = st.slider("恢复细节", 0, 100, 25)

    st.subheader("人脸修复")
    face_restore = st.checkbox("启用人脸修复", value=True)
    face_strength = st.slider("人脸修复强度", 0, 100, 80, disabled=not face_restore)

    if st.button("🚀 开始修复", type="primary"):
        out_path = RUN_DIR / ("restored_" + src_path.stem + ".jpg")
        cmd = [str(PY), str(SCRIPT), "restore", str(src_path), str(out_path),
               "--fix-compression", str(fix_compression),
               "--improve-detail", str(improve_detail),
               "--sharpen", str(sharpen),
               "--reduce-noise", str(reduce_noise),
               "--dehalo", str(dehalo),
               "--anti-alias", str(anti_alias),
               "--add-noise", str(add_noise),
               "--recover-detail", str(recover_detail),
               "--face-strength", str(face_strength)]
        if face_restore:
            cmd.append("--face-restore")

        run_with_log(cmd, "画质修复")

        st.success("修复完成!")
        col_before, col_after = st.columns(2)
        with col_before:
            st.image(str(src_path), caption="修复前")
        with col_after:
            try:
                st.image(str(out_path), caption="修复后")
            except Exception as e:
                st.warning(f"预览失败({e}),但文件已生成,可以直接下载")

        st.download_button("⬇️ 下载修复结果", data=out_path.read_bytes(),
                            file_name=f"restored_{uploaded.name}")
else:
    st.info("👆 先上传一张图片")
