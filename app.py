"""ClarityFix 网页界面:单张图片的画质修复(细节增强 + 人脸修复)。
拖动滑块会用缩小的预览图实时(模型已常驻缓存)展示效果,满意后再对
原图跑一次完整分辨率处理。"""
from pathlib import Path

import streamlit as st
from PIL import Image

import clarityfix

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "work" / "ui_run"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PREVIEW_SIZE = 500

st.set_page_config(page_title="ClarityFix", page_icon="🖼️", layout="centered")
st.title("🖼️ ClarityFix 图片画质修复")
st.caption("AI 细节增强 + 人脸修复,仅支持单张图片")


@st.cache_resource(show_spinner="首次加载细节增强模型...")
def load_upsampler():
    """本地 Mac 有 ncnn 快速引擎时不需要常驻 PyTorch 模型,返回 None。"""
    if clarityfix.find_realesrgan_bin() is not None:
        return None
    return clarityfix.get_pytorch_upsampler()


@st.cache_resource(show_spinner="首次加载人脸修复模型...")
def load_face_restorer():
    return clarityfix.get_gfpgan_restorer()


uploaded = st.file_uploader("上传图片", type=list(v.strip(".") for v in IMAGE_EXT))

if uploaded is not None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    src_path = RUN_DIR / ("restore_input" + Path(uploaded.name).suffix.lower())
    src_path.write_bytes(uploaded.getvalue())

    preview_src = RUN_DIR / "preview_input.jpg"
    im = Image.open(src_path).convert("RGB")
    im.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
    im.save(preview_src)

    st.image(str(src_path), caption="原图")

    st.subheader("修复参数(拖动滑块自动更新下方预览)")
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

    params = dict(
        fix_compression=fix_compression, improve_detail=improve_detail,
        sharpen=sharpen, reduce_noise=reduce_noise, dehalo=dehalo,
        anti_alias=anti_alias, add_noise=add_noise, recover_detail=recover_detail,
        face_restore=face_restore, face_strength=face_strength,
    )

    upsampler = load_upsampler() if improve_detail > 0 else None
    face_restorer = load_face_restorer() if face_restore else None

    st.subheader("预览(缩小图,松开滑块后自动刷新)")
    preview_out = RUN_DIR / "preview_out.jpg"
    with st.spinner("预览生成中..."):
        clarityfix.restore_image(preview_src, preview_out, params,
                                  upsampler=upsampler, face_restorer=face_restorer)
    st.image(str(preview_out))

    st.divider()
    if st.button("🚀 生成完整分辨率结果", type="primary"):
        out_path = RUN_DIR / ("restored_" + src_path.stem + ".jpg")
        with st.spinner("正在处理完整分辨率..."):
            clarityfix.restore_image(src_path, out_path, params,
                                      upsampler=upsampler, face_restorer=face_restorer)

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
