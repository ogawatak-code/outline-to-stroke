import streamlit as st
import numpy as np
from lxml import etree
import cairosvg
from skimage.morphology import skeletonize
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="アウトライン → ストローク変換",
    page_icon="✏️",
    layout="centered"
)

st.markdown("""
<style>
    .main { max-width: 800px; }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 2.8em;
        font-weight: 500;
    }
    .result-box {
        background: #f8f8f8;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        text-align: center;
    }
    h1 { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("✏️ アウトライン → ストローク変換")
st.caption("Illustratorでアウトライン化された線を、1本のストロークパスに変換してSVGで出力します")

with st.sidebar:
    st.header("⚙️ 変換設定")
    stroke_width = st.slider("出力ストローク幅", 0.1, 5.0, 1.0, 0.1)
    smooth = st.slider("スムージング強度", 1, 20, 6, 1,
                       help="大きいほど滑らかになりますが、細部が失われます")
    min_length = st.slider("最小パス長（px）", 2, 30, 5, 1,
                           help="これより短いパスはノイズとして除去します")
    scale = st.slider("処理解像度（倍率）", 1.0, 4.0, 2.0, 0.5,
                      help="大きいほど精度が上がりますが処理が遅くなります")
    st.divider()
    st.markdown("**Illustratorへ戻す手順**")
    st.markdown("1. SVGをダウンロード\n2. Illustratorで開く\n3. 別名で保存 → `.ai` 形式")


def svg_to_binary(svg_bytes, scale):
    tree = etree.fromstring(svg_bytes)
    vb = tree.get('viewBox')
    if vb:
        parts = vb.strip().split()
        svg_w, svg_h = float(parts[2]), float(parts[3])
    else:
        svg_w = float(tree.get('width', '400').replace('px','').replace('pt',''))
        svg_h = float(tree.get('height', '300').replace('px','').replace('pt',''))

    out_w = int(svg_w * scale)
    out_h = int(svg_h * scale)
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=out_w, output_height=out_h)
    img = Image.open(BytesIO(png_bytes)).convert('RGBA')
    arr = np.array(img)
    r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
    lum = 0.299*r + 0.587*g + 0.114*b
    binary = ((a > 40) & (lum < 180)).astype(np.uint8)
    return binary, svg_w, svg_h


def trace_paths(skel, smooth, min_length):
    h, w = skel.shape
    visited = np.zeros((h, w), dtype=bool)

    def neighbors(x, y):
        result = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h and skel[ny, nx] and not visited[ny, nx]:
                    result.append((nx, ny))
        return result

    def degree(x, y):
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h and skel[ny, nx]:
                    count += 1
        return count

    endpoints = [(x, y) for y in range(h) for x in range(w)
                 if skel[y, x] and degree(x, y) == 1]
    starts = endpoints if endpoints else [(x, y) for y in range(h) for x in range(w) if skel[y, x]]

    paths = []
    step = max(1, smooth // 2)

    for (sx, sy) in starts:
        if visited[sy, sx]:
            continue
        path = [(sx, sy)]
        visited[sy, sx] = True
        cur = (sx, sy)
        while True:
            ns = neighbors(cur[0], cur[1])
            if not ns:
                break
            visited[ns[0][1], ns[0][0]] = True
            path.append(ns[0])
            cur = ns[0]
        if len(path) < min_length:
            continue
        simplified = [path[0]]
        for i in range(step, len(path)-1, step):
            simplified.append(path[i])
        simplified.append(path[-1])
        paths.append(simplified)
    return paths


def build_svg(paths, svg_w, svg_h, stroke_width, scale):
    path_els = []
    for path in paths:
        pts = [(x/scale, y/scale) for x, y in path]
        if len(pts) < 2:
            continue
        d = f'M{pts[0][0]:.3f},{pts[0][1]:.3f}'
        for i in range(1, len(pts)):
            if i < len(pts)-1:
                mx = (pts[i][0] + pts[i+1][0]) / 2
                my = (pts[i][1] + pts[i+1][1]) / 2
                d += f' Q{pts[i][0]:.3f},{pts[i][1]:.3f} {mx:.3f},{my:.3f}'
            else:
                d += f' L{pts[i][0]:.3f},{pts[i][1]:.3f}'
        path_els.append(
            f'  <path d="{d}" fill="none" stroke="#000000" '
            f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    paths_str = '\n'.join(path_els)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
  version="1.1" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}px" height="{svg_h}px"
  xml:space="preserve">
  <desc>Stroke paths converted from outline. Open in Adobe Illustrator and save as .ai</desc>
{paths_str}
</svg>'''


uploaded = st.file_uploader("SVGファイルをアップロード", type=["svg"],
                             help="IllustratorからSVGとして書き出したファイルを選択してください")

if uploaded:
    svg_bytes = uploaded.read()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**入力プレビュー**")
        st.components.v1.html(svg_bytes.decode('utf-8'), height=300, scrolling=True)

    if st.button("▶ 変換実行", type="primary"):
        with st.spinner("処理中..."):
            try:
                binary, svg_w, svg_h = svg_to_binary(svg_bytes, scale)
                skel = skeletonize(binary > 0).astype(np.uint8)
                paths = trace_paths(skel, smooth, min_length)
                output_svg = build_svg(paths, svg_w, svg_h, stroke_width, scale)

                with col2:
                    st.markdown("**出力プレビュー**")
                    st.components.v1.html(output_svg, height=300, scrolling=True)

                st.success(f"変換完了！ {len(paths)} 本のストロークパスを生成しました")

                st.download_button(
                    label="⬇️ SVGをダウンロード（Illustratorで開いて .ai 保存）",
                    data=output_svg.encode("utf-8"),
                    file_name="stroke_converted.svg",
                    mime="image/svg+xml",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
else:
    st.info("👆 SVGファイルをアップロードしてください")
