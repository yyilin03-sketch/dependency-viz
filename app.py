import streamlit as st
import json
import streamlit.components.v1 as components

# ================= 1. 页面配置与 CSS 魔法 =================
st.set_page_config(
    page_title="依存句法分析",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# 引入 Google Fonts 和自定义 CSS
st.markdown("""
<style>
    /* 引入字体：Inter (UI) 和 Noto Serif SC (标题) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Serif+SC:wght@500;700&display=swap');

    /* 全局背景与字体 */
    .stApp {
        background-color: #f3f4f6; /* 极淡的灰背景，更有质感 */
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* 标题区域 */
    .header-container {
        text-align: center;
        padding: 40px 0 30px 0;
        background: linear-gradient(180deg, #ffffff 0%, #f3f4f6 100%);
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 30px;
    }
    .main-title {
        font-family: 'Noto Serif SC', serif;
        font-size: 2.8rem;
        color: #111827;
        font-weight: 700;
        margin-bottom: 10px;
        letter-spacing: -0.02em;
    }
    .sub-title {
        color: #6b7280;
        font-size: 1.1rem;
        font-weight: 400;
    }

    /* 卡片通用样式 (核心美化) */
    .card-container {
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        padding: 30px;
        margin-bottom: 24px;
        border: 1px solid #f3f4f6;
    }

    /* 输入区美化 */
    .input-wrapper {
        display: flex;
        gap: 15px;
        align-items: center;
        background: #f9fafb;
        padding: 10px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }

    /* 侧边栏美化 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    /* 隐藏 Streamlit 自带的页脚和汉堡菜单 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# ================= 2. 模型加载逻辑 (缓存) =================

@st.cache_resource
def load_hanlp():
    from hanlp_restful import HanLPClient
    # 替换为你的 Auth 或保持 None
    return HanLPClient('https://www.hanlp.com/api', auth=None, language='zh')


@st.cache_resource
def load_spacy():
    import spacy
    try:
        return spacy.load("zh_core_web_sm")
    except:
        st.error("请先安装模型: python -m spacy download zh_core_web_sm")
        return None


@st.cache_resource
def load_stanza():
    import stanza
    stanza.download('zh', verbose=False)
    return stanza.Pipeline('zh', processors='tokenize,pos,lemma,depparse', verbose=False)


# ================= 3. 分析逻辑适配器 =================

def analyze_hanlp(text, client):
    try:
        doc = client(text, tasks='dep')
        tokens = doc.get('tok/fine', doc.get('tok', []))
        pos = doc.get('pos/ctb', doc.get('pos/pku', doc.get('pos', [])))
        dep = doc.get('dep', [])
        if tokens and isinstance(tokens[0], list): tokens = tokens[0]
        if pos and isinstance(pos[0], list): pos = pos[0]
        if dep and isinstance(dep[0], list): dep = dep[0]
        data = []
        for i in range(len(tokens)):
            data.append({"id": i + 1, "text": tokens[i], "pos": pos[i] if i < len(pos) else "X", "head": dep[i][0],
                         "rel": dep[i][1], "out_degree": 0})
        return data
    except Exception as e:
        st.error(f"HanLP Error: {e}"); return None


def analyze_spacy(text, nlp):
    try:
        doc = nlp(text)
        data = []
        for token in doc:
            head_id = token.head.i + 1
            if token.head.i == token.i: head_id = 0
            data.append({"id": token.i + 1, "text": token.text, "pos": token.pos_, "head": head_id, "rel": token.dep_,
                         "out_degree": 0})
        return data
    except Exception as e:
        st.error(f"spaCy Error: {e}"); return None


def analyze_stanza(text, nlp):
    try:
        doc = nlp(text)
        sent = doc.sentences[0]
        data = []
        for word in sent.words:
            data.append({"id": word.id, "text": word.text, "pos": word.upos, "head": word.head, "rel": word.deprel,
                         "out_degree": 0})
        return data
    except Exception as e:
        st.error(f"Stanza Error: {e}"); return None


# ================= 4. 界面布局构建 =================

# 顶部 Hero 区域
st.markdown("""
<div class="header-container">
    <div class="main-title">Syntax Analysis</div>
    <div class="sub-title">Multi-model Dependency Parsing & Quantitative Linguistics Lab</div>
</div>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "选择分析模型 / Parser Model",
        ["HanLP (Cloud API)", "Stanza (Stanford)", "spaCy (Industrial)"]
    )

    st.markdown("---")
    st.markdown("### 💡About model")
    st.info("""
    **HanLP**: 擅长 CTB 标准的中文处理。
    **Stanza**: 基于 UD，本地部署。
    **spaCy**: 速度较快，本地部署。
    """)
    st.markdown("<div style='text-align:center; color:#9ca3af; font-size:12px; margin-top:50px;'>Designed by Ivy Yang</div>",
                unsafe_allow_html=True)

# 主内容区 (居中布局)
col_spacer1, col_main, col_spacer2 = st.columns([1, 8, 1])

with col_main:
    # 1. 输入卡片
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    c1, c2 = st.columns([5, 1], gap="medium")
    with c1:
        sentence_input = st.text_input("Input Sentence", value="我今天中午想吃拉面", label_visibility="collapsed",
                                       placeholder="请输入要分析的中文句子...")
    with c2:
        # 使用 use_container_width 让按钮填满列宽，对齐更美观
        run_button = st.button("开始分析", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. 结果展示区
    if run_button or sentence_input:
        data = None
        # 显示加载状态
        with st.status(f"正在调用 {model_choice} 进行计算...", expanded=True) as status:
            if "HanLP" in model_choice:
                nlp = load_hanlp()
                data = analyze_hanlp(sentence_input, nlp)
            elif "spaCy" in model_choice:
                nlp = load_spacy()
                if nlp: data = analyze_spacy(sentence_input, nlp)
            elif "Stanza" in model_choice:
                nlp = load_stanza()
                data = analyze_stanza(sentence_input, nlp)
            status.update(label="分析完成!", state="complete", expanded=False)

        if data:
            # 计算出度
            for word in data:
                if word['head'] > 0 and word['head'] <= len(data):
                    data[word['head'] - 1]['out_degree'] += 1

            json_data = json.dumps(data)

            # 使用卡片包裹结果
            st.markdown('<div class="card-container" style="padding: 0;">', unsafe_allow_html=True)

            # 嵌入 HTML
            components.html(f"""
            <!DOCTYPE html>
            <html>
            <head>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Noto+Serif+SC:wght@500&display=swap');

                body {{ font-family: 'Inter', sans-serif; margin: 0; padding: 30px; background: white; }}

                /* 画布区域 */
                .viz-wrapper {{
                    width: 100%; min-width: 800px; height: 360px; 
                    position: relative; margin: 0 auto 30px auto;
                    user-select: none;
                }}

                /* 单词块 */
                .words-row {{ 
                    display: flex; justify-content: space-between; align-items: flex-end; 
                    padding: 0 50px; position: absolute; bottom: 0; width: 100%; box-sizing: border-box; height: 60px; 
                }}
                .word-block {{ display: flex; flex-direction: column; align-items: center; min-width: 40px; cursor: pointer; z-index: 10; }}
                .word-text {{ font-size: 22px; color: #111827; margin-bottom: 6px; font-family: 'Noto Serif SC', serif; font-weight: 500; }}
                .word-pos {{ 
                    font-size: 11px; color: #6b7280; background: #f3f4f6; 
                    border: 1px solid #e5e7eb; padding: 2px 8px; border-radius: 12px; 
                    font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
                }}

                /* 统计面板 (KPI Cards) */
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    margin-bottom: 30px;
                    padding: 0 20px;
                }}
                .kpi-card {{
                    background: #f9fafb;
                    border: 1px solid #e5e7eb;
                    border-radius: 12px;
                    padding: 15px;
                    text-align: center;
                    transition: transform 0.2s;
                }}
                .kpi-card:hover {{ transform: translateY(-2px); border-color: #d1d5db; }}
                .kpi-val {{ font-size: 28px; font-weight: 700; color: #111827; display: block; margin-bottom: 4px; }}
                .kpi-label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }}

                /* 表格样式 */
                .table-wrapper {{ overflow-x: auto; padding: 0 20px; }}
                table {{ width: 100%; border-collapse: collapse; font-size: 14px; color: #374151; }}
                thead {{ background: #f9fafb; border-bottom: 2px solid #e5e7eb; }}
                th {{ text-align: left; padding: 12px 16px; font-weight: 600; color: #4b5563; text-transform: uppercase; font-size: 12px; }}
                td {{ padding: 12px 16px; border-bottom: 1px solid #f3f4f6; }}
                tr:last-child td {{ border-bottom: none; }}
                tr:hover td {{ background: #fdfdfd; }}

                /* SVG 线条 */
                svg {{ width: 100%; height: 100%; position: absolute; top: 0; left: 0; pointer-events: none; }}
                path {{ fill: none; stroke: #6b7280; stroke-width: 1.5px; pointer-events: stroke; transition: all 0.3s; }}
                .root-arc {{ stroke-dasharray: 4, 4; stroke: #9ca3af; }}
                text.dep-label {{ 
                    font-size: 12px; fill: #374151; text-anchor: middle; font-family: 'Inter', sans-serif; 
                    pointer-events: all; cursor: pointer; font-weight: 500;
                    paint-order: stroke; stroke: white; stroke-width: 5px; stroke-linecap: round; stroke-linejoin: round;
                }}

                /* 交互高亮 (Academic Blue) */
                .hover-mode .word-block, .hover-mode path, .hover-mode text {{ opacity: 0.1; transition: opacity 0.2s; }}
                .hover-mode .highlighted {{ opacity: 1 !important; }}
                .hover-mode path.highlighted {{ stroke: #2563eb; stroke-width: 2.5px; }}
                .hover-mode text.highlighted {{ fill: #2563eb; font-weight: 700; }}
                .hover-mode .word-text.highlighted {{ color: #2563eb; }}

            </style>
            </head>
            <body>

                <div class="viz-wrapper" id="canvas-area">
                    <div class="words-row" id="words-row"></div>
                </div>

                <div id="stats-section"></div>

                <script>
                    const data = {json_data};
                    const container = document.getElementById('canvas-area');
                    const wordsRow = document.getElementById('words-row');

                    function render() {{
                        // 1. 生成单词DOM
                        const wordEls = [];
                        data.forEach(w => {{
                            const el = document.createElement('div'); el.className = 'word-block'; el.id = 'w-'+w.id;
                            el.innerHTML = `<div class="word-text">${{w.text}}</div><div class="word-pos">${{w.pos}}</div>`;
                            el.onmouseenter = () => highlight(w.id); el.onmouseleave = clearH;
                            wordsRow.appendChild(el); wordEls.push(el);
                        }});

                        // 2. 延迟绘图
                        setTimeout(() => {{
                            const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
                            // 箭头定义
                            const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
                            const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
                            marker.setAttribute("id", "arrow"); marker.setAttribute("markerWidth", "10"); marker.setAttribute("markerHeight", "10");
                            marker.setAttribute("refX", "8"); marker.setAttribute("refY", "3"); marker.setAttribute("orient", "auto");
                            const mPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
                            mPath.setAttribute("d", "M0,0 L0,6 L9,3 z"); mPath.setAttribute("fill", "#6b7280");
                            marker.appendChild(mPath); defs.appendChild(marker); svg.appendChild(defs);
                            container.appendChild(svg);

                            // 计算坐标
                            const rect = container.getBoundingClientRect();
                            const centers = {{}};
                            wordEls.forEach((el, i) => {{
                                const r = el.getBoundingClientRect();
                                centers[data[i].id] = {{ x: r.left + r.width/2 - rect.left, y: 300 }};
                            }});

                            // 绘制
                            let tdd = 0, n = 0;
                            data.forEach(w => {{
                                if (w.head === 0) {{
                                    const pos = centers[w.id];
                                    drawPath(svg, `M${{pos.x}},${{pos.y}} V${{pos.y - 70}}`, 'root-arc', w.id, 0);
                                    drawLabel(svg, pos.x, pos.y - 80, 'ROOT', w.id, 0);
                                    return;
                                }}
                                const s = centers[w.head], e = centers[w.id];
                                const dist = Math.abs(w.head - w.id); tdd += dist; n++;
                                const h = 30 + (dist * 12); const cpY = 300 - h * 1.5;
                                drawPath(svg, `M${{s.x}},${{s.y}} C${{s.x}},${{cpY}} ${{e.x}},${{cpY}} ${{e.x}},${{e.y}}`, '', w.id, w.head);
                                drawLabel(svg, (s.x + e.x)/2, 300 - h * 1.15, w.rel, w.id, w.head);
                            }});

                            renderDashboard(data, tdd, n);
                        }}, 50);
                    }}

                    function drawPath(svg, d, cls, dep, head) {{
                        const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
                        p.setAttribute("d", d); if(cls) p.setAttribute("class", cls);
                        if(head!==0) p.setAttribute("marker-end", "url(#arrow)");
                        p.dataset.h = head; p.dataset.d = dep;
                        p.onmouseenter = () => highlightArc(head, dep); p.onmouseleave = clearH;
                        svg.appendChild(p);
                    }}
                    function drawLabel(svg, x, y, txt, dep, head) {{
                        const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
                        t.setAttribute("x", x); t.setAttribute("y", y); t.textContent = txt; t.className = 'dep-label';
                        if(head!==0) {{ t.onmouseenter = () => highlightArc(head, dep); t.onmouseleave = clearH; }}
                        svg.appendChild(t);
                    }}

                    function renderDashboard(data, tdd, n) {{
                        const mdd = n ? (tdd/n).toFixed(2) : "0.00";
                        const statsDiv = document.getElementById('stats-section');

                        let html = `
                        <div class="dashboard-grid">
                            <div class="kpi-card"><span class="kpi-val">${{tdd}}</span><span class="kpi-label">Total Distance (TDD)</span></div>
                            <div class="kpi-card"><span class="kpi-val">${{n}}</span><span class="kpi-label">Relations (n)</span></div>
                            <div class="kpi-card"><span class="kpi-val" style="color:#2563eb">${{mdd}}</span><span class="kpi-label">Mean Distance (MDD)</span></div>
                        </div>
                        <div class="table-wrapper">
                            <table>
                                <thead><tr><th>Word</th><th>Pos Tag</th><th>Relation</th><th>Head</th><th>Distance</th><th>Out-Degree</th></tr></thead>
                                <tbody>`;

                        data.forEach(w => {{
                            const hTxt = w.head===0 ? "ROOT" : data.find(x=>x.id===w.head).text;
                            const dist = w.head===0 ? "-" : Math.abs(w.head-w.id);
                            html += `<tr>
                                <td style="font-weight:600">${{w.text}}</td>
                                <td><span style="background:#eff6ff;color:#1e40af;padding:2px 6px;border-radius:4px;font-size:11px;font-weight:600">${{w.pos}}</span></td>
                                <td>${{w.rel}}</td>
                                <td>${{hTxt}}</td>
                                <td>${{dist}}</td>
                                <td>${{w.out_degree}}</td>
                            </tr>`;
                        }});
                        html += `</tbody></table></div>`;
                        statsDiv.innerHTML = html;
                    }}

                    function highlight(id) {{
                        document.body.classList.add('hover-mode'); document.getElementById('w-'+id).classList.add('highlighted');
                        document.querySelectorAll('path').forEach(p => {{
                            if(p.dataset.h==id || p.dataset.d==id) {{
                                p.classList.add('highlighted');
                                if(p.dataset.h!=0) document.getElementById('w-'+p.dataset.h).classList.add('highlighted');
                                document.getElementById('w-'+p.dataset.d).classList.add('highlighted');
                            }}
                        }});
                    }}
                    function highlightArc(h, d) {{
                        document.body.classList.add('hover-mode');
                        const p = document.querySelector(`path[data-h="${{h}}"][data-d="${{d}}"]`); if(p) p.classList.add('highlighted');
                        if(h!=0) document.getElementById('w-'+h).classList.add('highlighted');
                        document.getElementById('w-'+d).classList.add('highlighted');
                    }}
                    function clearH() {{ document.body.classList.remove('hover-mode'); document.querySelectorAll('.highlighted').forEach(e => e.classList.remove('highlighted')); }}

                    render();
                </script>
            </body>
            </html>
            """, height=850, scrolling=True)

            st.markdown('</div>', unsafe_allow_html=True)