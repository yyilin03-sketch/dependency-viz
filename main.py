import sys
import json
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QComboBox, QMessageBox, QProgressBar, QFrame,QListView)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QIcon


# ================= 1. 后端分析线程 (保持不变) =================
class AnalysisThread(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, text, model_name, hanlp_key):
        super().__init__()
        self.text = text
        self.model_name = model_name
        self.hanlp_key = hanlp_key

    def run(self):
        try:
            data = None
            # --- HanLP 分支 ---
            if "HanLP" in self.model_name:
                from hanlp_restful import HanLPClient
                auth = self.hanlp_key if self.hanlp_key and "粘贴" not in self.hanlp_key else None
                client = HanLPClient('https://www.hanlp.com/api', auth=auth, language='zh')
                doc = client(self.text, tasks='dep')

                tokens = doc.get('tok/fine', doc.get('tok', []))
                pos = doc.get('pos/ctb', doc.get('pos/pku', doc.get('pos', [])))
                dep = doc.get('dep', [])

                if tokens and isinstance(tokens[0], list): tokens = tokens[0]
                if pos and isinstance(pos[0], list): pos = pos[0]
                if dep and isinstance(dep[0], list): dep = dep[0]

                data = []
                for i in range(len(tokens)):
                    data.append(
                        {"id": i + 1, "text": tokens[i], "pos": pos[i] if i < len(pos) else "X", "head": dep[i][0],
                         "rel": dep[i][1], "out_degree": 0})

            # --- Stanza 分支 ---
            elif "Stanza" in self.model_name:
                import stanza
                # 自动下载模型
                stanza.download('zh', verbose=False)
                nlp = stanza.Pipeline('zh', processors='tokenize,pos,lemma,depparse', verbose=False)
                doc = nlp(self.text)
                sent = doc.sentences[0]
                data = []
                for word in sent.words:
                    data.append(
                        {"id": word.id, "text": word.text, "pos": word.upos, "head": word.head, "rel": word.deprel,
                         "out_degree": 0})

            # 计算出度
            if data:
                for word in data:
                    if word['head'] > 0 and word['head'] <= len(data):
                        data[word['head'] - 1]['out_degree'] += 1

            self.finished.emit(data)

        except Exception as e:
            self.error.emit(str(e))


# ================= 2. 主窗口界面 (全面美化) =================
# ================= 2. 主窗口界面 (优化输入框尺寸) =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Syntax Analysis")
        self.resize(1280, 850)

        # 设置全局字体
        font = QFont("Segoe UI", 10)
        font.setStyleHint(QFont.SansSerif)
        QApplication.setFont(font)

        # 主界面容器
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #f3f4f6;")
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)  # 增加整体边距
        main_layout.setSpacing(25)

        # --- 顶部控制卡片 ---
        control_card = QFrame()
        control_card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 16px; /* 更圆润 */
                border: 1px solid #e5e7eb;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
        """)
        # 使用水平布局
        control_layout = QHBoxLayout(control_card)
        control_layout.setContentsMargins(25, 20, 25, 20)  # 卡片内部更宽敞
        control_layout.setSpacing(20)  # 控件之间的间距加大

        # 1. 模型选择 (稍微加大)
        # 1. 模型选择
        label_model = QLabel("分析内核:")
        label_model.setStyleSheet("font-size: 16px; font-weight: bold; color: #374151; border: none;")

        self.combo_model = QComboBox()
        self.combo_model.addItems(["HanLP (云端API)", "Stanza (学术标准)"])
        self.combo_model.setFixedWidth(220)  # 稍微加宽一点
        self.combo_model.setFixedHeight(45)  # 主按钮高度

        # 【关键步骤】设置视图为 QListView，这样才能用 CSS 撑开下拉项的高度
        self.combo_model.setView(QListView())

        self.combo_model.setStyleSheet("""
                    /* 1. 主按钮样式 */
                    QComboBox {
                        border: 2px solid #e5e7eb;
                        border-radius: 10px;
                        padding-left: 15px;
                        background-color: #ffffff;
                        color: #111827;
                        font-size: 16px; /* 字号加大 */
                        font-weight: 500;
                    }
                    QComboBox:hover {
                        border: 2px solid #d1d5db;
                    }
                    QComboBox::drop-down {
                        border: none;
                        width: 30px;
                    }
                    /* 下拉箭头 */
                    QComboBox::down-arrow {
                        image: url(none);
                        border-left: 6px solid transparent;
                        border-right: 6px solid transparent;
                        border-top: 7px solid #6b7280;
                        margin-right: 15px;
                    }

                    /* 2. 下拉弹窗样式 (QAbstractItemView) */
                    QComboBox QAbstractItemView {
                        border: 1px solid #e5e7eb;
                        background-color: #ffffff;
                        selection-background-color: #eff6ff; /* 鼠标悬停时的背景色(浅蓝) */
                        selection-color: #2563eb; /* 鼠标悬停时的文字色 */
                        outline: none;
                        border-radius: 10px;
                        padding: 5px; /* 弹窗整体内边距 */
                        margin-top: 5px;
                    }

                    /* 3. 每一个选项的样式 */
                    QComboBox QAbstractItemView::item {
                        min-height: 40px; /* 【重点】强制每一项的高度为 40px */
                        padding-left: 10px; /* 文字左边距 */
                        border-radius: 5px; /* 选项圆角 */
                    }
                """)

        # 2. 输入框 (重点修改对象！！！)
        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("")
        self.input_text.setText("我想吃拉面")
        self.input_text.setMinimumHeight(45)  # 强制最小高度，让它变“胖”
        self.input_text.setStyleSheet("""
            QLineEdit {
                border: 2px solid #e5e7eb; /* 边框加粗一点点 */
                border-radius: 10px;
                padding: 0 15px; /* 左右留白 */
                background-color: #ffffff;
                color: #111827;
                font-size: 18px; /* 字号加大！看起来更清楚 */
                font-family: 'Microsoft YaHei', sans-serif;
            }
            QLineEdit:focus { 
                border: 2px solid #2563eb; 
                background-color: #feffff;
            }
        """)

        # 3. 按钮 (加大)
        self.btn_run = QPushButton("START")
        self.btn_run.setFixedWidth(140)
        self.btn_run.setFixedHeight(45)  # 和输入框一样高
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 10px;
                border: none;
            }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:pressed { background-color: #1e40af; }
            QPushButton:disabled { background-color: #9ca3af; }
        """)
        self.btn_run.clicked.connect(self.start_analysis)

        # 添加到布局
        control_layout.addWidget(label_model)
        control_layout.addWidget(self.combo_model)
        control_layout.addWidget(self.input_text, 1)  # 参数 1 保证它占满剩余空间
        control_layout.addWidget(self.btn_run)

        main_layout.addWidget(control_card)

        # --- 进度条 ---
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(4)
        self.progress.setStyleSheet(
            "QProgressBar {border: none; background: #e5e7eb; border-radius: 2px;} QProgressBar::chunk {background-color: #2563eb; border-radius: 2px;}")
        self.progress.hide()
        main_layout.addWidget(self.progress)

        # --- 浏览器视图 ---
        self.webview = QWebEngineView()
        self.webview.page().setBackgroundColor(Qt.transparent)
        self.webview.setStyleSheet("background: transparent; border: none;")

        # 初始页 HTML (略...)
        # 这里你可以保留之前的 welcome_html 代码，不用变

        # 初始欢迎页
        welcome_html = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body { 
                font-family: 'Segoe UI', Roboto, sans-serif; 
                display: flex; justify-content: center; align-items: center; 
                height: 100vh; margin: 0; color: #4b5563; 
            }
            .welcome-container {
                text-align: center; padding: 40px;
                background: #ffffff; border-radius: 16px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
                border: 1px solid #e5e7eb;
            }
            h1 { color: #111827; margin-bottom: 10px; }
            p { font-size: 1.1rem; color: #6b7280; }
        </style>
        </head>
        <body>
            <div class="welcome-container">
                <h1>欢迎使用句法分析实验室</h1>
                <p>请在上方输入句子，选择模型后点击“开始分析”。</p>
            </div>
        </body>
        </html>
        """
        self.webview.setHtml(welcome_html)
        main_layout.addWidget(self.webview, 1)

    def start_analysis(self):
        text = self.input_text.text()
        if not text.strip():
            QMessageBox.warning(self, "提示", "请输入句子")
            return

        model = self.combo_model.currentText()

        # 👇👇👇 你的 HanLP Key 填在这里 👇👇👇
        hanlp_key = "OTUxOUBiYnMuaGFubHAuY29tOk9OTFE1N0V6SlJUT3dwVXE="

        self.btn_run.setEnabled(False)
        self.btn_run.setText("analyzing...")
        self.progress.show()

        self.thread = AnalysisThread(text, model, hanlp_key)
        self.thread.finished.connect(self.on_success)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_success(self, data):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("开始")
        self.progress.hide()

        if not data:
            QMessageBox.warning(self, "提示", "分析未返回数据，请检查输入。")
            return

        json_str = json.dumps(data)
        html_content = self.get_html_template(json_str)
        self.webview.setHtml(html_content)

    def on_error(self, err_msg):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("开始")
        self.progress.hide()
        QMessageBox.critical(self, "错误", f"分析过程中发生错误:\n{err_msg}")

    def get_html_template(self, json_data):
        # 使用了 v3.0 风格的现代化 CSS
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Serif+SC:wght@700&display=swap');

            body {{ 
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
                margin: 0; padding: 0; 
                background-color: transparent; /* 让 Qt 窗口背景透过来 */
            }}

            .container {{
                display: flex; flex-direction: column; gap: 24px;
            }}

            /* 通用卡片样式 */
            .card {{
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
                border: 1px solid #e5e7eb;
                padding: 24px;
            }}

            /* 可视化区域 */
            .viz-wrapper {{ 
                width: 100%; height: 360px; position: relative; 
                margin: 0 auto; user-select: none; overflow: visible;
            }}
            .words-row {{ 
                display: flex; justify-content: space-between; align-items: flex-end; 
                padding: 0 60px; position: absolute; bottom: 0; 
                width: 100%; box-sizing: border-box; height: 70px; 
            }}
            .word-block {{ 
                display: flex; flex-direction: column; align-items: center; 
                min-width: 50px; cursor: pointer; z-index: 10; transition: transform 0.2s;
            }}
            .word-block:hover {{ transform: translateY(-3px); }}
            .word-text {{ 
                font-size: 24px; color: #111827; margin-bottom: 8px; 
                font-family: 'Noto Serif SC', serif; font-weight: 700; 
            }}
            .word-pos {{ 
                font-size: 12px; color: #4b5563; background: #f3f4f6; 
                border: 1px solid #e5e7eb; padding: 3px 10px; border-radius: 14px; 
                font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
            }}

            /* KPI 仪表盘 */
            .dashboard-grid {{
                display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;
            }}
            .kpi-card {{
                background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px;
                padding: 20px; text-align: center; transition: all 0.2s;
            }}
            .kpi-card:hover {{ background: #fff; border-color: #c7d2fe; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1); }}
            .kpi-val {{ font-size: 32px; font-weight: 800; color: #111827; display: block; margin-bottom: 4px; line-height: 1; }}
            .kpi-label {{ font-size: 13px; color: #6b7280; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; }}

            /* 表格样式 */
            .table-wrapper {{ overflow-x: auto; }}
            table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 14px; color: #374151; }}
            th {{ 
                text-align: left; padding: 12px 16px; font-weight: 600; 
                color: #4b5563; text-transform: uppercase; font-size: 12px; 
                background: #f9fafb; border-bottom: 2px solid #e5e7eb; 
                border-top: 1px solid #f3f4f6;
            }}
            th:first-child {{ border-top-left-radius: 8px; border-left: 1px solid #f3f4f6; }}
            th:last-child {{ border-top-right-radius: 8px; border-right: 1px solid #f3f4f6; }}
            td {{ padding: 14px 16px; border-bottom: 1px solid #f3f4f6; background: #fff; transition: background 0.15s; }}
            tr:last-child td {{ border-bottom: none; }}
            tr:last-child td:first-child {{ border-bottom-left-radius: 8px; }}
            tr:last-child td:last-child {{ border-bottom-right-radius: 8px; }}
            tr:hover td {{ background: #f8fafc; }}
            .pos-tag {{
                background: #eff6ff; color: #1d4ed8; padding: 3px 8px; 
                border-radius: 6px; font-size: 12px; font-weight: 600; border: 1px solid #dbeafe;
            }}

            /* SVG 样式 */
            svg {{ width: 100%; height: 100%; position: absolute; top: 0; left: 0; pointer-events: none; }}
            path {{ 
                fill: none; stroke: #9ca3af; stroke-width: 1.5px; 
                pointer-events: stroke; transition: all 0.3s; cursor: pointer;
            }}
            .root-arc {{ stroke-dasharray: 4, 4; stroke: #d1d5db; }}
            text.dep-label {{ 
                font-size: 13px; fill: #4b5563; text-anchor: middle; font-family: 'Inter', sans-serif; 
                pointer-events: all; cursor: pointer; font-weight: 600;
                paint-order: stroke; stroke: white; stroke-width: 6px; stroke-linecap: round; stroke-linejoin: round;
            }}

            /* 高亮交互 */
            .hover-mode .word-block, .hover-mode path, .hover-mode text {{ opacity: 0.2; transition: opacity 0.2s; }}
            .hover-mode .highlighted {{ opacity: 1 !important; }}
            .hover-mode path.highlighted {{ stroke: #2563eb; stroke-width: 2.5px; }}
            .hover-mode text.highlighted {{ fill: #2563eb; font-weight: 700; }}
            .hover-mode .word-text.highlighted {{ color: #2563eb; }}
            .hover-mode .word-pos.highlighted {{ background: #dbeafe; color: #1e40af; border-color: #bfdbfe; }}
        </style>
        </head>
        <body>
            <div class="container">
                <div class="card" style="padding-bottom: 0;">
                    <div class="viz-wrapper" id="canvas-area">
                        <div class="words-row" id="words-row"></div>
                    </div>
                </div>

                <div class="card">
                    <div id="stats-section"></div>
                </div>
            </div>

            <script>
                const data = {json_data};
                const container = document.getElementById('canvas-area');
                const wordsRow = document.getElementById('words-row');

                function render() {{
                    const wordEls = [];
                    data.forEach(w => {{
                        const el = document.createElement('div'); el.className = 'word-block'; el.id = 'w-'+w.id;
                        el.innerHTML = `<div class="word-text">${{w.text}}</div><div class="word-pos" id="pos-${{w.id}}">${{w.pos}}</div>`;
                        el.onmouseenter = () => highlight(w.id); el.onmouseleave = clearH;
                        wordsRow.appendChild(el); wordEls.push(el);
                    }});

                    setTimeout(() => {{
                        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
                        const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
                        const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
                        marker.setAttribute("id", "arrow"); marker.setAttribute("markerWidth", "10"); marker.setAttribute("markerHeight", "10");
                        marker.setAttribute("refX", "8"); marker.setAttribute("refY", "3"); marker.setAttribute("orient", "auto");
                        const mPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
                        mPath.setAttribute("d", "M0,0 L0,6 L9,3 z"); mPath.setAttribute("fill", "#9ca3af");
                        marker.appendChild(mPath); defs.appendChild(marker); svg.appendChild(defs);
                        container.appendChild(svg);

                        const rect = container.getBoundingClientRect();
                        const centers = {{}};
                        wordEls.forEach((el, i) => {{
                            const r = el.getBoundingClientRect();
                            centers[data[i].id] = {{ x: r.left + r.width/2 - rect.left, y: 300 }};
                        }});

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
                            const h = 40 + (dist * 14); const cpY = 300 - h * 1.3;
                            drawPath(svg, `M${{s.x}},${{s.y}} C${{s.x}},${{cpY}} ${{e.x}},${{cpY}} ${{e.x}},${{e.y}}`, '', w.id, w.head);
                            drawLabel(svg, (s.x + e.x)/2, 300 - h * 1.1, w.rel, w.id, w.head);
                        }});

                        renderDashboard(data, tdd, n);
                    }}, 100);
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
                    <div class="table-wrapper" style="margin-top: 24px;">
                        <table>
                            <thead><tr><th>Word</th><th>Pos Tag</th><th>Relation</th><th>Head</th><th>Distance</th><th>Out-Degree</th></tr></thead>
                            <tbody>`;

                    data.forEach(w => {{
                        const hTxt = w.head===0 ? "ROOT" : data.find(x=>x.id===w.head).text;
                        const dist = w.head===0 ? "-" : Math.abs(w.head-w.id);
                        html += `<tr>
                            <td style="font-weight:600; color:#111827;">${{w.text}}</td>
                            <td><span class="pos-tag">${{w.pos}}</span></td>
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
                    document.getElementById('pos-'+id).classList.add('highlighted');
                    document.querySelectorAll('path').forEach(p => {{
                        if(p.dataset.h==id || p.dataset.d==id) {{
                            p.classList.add('highlighted');
                            if(p.dataset.h!=0) {{
                                document.getElementById('w-'+p.dataset.h).classList.add('highlighted');
                                document.getElementById('pos-'+p.dataset.h).classList.add('highlighted');
                            }}
                            document.getElementById('w-'+p.dataset.d).classList.add('highlighted');
                            document.getElementById('pos-'+p.dataset.d).classList.add('highlighted');
                        }}
                    }});
                }}
                function highlightArc(h, d) {{
                    document.body.classList.add('hover-mode');
                    const p = document.querySelector(`path[data-h="${{h}}"][data-d="${{d}}"]`); if(p) p.classList.add('highlighted');
                    if(h!=0) {{
                        document.getElementById('w-'+h).classList.add('highlighted');
                        document.getElementById('pos-'+h).classList.add('highlighted');
                    }}
                    document.getElementById('w-'+d).classList.add('highlighted');
                    document.getElementById('pos-'+d).classList.add('highlighted');
                }}
                function clearH() {{ document.body.classList.remove('hover-mode'); document.querySelectorAll('.highlighted').forEach(e => e.classList.remove('highlighted')); }}

                render();
            </script>
        </body>
        </html>
        """


if __name__ == "__main__":
    # 启用高分屏支持，让界面在 Mac/高分屏 Windows 上更清晰
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
