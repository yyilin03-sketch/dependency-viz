import sys
import json
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QComboBox, QMessageBox, QProgressBar)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import Qt, QThread, Signal

# ================= 1. 后端分析线程 =================
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
                auth = self.hanlp_key if self.hanlp_key else None
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
                    data.append({"id": i+1, "text": tokens[i], "pos": pos[i] if i<len(pos) else "X", "head": dep[i][0], "rel": dep[i][1], "out_degree": 0})

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
                    data.append({"id": word.id, "text": word.text, "pos": word.upos, "head": word.head, "rel": word.deprel, "out_degree": 0})

            # 计算出度
            if data:
                for word in data:
                    if word['head'] > 0 and word['head'] <= len(data):
                        data[word['head'] - 1]['out_degree'] += 1
            
            self.finished.emit(data)

        except Exception as e:
            self.error.emit(str(e))

# ================= 2. 主窗口界面 =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zoe的句法分析实验室 (精简版)")
        self.resize(1100, 750)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- 顶部控制栏 ---
        controls_layout = QHBoxLayout()
        
        self.combo_model = QComboBox()
        # 移除了 spaCy，只保留这两个
        self.combo_model.addItems(["HanLP (云端API)", "Stanza (学术标准)"])
        self.combo_model.setStyleSheet("font-size: 14px; padding: 5px;")
        
        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("请输入要分析的句子...")
        self.input_text.setText("他把那棵树种在了院子里")
        self.input_text.setStyleSheet("font-size: 14px; padding: 5px;")
        
        self.btn_run = QPushButton("开始分析")
        self.btn_run.setStyleSheet("background-color: #2c3e50; color: white; padding: 8px 20px; border-radius: 5px;")
        self.btn_run.clicked.connect(self.start_analysis)

        controls_layout.addWidget(QLabel("模型:"))
        controls_layout.addWidget(self.combo_model)
        controls_layout.addWidget(self.input_text, 1)
        controls_layout.addWidget(self.btn_run)
        
        layout.addLayout(controls_layout)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)

        self.webview = QWebEngineView()
        self.webview.setHtml("<div style='text-align:center;margin-top:50px;color:#888'><h1>👋 欢迎使用</h1><p>已移除 spaCy 依赖，轻量启动</p></div>")
        layout.addWidget(self.webview)

    def start_analysis(self):
        text = self.input_text.text()
        if not text.strip():
            QMessageBox.warning(self, "提示", "请输入句子")
            return

        model = self.combo_model.currentText()
        
        # 
        hanlp_key = "OTUxOUBiYnMuaGFubHAuY29tOk9OTFE1N0V6SlJUT3dwVXE=" 

        self.btn_run.setEnabled(False)
        self.progress.show()
        
        self.thread = AnalysisThread(text, model, hanlp_key)
        self.thread.finished.connect(self.on_success)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_success(self, data):
        self.btn_run.setEnabled(True)
        self.progress.hide()
        
        if not data:
            QMessageBox.warning(self, "失败", "分析未返回数据")
            return

        json_str = json.dumps(data)
        html_content = self.get_html_template(json_str)
        self.webview.setHtml(html_content)

    def on_error(self, err_msg):
        self.btn_run.setEnabled(True)
        self.progress.hide()
        QMessageBox.critical(self, "错误", f"分析出错:\n{err_msg}")

    def get_html_template(self, json_data):
        # 保持之前的 HTML 模板不变
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 0; padding: 20px; background: white; }}
            .viz-wrapper {{ width: 100%; height: 380px; position: relative; margin: 0 auto; user-select: none; }}
            .words-row {{ display: flex; justify-content: space-between; align-items: flex-end; padding: 0 50px; position: absolute; bottom: 0; width: 100%; box-sizing: border-box; height: 60px; }}
            .word-block {{ display: flex; flex-direction: column; align-items: center; min-width: 40px; cursor: pointer; z-index: 10; }}
            .word-text {{ font-size: 22px; color: #111827; margin-bottom: 6px; font-weight: bold; }}
            .word-pos {{ font-size: 12px; color: #6b7280; background: #f3f4f6; border: 1px solid #e5e7eb; padding: 2px 8px; border-radius: 12px; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 20px; }}
            th {{ text-align: left; background: #f9fafb; padding: 10px; border-bottom: 2px solid #ddd; }}
            td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
            svg {{ width: 100%; height: 100%; position: absolute; top: 0; left: 0; pointer-events: none; }}
            path {{ fill: none; stroke: #666; stroke-width: 1.5px; transition: all 0.3s; pointer-events: stroke;}}
            .root-arc {{ stroke-dasharray: 4, 4; stroke: #999; }}
            text {{ font-size: 12px; fill: #333; text-anchor: middle; pointer-events: all; cursor: pointer; paint-order: stroke; stroke: white; stroke-width: 4px; }}
            .hover-mode .word-block, .hover-mode path, .hover-mode text {{ opacity: 0.1; }}
            .hover-mode .highlighted {{ opacity: 1 !important; stroke: #2563eb; fill: #2563eb; color: #2563eb; }}
            path.highlighted {{ stroke: #2563eb; stroke-width: 2.5px; }}
        </style>
        </head>
        <body>
            <div class="viz-wrapper" id="canvas"><div class="words-row" id="words"></div></div>
            <div id="table-area"></div>
            <script>
                const data = {json_data};
                const container = document.getElementById('canvas');
                const wordsRow = document.getElementById('words');

                function render() {{
                    const wordEls = [];
                    data.forEach(w => {{
                        const el = document.createElement('div'); el.className = 'word-block'; el.id = 'w-'+w.id;
                        el.innerHTML = '<div class="word-text">'+w.text+'</div><div class="word-pos">'+w.pos+'</div>';
                        el.onmouseenter = function(){{highlight(w.id)}}; el.onmouseleave = clearH;
                        wordsRow.appendChild(el); wordEls.push(el);
                    }});

                    setTimeout(function() {{
                        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
                        const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
                        const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
                        marker.setAttribute("id", "arrow"); marker.setAttribute("markerWidth", "10"); marker.setAttribute("markerHeight", "10"); marker.setAttribute("refX", "8"); marker.setAttribute("refY", "3"); marker.setAttribute("orient", "auto");
                        const mPath = document.createElementNS("http://www.w3.org/2000/svg", "path"); mPath.setAttribute("d", "M0,0 L0,6 L9,3 z"); mPath.setAttribute("fill", "#666");
                        marker.appendChild(mPath); defs.appendChild(marker); svg.appendChild(defs);
                        container.appendChild(svg);

                        const rect = container.getBoundingClientRect();
                        const centers = {{}};
                        wordEls.forEach(function(el, i) {{ const r = el.getBoundingClientRect(); centers[data[i].id] = {{ x: r.left + r.width/2 - rect.left, y: 320 }}; }});

                        let tableHtml = '<table><thead><tr><th>Word</th><th>Pos</th><th>Rel</th><th>Head</th><th>Dist</th><th>Degree</th></tr></thead><tbody>';

                        data.forEach(function(w) {{
                            const hTxt = w.head===0 ? "ROOT" : data.find(function(x){{return x.id===w.head}}).text;
                            const dist = w.head===0 ? "-" : Math.abs(w.head-w.id);
                            tableHtml += '<tr><td><b>'+w.text+'</b></td><td>'+w.pos+'</td><td>'+w.rel+'</td><td>'+hTxt+'</td><td>'+dist+'</td><td>'+w.out_degree+'</td></tr>';

                            if (w.head === 0) {{
                                const pos = centers[w.id];
                                drawPath(svg, 'M'+pos.x+','+pos.y+' V'+(pos.y-70), 'root-arc', w.id, 0);
                                drawLabel(svg, pos.x, pos.y-80, 'ROOT', w.id, 0);
                                return;
                            }}
                            const s = centers[w.head]; const e = centers[w.id];
                            const h = 30 + (Math.abs(w.head - w.id) * 12); const cpY = 320 - h * 1.5;
                            drawPath(svg, 'M'+s.x+','+s.y+' C'+s.x+','+cpY+' '+e.x+','+cpY+' '+e.x+','+e.y, '', w.id, w.head);
                            drawLabel(svg, (s.x+e.x)/2, 320 - h * 1.15, w.rel, w.id, w.head);
                        }});
                        tableHtml += '</tbody></table>';
                        document.getElementById('table-area').innerHTML = tableHtml;

                    }}, 100);
                }}

                function drawPath(svg, d, cls, dep, head) {{ const p = document.createElementNS("http://www.w3.org/2000/svg", "path"); p.setAttribute("d", d); if(cls) p.setAttribute("class", cls); if(head!==0) p.setAttribute("marker-end", "url(#arrow)"); p.dataset.h = head; p.dataset.d = dep; p.onmouseenter = function(){{highlightArc(head, dep)}}; p.onmouseleave = clearH; svg.appendChild(p); }}
                function drawLabel(svg, x, y, txt, dep, head) {{ const t = document.createElementNS("http://www.w3.org/2000/svg", "text"); t.setAttribute("x", x); t.setAttribute("y", y); t.textContent = txt; if(head!==0) {{ t.onmouseenter = function(){{highlightArc(head, dep)}}; t.onmouseleave = clearH; }} svg.appendChild(t); }}
                function highlight(id) {{ document.body.classList.add('hover-mode'); document.getElementById('w-'+id).classList.add('highlighted'); document.querySelectorAll('path').forEach(function(p){{ if(p.dataset.h==id || p.dataset.d==id) {{ p.classList.add('highlighted'); if(p.dataset.h!=0) document.getElementById('w-'+p.dataset.h).classList.add('highlighted'); document.getElementById('w-'+p.dataset.d).classList.add('highlighted'); }} }}); }}
                function highlightArc(h, d) {{ document.body.classList.add('hover-mode'); const p = document.querySelector('path[data-h="'+h+'"][data-d="'+d+'"]'); if(p) p.classList.add('highlighted'); if(h!=0) document.getElementById('w-'+h).classList.add('highlighted'); document.getElementById('w-'+d).classList.add('highlighted'); }}
                function clearH() {{ document.body.classList.remove('hover-mode'); document.querySelectorAll('.highlighted').forEach(function(e){{e.classList.remove('highlighted')}}); }}
                render();
            </script>
        </body>
        </html>
        """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
