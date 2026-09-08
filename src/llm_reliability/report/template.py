"""Self-contained interactive HTML, CSS, and JavaScript report template.

Zero external CDN or internet dependencies: 100% offline, self-contained single file.
"""

HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
:root {{
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-card: #1e293b;
  --bg-card-hover: #334155;
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --border: #334155;
  --border-light: #475569;
  --accent: #38bdf8;
  --accent-hover: #0284c7;
  --success: #10b981;
  --success-bg: rgba(16, 185, 129, 0.15);
  --warning: #f59e0b;
  --warning-bg: rgba(245, 158, 11, 0.15);
  --danger: #ef4444;
  --danger-bg: rgba(239, 68, 68, 0.15);
  --info: #3b82f6;
  --info-bg: rgba(59, 130, 246, 0.15);
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}

body.light {{
  --bg-primary: #f8fafc;
  --bg-secondary: #f1f5f9;
  --bg-card: #ffffff;
  --bg-card-hover: #f8fafc;
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --text-muted: #94a3b8;
  --border: #e2e8f0;
  --border-light: #cbd5e1;
  --accent: #0284c7;
  --accent-hover: #0369a1;
}}

* {{
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}}

body {{
  font-family: var(--font-sans);
  background-color: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.5;
  padding: 24px;
  transition: background-color 0.2s, color 0.2s;
}}

.container {{
  max-width: 1280px;
  margin: 0 auto;
}}

header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}}

h1 {{
  font-size: 24px;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 12px;
}}

.badge-logo {{
  background: linear-gradient(135deg, #0284c7, #38bdf8);
  color: white;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 700;
}}

.theme-toggle {{
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  color: var(--text-primary);
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
}}
.theme-toggle:hover {{
  background: var(--bg-card-hover);
}}

/* Executive Summary */
.summary-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}}

.summary-card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}}

.summary-card-title {{
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  font-weight: 600;
  margin-bottom: 6px;
}}

.summary-card-value {{
  font-size: 20px;
  font-weight: 700;
}}

.badge {{
  display: inline-block;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}}
.badge-critical, .badge-danger {{ background: var(--danger-bg); color: var(--danger); border: 1px solid var(--danger); }}
.badge-high, .badge-warning {{ background: var(--warning-bg); color: var(--warning); border: 1px solid var(--warning); }}
.badge-medium {{ background: rgba(234, 179, 8, 0.15); color: #eab308; border: 1px solid #eab308; }}
.badge-low, .badge-info {{ background: var(--info-bg); color: var(--info); border: 1px solid var(--info); }}
.badge-success {{ background: var(--success-bg); color: var(--success); border: 1px solid var(--success); }}

/* Card Sections */
.section-card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 24px;
}}

.section-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}}

.section-title {{
  font-size: 16px;
  font-weight: 700;
}}

/* Timeline Waterfall */
.timeline-container {{
  position: relative;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 16px;
  margin-top: 12px;
  overflow-x: auto;
}}

.timeline-row {{
  display: flex;
  align-items: center;
  margin-bottom: 10px;
  font-size: 12px;
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
}}
.timeline-row:hover {{
  background: var(--bg-card-hover);
}}

.timeline-name {{
  width: 220px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-family: var(--font-mono);
  margin-right: 12px;
}}

.timeline-bar-wrapper {{
  flex: 1;
  height: 22px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
  position: relative;
}}

.timeline-bar {{
  position: absolute;
  top: 0;
  height: 100%;
  border-radius: 4px;
  background: var(--accent);
  display: flex;
  align-items: center;
  padding: 0 6px;
  color: white;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
}}
.timeline-bar.kind-llm {{ background: #8b5cf6; }}
.timeline-bar.kind-retrieval {{ background: #3b82f6; }}
.timeline-bar.kind-tool {{ background: #10b981; }}
.timeline-bar.kind-chain {{ background: #f59e0b; }}
.timeline-bar.status-error {{ background: var(--danger); }}

/* Hypotheses & Evidence */
.hyp-card {{
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 14px;
  margin-bottom: 12px;
}}

.hyp-header {{
  display: flex;
  justify-content: space-between;
  font-weight: 600;
  margin-bottom: 6px;
}}

.progress-bar-container {{
  width: 120px;
  height: 10px;
  background: var(--border);
  border-radius: 5px;
  overflow: hidden;
  display: inline-block;
  vertical-align: middle;
  margin-left: 8px;
}}

.progress-bar-fill {{
  height: 100%;
  background: var(--accent);
}}

/* Tables */
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  margin-top: 10px;
}}

th, td {{
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}}

th {{
  background: var(--bg-secondary);
  color: var(--text-muted);
  font-weight: 600;
  text-transform: uppercase;
  font-size: 11px;
}}

tr:hover {{
  background: var(--bg-card-hover);
}}

/* Code / Snippet */
pre {{
  background: #020617;
  color: #e2e8f0;
  padding: 12px;
  border-radius: 6px;
  font-family: var(--font-mono);
  font-size: 12px;
  overflow-x: auto;
  margin-top: 6px;
  position: relative;
}}

.copy-btn {{
  position: absolute;
  top: 6px;
  right: 6px;
  background: #334155;
  color: #f8fafc;
  border: none;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 11px;
  cursor: pointer;
}}
.copy-btn:hover {{ background: #475569; }}

/* Modal / Drawer for Span Detail */
#span-modal {{
  display: none;
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.6);
  z-index: 999;
  justify-content: center;
  align-items: center;
}}
#span-modal.open {{ display: flex; }}

.modal-content {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  max-width: 800px;
  width: 90%;
  max-height: 85vh;
  overflow-y: auto;
  padding: 24px;
  position: relative;
}}

.modal-close {{
  position: absolute;
  top: 16px;
  right: 16px;
  background: transparent;
  border: none;
  color: var(--text-muted);
  font-size: 20px;
  cursor: pointer;
}}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1><span class="badge-logo">AGY</span> {title}</h1>
    <button class="theme-toggle" onclick="toggleTheme()">Toggle Theme</button>
  </header>

  <!-- Executive Summary -->
  <div class="summary-grid">
    <div class="summary-card">
      <div class="summary-card-title">Primary Diagnosis</div>
      <div class="summary-card-value">
        <span class="badge badge-{primary_severity_class}">{primary_category}</span>
      </div>
    </div>
    <div class="summary-card">
      <div class="summary-card-title">Confidence</div>
      <div class="summary-card-value">
        {confidence_pct}%
        <div class="progress-bar-container">
          <div class="progress-bar-fill" style="width: {confidence_pct}%"></div>
        </div>
      </div>
    </div>
    <div class="summary-card">
      <div class="summary-card-title">Total Latency</div>
      <div class="summary-card-value">{total_latency_ms} ms</div>
    </div>
    <div class="summary-card">
      <div class="summary-card-title">Total Spans / Failures</div>
      <div class="summary-card-value">{total_spans} / {failure_count}</div>
    </div>
  </div>

  <!-- Waterfall Timeline -->
  {waterfall_section}

  <!-- Root Cause & Hypotheses -->
  <div class="section-card">
    <div class="section-header">
      <div class="section-title">Root Cause Hypotheses & Evidence Chain</div>
    </div>
    {hypotheses_content}
  </div>

  <!-- Recommendations -->
  <div class="section-card">
    <div class="section-header">
      <div class="section-title">Actionable Developer Remediation</div>
    </div>
    {recommendations_content}
  </div>

  <!-- Detailed Metrics Breakdown -->
  <div class="section-card">
    <div class="section-header">
      <div class="section-title">Diagnostic Metrics Breakdown</div>
    </div>
    {metrics_table}
  </div>
</div>

<!-- Modal for Span Details -->
<div id="span-modal" onclick="closeModal(event)">
  <div class="modal-content" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeModal()">&times;</button>
    <h2 id="modal-span-name" style="margin-bottom: 12px; font-size: 18px;"></h2>
    <div id="modal-span-details"></div>
  </div>
</div>

<script>
function toggleTheme() {{
  document.body.classList.toggle('light');
}}

function showSpanModal(spanData) {{
  document.getElementById('modal-span-name').innerText = spanData.name;
  let html = '<p><strong>Kind:</strong> ' + spanData.kind + ' | <strong>Status:</strong> ' + spanData.status + ' | <strong>Duration:</strong> ' + spanData.duration_ms + 'ms</p>';
  if (spanData.error) {{
    html += '<p style="color: var(--danger); margin-top: 8px;"><strong>Error:</strong> ' + spanData.error + '</p>';
  }}
  if (spanData.prompt) {{
    html += '<div style="margin-top: 12px;"><strong>Prompt:</strong><pre>' + escapeHtml(spanData.prompt) + '</pre></div>';
  }}
  if (spanData.response) {{
    html += '<div style="margin-top: 12px;"><strong>Response:</strong><pre>' + escapeHtml(spanData.response) + '</pre></div>';
  }}
  if (spanData.tool_args) {{
    html += '<div style="margin-top: 12px;"><strong>Tool Arguments:</strong><pre>' + escapeHtml(JSON.stringify(spanData.tool_args, null, 2)) + '</pre></div>';
  }}
  if (spanData.tool_output) {{
    html += '<div style="margin-top: 12px;"><strong>Tool Output:</strong><pre>' + escapeHtml(JSON.stringify(spanData.tool_output, null, 2)) + '</pre></div>';
  }}
  if (spanData.query) {{
    html += '<div style="margin-top: 12px;"><strong>Retrieval Query:</strong> ' + escapeHtml(spanData.query) + '</div>';
  }}
  document.getElementById('modal-span-details').innerHTML = html;
  document.getElementById('span-modal').classList.add('open');
}}

function closeModal() {{
  document.getElementById('span-modal').classList.remove('open');
}}

function copyCode(btn) {{
  const pre = btn.parentElement;
  const text = pre.innerText.replace('Copy', '').trim();
  navigator.clipboard.writeText(text).then(() => {{
    btn.innerText = 'Copied!';
    setTimeout(() => {{ btn.innerText = 'Copy'; }}, 2000);
  }});
}}

function escapeHtml(str) {{
  if (!str) return '';
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}}
</script>
</body>
</html>
"""
