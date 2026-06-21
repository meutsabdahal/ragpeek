from __future__ import annotations
import json
from html import escape as html_escape
from ragpeek.session import TraceSession
from ragpeek.serialization import trace_to_dict


def render_html(session: TraceSession) -> str:
    data = json.dumps(trace_to_dict(session), indent=2).replace("<", "\\u003c")
    title = html_escape(session.query[:60], quote=True)
    query = html_escape(session.query, quote=True)
    section_count = len(session.retrieval_spans) + len(session.generation_spans)
    collapse_sections = section_count > 4

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RAG Trace — {title}</title>
<style>
  body{{font-family:system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 20px;color:#1a1a1a;background:#fff}}
  h1{{font-size:18px;font-weight:600}}
  .query{{background:#f5f5f5;border-radius:8px;padding:10px 14px;font-size:14px;margin:12px 0 12px;white-space:pre-wrap;word-break:break-word}}
  .trace-mode{{display:inline-block;background:#eef2ff;color:#3730a3;border-radius:999px;padding:4px 10px;font-size:11px;font-weight:600;margin-bottom:16px}}
  details.section{{border:1px solid #e5e5e5;border-radius:8px;margin-bottom:16px;overflow:hidden;background:#fff}}
  details.section[open]{{box-shadow:0 1px 0 rgba(0,0,0,0.02)}}
  summary.section-header{{background:#fafafa;border-bottom:1px solid #e5e5e5;padding:10px 16px;font-size:13px;font-weight:600;display:flex;gap:12px;align-items:center;cursor:pointer;list-style:none}}
  summary.section-header::-webkit-details-marker{{display:none}}
  .latency{{font-size:11px;color:#999;font-weight:400}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{text-align:left;padding:8px 12px;border-bottom:1px solid #e5e5e5;font-weight:500;color:#666}}
  td{{padding:8px 12px;border-bottom:1px solid #f0f0f0;vertical-align:top;word-break:break-word;white-space:pre-wrap}}
  .score-high{{color:#166534;font-weight:500}}
  .score-mid{{color:#92400e;font-weight:500}}
  .score-low{{color:#991b1b;font-weight:500}}
  .diagnosis{{padding:10px 16px;font-size:12px;color:#555;border-top:1px solid #f0f0f0}}
  .warn{{color:#92400e}}
  .ok{{color:#166534}}
  .response{{padding:12px 16px;font-size:13px;line-height:1.6;white-space:pre-wrap;word-break:break-word}}
  .meta{{padding:8px 16px;font-size:11px;color:#999;border-bottom:1px solid #f0f0f0}}
</style>
</head>
<body>
<h1>RAG Trace</h1>
<div class="query"><strong>Query:</strong> {query}</div>
<div class="trace-mode">Trace mode: <span id="trace-mode"></span></div>
<div id="root"></div>
<script>
const data = {data};
document.getElementById('trace-mode').textContent = data.analysis_report?.trace_mode || 'semantic';

function score_class(s) {{
  if (s >= 0.7) return 'score-high';
  if (s >= 0.5) return 'score-mid';
  return 'score-low';
}}

function score_signal(s) {{
  if (s >= 0.7) return '✓';
  if (s >= 0.5) return '⚠';
  return '✗';
}}

function createElement(tag, className, text) {{
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}}

function appendText(element, text) {{
  element.appendChild(document.createTextNode(text));
}}

function appendDiagnosis(section, diagnosis) {{
  const diagnosisContainer = createElement('div', 'diagnosis');
  diagnosis.forEach(d => {{
    diagnosisContainer.appendChild(
      createElement('div', d.includes('healthy') ? 'ok' : 'warn', d)
    );
  }});
  section.appendChild(diagnosisContainer);
}}

const root = document.getElementById('root');

data.retrieval_spans.forEach((span, i) => {{
  const section = document.createElement('details');
  section.className = 'section';
  if (!{str(collapse_sections).lower()}) section.open = true;
  const header = document.createElement('summary');
  header.className = 'section-header';
  appendText(header, `Retrieval ${{i + 1}}`);
  header.appendChild(
    createElement(
      'span',
      'latency',
      `${{span.latency_ms}}ms · k=${{span.k_returned}}/${{span.k_requested}}`
    )
  );
  section.appendChild(header);

  const table = document.createElement('table');
  const tableHeader = document.createElement('tr');
  ['Chunk', 'Score', ''].forEach(label => {{
    const th = document.createElement('th');
    th.textContent = label;
    tableHeader.appendChild(th);
  }});
  table.appendChild(tableHeader);

  span.chunks.forEach((chunk, j) => {{
    const row = document.createElement('tr');
    const chunkCell = document.createElement('td');
    chunkCell.textContent = chunk;
    row.appendChild(chunkCell);

    const scoreCell = createElement('td', score_class(span.scores[j]), span.scores[j].toFixed(2));
    const signalCell = createElement('td', score_class(span.scores[j]), score_signal(span.scores[j]));
    row.appendChild(scoreCell);
    row.appendChild(signalCell);
    table.appendChild(row);
  }});

  section.appendChild(table);
  appendDiagnosis(section, span.diagnosis);
  root.appendChild(section);
}});

data.generation_spans.forEach((span, i) => {{
  const section = document.createElement('details');
  section.className = 'section';
  if (!{str(collapse_sections).lower()}) section.open = true;
  const header = document.createElement('summary');
  header.className = 'section-header';
  appendText(header, `Generation ${{i + 1}}`);
  const linkedRetrievalCount = span.linked_retrieval_indices.length;
  const linkedRetrievalLabel = linkedRetrievalCount > 1
    ? ` · multi-hop=${{linkedRetrievalCount}} retrievals`
    : linkedRetrievalCount === 1
      ? ' · retrieval-linked'
      : '';
  header.appendChild(
    createElement(
      'span',
      'latency',
      `${{span.latency_ms}}ms · model=${{span.model}}${{linkedRetrievalLabel}}`
    )
  );
  section.appendChild(header);
  section.appendChild(
    createElement(
      'div',
      'meta',
      `Prompt tokens: ${{span.prompt_tokens}} · Response tokens: ${{span.response_tokens}}`
    )
  );
  if (span.linked_retrieval_indices.length) {{
    section.appendChild(
      createElement(
        'div',
        'meta',
        `Linked retrievals: ${{linkedRetrievalCount}}${{linkedRetrievalCount > 1 ? ' (multi-hop)' : ''}}`
      )
    );
  }}
  section.appendChild(createElement('div', 'response', span.response));
  appendDiagnosis(section, span.diagnosis);
  root.appendChild(section);
}});

const footer = document.createElement('div');
footer.style = 'font-size:11px;color:#aaa;margin-top:16px';
footer.textContent = `Total latency: ${{data.total_latency_ms}}ms · Session: ${{data.session_id}} · Mode: ${{data.analysis_report?.trace_mode || 'semantic'}}`;
root.appendChild(footer);
</script>
</body>
</html>"""
