import os
import re
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from html import escape
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HUB_URL = os.environ.get(
    "HUB_URL",
    "https://www.leepsyclinic.com/2026/04/psychiatric-researchs-AI-report.html",
)
GLM_API_KEY = os.environ["GLM_API_KEY"]
GLM_API_BASE = os.environ.get(
    "GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4"
)
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4.7-flash")
TZ = timezone(timedelta(hours=8))
HISTORY_DIR = Path("history")
REPORTS_JSON = Path("reports_status.json")
API_DELAY = float(os.environ.get("API_DELAY", "5"))
MAX_RETRIES = 3


def fetch_page(url: str, timeout: int = 30) -> str:
    resp = requests.get(url, timeout=timeout, headers={
        "User-Agent": "Mozilla/5.0 (compatible; ReportsCheckBot/1.0)"
    })
    resp.raise_for_status()
    return resp.text


def extract_report_links(hub_html: str) -> list[dict]:
    soup = BeautifulSoup(hub_html, "html.parser")
    reports = []
    seen = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if "github.io" not in href:
            continue
        name = a_tag.get_text(strip=True)
        if not name or href in seen:
            continue
        seen.add(href)
        parent_div = a_tag.find_parent("div")
        theme = ""
        if parent_div:
            for sibling in parent_div.find_previous_siblings():
                text = sibling.get_text(strip=True)
                if any(kw in text for kw in ["Mood", "Trauma", "Neuro", "Psychiatry", "Child", "Relationship", "Body"]):
                    theme = text.split("\n")[0] if "\n" in text else text
                    break
        reports.append({"name": name, "url": href, "theme": theme})

    return reports


def call_glm_api(payload: dict) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{GLM_API_BASE}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {GLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=90,
            )
            if resp.status_code == 429:
                wait = API_DELAY * (attempt + 2)
                log.warning("Rate limited (429), waiting %.0fs before retry %d/%d", wait, attempt + 1, MAX_RETRIES)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"^```json\s*|^```\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except (json.JSONDecodeError, KeyError) as e:
            log.error("Parse error on attempt %d: %s — raw: %s", attempt + 1, e, resp.text[:200] if 'resp' in dir() else "no response")
            if attempt < MAX_RETRIES - 1:
                time.sleep(API_DELAY * (attempt + 1))
            continue
        except requests.exceptions.RequestException as e:
            log.error("Request error on attempt %d: %s", attempt + 1, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(API_DELAY * (attempt + 1))
            continue
    raise RuntimeError(f"All {MAX_RETRIES} API attempts failed")


def analyze_report_with_ai(report_url: str, content: str) -> dict:
    if len(content) > 12000:
        content = content[:12000]

    safe_url = report_url[:200]

    payload = {
        "model": GLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一個精神醫學文獻日報品質檢查員。請分析以下日報網頁內容，並以 JSON 回答：\n"
                    "1. \"has_research_data\": boolean - 是否包含研究文獻資料（論文標題、摘要、DOI等）\n"
                    "2. \"is_updated_today\": boolean - 內容中的日期是否為今天或最近的日期\n"
                    "3. \"update_date\": string - 內容中顯示的更新日期（格式 YYYY-MM-DD），若無則 null\n"
                    "4. \"research_count\": integer - 識別出的研究文獻數量\n"
                    "5. \"summary\": string - 一句話描述此日報的狀態（繁體中文）\n"
                    "6. \"issues\": array of strings - 發現的問題列表（如無問題則空陣列）\n"
                    "只回傳 JSON，不要其他文字。"
                ),
            },
            {
                "role": "user",
                "content": f"URL: {safe_url}\n\n---\n\n{content}",
            },
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    try:
        return call_glm_api(payload)
    except Exception as e:
        log.error("AI analysis failed for %s: %s", report_url, e)
        return {
            "has_research_data": False,
            "is_updated_today": False,
            "update_date": None,
            "research_count": 0,
            "summary": f"AI 分析失敗: {e}",
            "issues": [str(e)],
        }


def check_single_report(report: dict) -> dict:
    url = report["url"]
    try:
        html = fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        ai_result = analyze_report_with_ai(url, text)
    except Exception as e:
        log.error("Failed to fetch %s: %s", url, e)
        ai_result = {
            "has_research_data": False,
            "is_updated_today": False,
            "update_date": None,
            "research_count": 0,
            "summary": f"抓取失敗: {e}",
            "issues": [str(e)],
        }

    status = "pass" if ai_result.get("has_research_data") and ai_result.get("is_updated_today") else "fail"
    if not ai_result.get("has_research_data"):
        status = "no_data"
    elif not ai_result.get("is_updated_today"):
        status = "stale"

    return {
        "name": report["name"],
        "theme": report["theme"],
        "url": url,
        "status": status,
        "checked_at": datetime.now(TZ).isoformat(),
        **ai_result,
    }


def save_history(results: list[dict]):
    HISTORY_DIR.mkdir(exist_ok=True)
    date_str = datetime.now(TZ).strftime("%Y-%m-%d")
    path = HISTORY_DIR / f"{date_str}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("History saved to %s", path)


def load_history(days: int = 7) -> list[list[dict]]:
    history = []
    if not HISTORY_DIR.exists():
        return history
    files = sorted(HISTORY_DIR.glob("*.json"))[-days:]
    for f in files:
        with open(f, encoding="utf-8") as fh:
            history.append(json.load(fh))
    return history


def generate_html(results: list[dict], history: list[list[dict]]) -> str:
    now = datetime.now(TZ)
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "pass")
    stale = sum(1 for r in results if r["status"] == "stale")
    no_data = sum(1 for r in results if r["status"] == "no_data")
    failed = sum(1 for r in results if r["status"] == "fail")
    health_pct = round((passed / total * 100) if total else 0, 1)

    history_dates = []
    history_scores = []
    for h in history:
        if not h:
            continue
        h_date = h[0].get("checked_at", "")[:10]
        h_total = len(h)
        h_pass = sum(1 for r in h if r["status"] == "pass")
        history_dates.append(h_date)
        history_scores.append(round(h_pass / h_total * 100, 1) if h_total else 0)

    themes = {}
    for r in results:
        t = r.get("theme", "其他")
        themes.setdefault(t, []).append(r)

    rows_html = ""
    for r in results:
        icon = {"pass": "✅", "stale": "⚠️", "no_data": "📭", "fail": "❌"}.get(r["status"], "❓")
        theme_badge = escape(r.get("theme", ""))
        name = escape(r.get("name", ""))
        summary = escape(r.get("summary", ""))
        update_date = escape(str(r.get("update_date", "N/A")))
        checked_at = escape(r.get("checked_at", ""))
        url = escape(r.get("url", "#"))
        rows_html += f"""
        <tr class="status-{r['status']}">
          <td>{icon}</td>
          <td><a href="{url}" target="_blank">{name}</a></td>
          <td>{theme_badge}</td>
          <td>{update_date}</td>
          <td>{r.get('research_count', 0)}</td>
          <td>{summary}</td>
          <td>{checked_at}</td>
        </tr>"""

    history_rows = ""
    for date_str, score in zip(history_dates, history_scores):
        color = "#22c55e" if score >= 80 else "#eab308" if score >= 50 else "#ef4444"
        history_rows += f'<tr><td>{escape(date_str)}</td><td style="color:{color};font-weight:bold">{score}%</td></tr>'

    theme_summary = ""
    for t, reps in themes.items():
        t_pass = sum(1 for r in reps if r["status"] == "pass")
        theme_summary += f'<div class="theme-card"><h3>{escape(t)}</h3><p>{t_pass}/{len(reps)} 通過</p></div>'

    health_color = "var(--green)" if health_pct >= 80 else "var(--yellow)" if health_pct >= 50 else "var(--red)"

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>精神醫學文獻日報監控儀表板</title>
<style>
:root {{ --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --green: #22c55e; --yellow: #eab308; --red: #ef4444; --blue: #3b82f6; }}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'Noto Sans TC', sans-serif; background: var(--bg); color: var(--text); padding: 1rem; }}
h1 {{ text-align: center; margin: 1rem 0; font-size: 1.5rem; }}
.meta {{ text-align: center; color: #94a3b8; margin-bottom: 1.5rem; font-size: 0.85rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
.card {{ background: var(--card); border-radius: 12px; padding: 1.2rem; text-align: center; }}
.card .num {{ font-size: 2.2rem; font-weight: 700; }}
.card .label {{ font-size: 0.85rem; color: #94a3b8; margin-top: 0.3rem; }}
.health {{ color: {health_color}; }}
.themes {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 0.8rem; margin-bottom: 2rem; }}
.theme-card {{ background: var(--card); border-radius: 8px; padding: 0.8rem 1rem; }}
.theme-card h3 {{ font-size: 0.95rem; color: var(--blue); }}
.theme-card p {{ font-size: 0.8rem; color: #94a3b8; }}
table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 12px; overflow: hidden; }}
th {{ background: #334155; padding: 0.7rem 0.5rem; text-align: left; font-size: 0.85rem; }}
td {{ padding: 0.6rem 0.5rem; border-top: 1px solid #334155; font-size: 0.82rem; }}
tr:hover {{ background: #2d3a4f; }}
.status-pass td:nth-child(1) {{ color: var(--green); }}
.status-stale td:nth-child(1) {{ color: var(--yellow); }}
.status-no_data td:nth-child(1) {{ color: var(--red); }}
.status-fail td:nth-child(1) {{ color: var(--red); }}
a {{ color: var(--blue); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.history {{ margin-top: 2rem; }}
.history table {{ max-width: 400px; }}
footer {{ text-align: center; margin-top: 2rem; color: #64748b; font-size: 0.75rem; }}
@media (max-width: 768px) {{ .grid {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
</head>
<body>
<h1>精神醫學文獻日報監控儀表板</h1>
<p class="meta">最後檢查時間：{now.strftime('%Y-%m-%d %H:%M:%S')} (UTC+8) ｜ 來源：<a href="{escape(HUB_URL)}" target="_blank">{escape(HUB_URL)}</a></p>

<div class="grid">
  <div class="card"><div class="num">{total}</div><div class="label">總報表數</div></div>
  <div class="card"><div class="num health">{health_pct}%</div><div class="label">健康度</div></div>
  <div class="card"><div class="num" style="color:var(--green)">{passed}</div><div class="label">正常更新</div></div>
  <div class="card"><div class="num" style="color:var(--yellow)">{stale}</div><div class="label">未更新</div></div>
  <div class="card"><div class="num" style="color:var(--red)">{no_data + failed}</div><div class="label">無資料/異常</div></div>
</div>

<div class="themes">{theme_summary}</div>

<table>
  <thead>
    <tr><th>狀態</th><th>報表名稱</th><th>主題</th><th>更新日期</th><th>文獻數</th><th>AI 分析摘要</th><th>檢查時間</th></tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>

<div class="history">
  <h2>近 7 日健康趨勢</h2>
  <table>
    <thead><tr><th>日期</th><th>健康度</th></tr></thead>
    <tbody>{history_rows}</tbody>
  </table>
</div>

<footer>Reports Check Monitor · GitHub Actions · GLM-4.7-Flash</footer>
</body>
</html>"""


def main():
    log.info("=== Starting daily report check ===")

    log.info("Fetching hub page: %s", HUB_URL)
    hub_html = fetch_page(HUB_URL)
    reports = extract_report_links(hub_html)
    log.info("Found %d report links", len(reports))

    if not reports:
        log.warning("No report links found on hub page")
        return

    results = []
    for i, report in enumerate(reports, 1):
        log.info("[%d/%d] Checking: %s", i, len(reports), report["name"])
        result = check_single_report(report)
        results.append(result)
        log.info("  -> status=%s research_count=%s update=%s",
                 result["status"], result.get("research_count"), result.get("update_date"))
        if i < len(reports):
            time.sleep(API_DELAY)

    with open(REPORTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    save_history(results)
    history = load_history(7)
    html = generate_html(results, history)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    log.info("=== Check complete. %d/%d passed ===",
             sum(1 for r in results if r["status"] == "pass"), len(results))


if __name__ == "__main__":
    main()
