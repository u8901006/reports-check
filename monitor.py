import os
import re
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from html import escape
from pathlib import Path
from urllib.parse import urljoin

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
YESTERDAY = (datetime.now(TZ) - timedelta(days=1)).strftime("%Y-%m-%d")
YESTERDAY_SLUG = (datetime.now(TZ) - timedelta(days=1)).strftime("%Y-%m-%d")


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


def find_yesterday_report_url(index_url: str, index_html: str) -> str | None:
    soup = BeautifulSoup(index_html, "html.parser")
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        text = a_tag.get_text(strip=True)
        if YESTERDAY_SLUG in href or YESTERDAY_SLUG.replace("-", "/") in text:
            return urljoin(index_url + "/", href)
    return None


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
            resp_text = resp.text[:200] if "resp" in dir() else "no response"
            log.error("Parse error on attempt %d: %s — raw: %s", attempt + 1, e, resp_text)
            if attempt < MAX_RETRIES - 1:
                time.sleep(API_DELAY * (attempt + 1))
            continue
        except requests.exceptions.RequestException as e:
            log.error("Request error on attempt %d: %s", attempt + 1, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(API_DELAY * (attempt + 1))
            continue
    raise RuntimeError(f"All {MAX_RETRIES} API attempts failed")


def analyze_report_with_ai(report_url: str, content: str, target_date: str) -> dict:
    if len(content) > 12000:
        content = content[:12000]

    safe_url = report_url[:200]

    payload = {
        "model": GLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一個精神醫學文獻日報品質檢查員。請分析以下日報網頁內容，判斷日報是否正常更新。\n"
                    f"今天檢查的目標日期是：{target_date}（前一天）\n\n"
                    "請以 JSON 回答：\n"
                    "1. \"has_research_data\": boolean - 是否包含研究文獻資料（論文標題、摘要、DOI 等）\n"
                    f"2. \"is_updated_on_target_date\": boolean - 此日報是否為 {target_date} 的日報（日期相符）\n"
                    "3. \"update_date\": string - 內容中顯示的日報日期（格式 YYYY-MM-DD），若無則 null\n"
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
            "is_updated_on_target_date": False,
            "update_date": None,
            "research_count": 0,
            "summary": f"AI 分析失敗: {e}",
            "issues": [str(e)],
        }


def check_single_report(report: dict) -> dict:
    index_url = report["url"].rstrip("/")
    yesterday_url = None
    ai_result = {}

    try:
        index_html = fetch_page(index_url)
        yesterday_url = find_yesterday_report_url(index_url, index_html)
    except Exception as e:
        log.error("Failed to fetch index %s: %s", index_url, e)
        ai_result = {
            "has_research_data": False,
            "is_updated_on_target_date": False,
            "update_date": None,
            "research_count": 0,
            "summary": f"首頁抓取失敗: {e}",
            "issues": [str(e)],
        }
        return {
            "name": report["name"],
            "theme": report["theme"],
            "url": index_url,
            "yesterday_url": None,
            "status": "fail",
            "checked_at": datetime.now(TZ).isoformat(),
            **ai_result,
        }

    if not yesterday_url:
        log.warning("No yesterday (%s) report found on %s", YESTERDAY, index_url)
        ai_result = {
            "has_research_data": False,
            "is_updated_on_target_date": False,
            "update_date": None,
            "research_count": 0,
            "summary": f"找不到 {YESTERDAY} 的日報",
            "issues": [f"首頁未列出 {YESTERDAY} 的日報連結"],
        }
        return {
            "name": report["name"],
            "theme": report["theme"],
            "url": index_url,
            "yesterday_url": None,
            "status": "missing",
            "checked_at": datetime.now(TZ).isoformat(),
            **ai_result,
        }

    log.info("  Found yesterday report: %s", yesterday_url)
    try:
        report_html = fetch_page(yesterday_url)
        soup = BeautifulSoup(report_html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        ai_result = analyze_report_with_ai(yesterday_url, text, YESTERDAY)
    except Exception as e:
        log.error("Failed to fetch yesterday report %s: %s", yesterday_url, e)
        ai_result = {
            "has_research_data": False,
            "is_updated_on_target_date": False,
            "update_date": None,
            "research_count": 0,
            "summary": f"昨日日報抓取失敗: {e}",
            "issues": [str(e)],
        }

    has_data = ai_result.get("has_research_data", False)
    is_on_date = ai_result.get("is_updated_on_target_date", False)

    if has_data and is_on_date:
        status = "pass"
    elif has_data and not is_on_date:
        status = "wrong_date"
    elif not has_data and is_on_date:
        status = "no_data"
    else:
        status = "fail"

    return {
        "name": report["name"],
        "theme": report["theme"],
        "url": index_url,
        "yesterday_url": yesterday_url,
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


STATUS_ICONS = {
    "pass": "✅",
    "wrong_date": "⚠️",
    "no_data": "📭",
    "missing": "🚫",
    "fail": "❌",
}

STATUS_LABELS = {
    "pass": "正常",
    "wrong_date": "日期不符",
    "no_data": "無研究資料",
    "missing": "缺少日報",
    "fail": "異常",
}


def generate_html(results: list[dict], history: list[list[dict]]) -> str:
    now = datetime.now(TZ)
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "pass")
    wrong_date = sum(1 for r in results if r["status"] == "wrong_date")
    no_data = sum(1 for r in results if r["status"] == "no_data")
    missing = sum(1 for r in results if r["status"] == "missing")
    failed = sum(1 for r in results if r["status"] == "fail")
    unhealthy = wrong_date + no_data + missing + failed
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
        icon = STATUS_ICONS.get(r["status"], "❓")
        label = STATUS_LABELS.get(r["status"], "未知")
        name = escape(r.get("name", ""))
        summary = escape(r.get("summary", ""))
        update_date = escape(str(r.get("update_date", "N/A")))
        checked_at = escape(r.get("checked_at", ""))
        y_url = escape(r.get("yesterday_url", "")) if r.get("yesterday_url") else ""
        index_url = escape(r.get("url", "#"))
        theme_badge = escape(r.get("theme", ""))
        report_link = f'<a href="{y_url}" target="_blank">{name}</a>' if y_url else f'<a href="{index_url}" target="_blank">{name}</a>'
        rows_html += f"""
        <tr class="status-{r['status']}">
          <td>{icon} {label}</td>
          <td>{report_link}</td>
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
:root {{ --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --green: #22c55e; --yellow: #eab308; --red: #ef4444; --blue: #3b82f6; --orange: #f97316; }}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'Noto Sans TC', sans-serif; background: var(--bg); color: var(--text); padding: 1rem; }}
h1 {{ text-align: center; margin: 1rem 0; font-size: 1.5rem; }}
.meta {{ text-align: center; color: #94a3b8; margin-bottom: 1.5rem; font-size: 0.85rem; }}
.target-date {{ color: var(--orange); font-weight: 600; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
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
.status-wrong_date td:nth-child(1) {{ color: var(--orange); }}
.status-no_data td:nth-child(1) {{ color: var(--red); }}
.status-missing td:nth-child(1) {{ color: var(--red); }}
.status-fail td:nth-child(1) {{ color: var(--red); }}
a {{ color: var(--blue); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.history {{ margin-top: 2rem; }}
.history table {{ max-width: 400px; }}
.legend {{ margin-bottom: 1rem; padding: 0.8rem 1rem; background: var(--card); border-radius: 8px; font-size: 0.8rem; color: #94a3b8; }}
.legend span {{ margin-right: 1rem; }}
footer {{ text-align: center; margin-top: 2rem; color: #64748b; font-size: 0.75rem; }}
@media (max-width: 768px) {{ .grid {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
</head>
<body>
<h1>精神醫學文獻日報監控儀表板</h1>
<p class="meta">檢查目標：<span class="target-date">{YESTERDAY}</span>（前一天日報）｜ 檢查時間：{now.strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)｜ 來源：<a href="{escape(HUB_URL)}" target="_blank">李政洋身心診所</a></p>

<div class="legend">
  <span>✅ 正常 — 日報存在、日期正確、有研究資料</span>
  <span>⚠️ 日期不符 — 有內容但日期非前一天</span>
  <span>📭 無研究資料 — 日報存在但缺少文獻</span>
  <span>🚫 缺少日報 — 前一天日報連結不存在</span>
  <span>❌ 異常 — 抓取或分析失敗</span>
</div>

<div class="grid">
  <div class="card"><div class="num">{total}</div><div class="label">總報表數</div></div>
  <div class="card"><div class="num health">{health_pct}%</div><div class="label">健康度</div></div>
  <div class="card"><div class="num" style="color:var(--green)">{passed}</div><div class="label">✅ 正常</div></div>
  <div class="card"><div class="num" style="color:var(--orange)">{wrong_date}</div><div class="label">⚠️ 日期不符</div></div>
  <div class="card"><div class="num" style="color:var(--red)">{no_data}</div><div class="label">📭 無研究資料</div></div>
  <div class="card"><div class="num" style="color:var(--red)">{missing}</div><div class="label">🚫 缺少日報</div></div>
  <div class="card"><div class="num" style="color:var(--red)">{failed}</div><div class="label">❌ 異常</div></div>
</div>

<div class="themes">{theme_summary}</div>

<table>
  <thead>
    <tr><th>狀態</th><th>報表名稱</th><th>主題</th><th>日報日期</th><th>文獻數</th><th>AI 分析摘要</th><th>檢查時間</th></tr>
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

<footer>Reports Check Monitor · GitHub Actions · GLM-4.7-Flash · 檢查前一天日報狀態</footer>
</body>
</html>"""


def main():
    log.info("=== Starting daily report check ===")
    log.info("Target date (yesterday): %s", YESTERDAY)

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
        log.info("  -> status=%s research_count=%s update=%s yesterday_url=%s",
                 result["status"], result.get("research_count"),
                 result.get("update_date"), result.get("yesterday_url"))
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
