import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from groq import Groq
import requests
import xml.etree.ElementTree as ET
import re
from io import StringIO
import streamlit.components.v1 as components
import html
# ===========================
# Google Analytics
# ===========================
GA_MEASUREMENT_ID = st.secrets.get("GA_MEASUREMENT_ID", "")

def sanitize_html(text: str) -> str:
    return html.escape(text, quote=True)
def inject_ga():
    """Google Analyticsタグを注入"""
    if not GA_MEASUREMENT_ID or not GA_MEASUREMENT_ID.startswith("G-"):
        return

    components.html(
        f"""
        <script async src="https://www.googletagmanager.com/gtag/js?id={sanitize_html(GA_MEASUREMENT_ID)}"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){{dataLayer.push(arguments);}}
          gtag('js', new Date());
          gtag('config', '{sanitize_html(GA_MEASUREMENT_ID)}', {{
              'send_page_view': false
          }});
        </script>
        """,
        height=0,
        width=0,
    )
inject_ga()
def track_page_view():
    if not GA_MEASUREMENT_ID:
        return

    components.html(
        """
        <script>
        if (typeof gtag !== 'undefined') {
            gtag('event', 'page_view', {
                page_title: document.title,
                page_location: window.location.href
            });
        }
        </script>
        """,
        height=0,
        width=0,
    )

track_page_view()
# -----------------------------
# フォント設定（日本語対応）
# -----------------------------
import matplotlib.font_manager as fm
import os

_FONT_PATHS = [
    "font/NotoSansCJK-Regular.ttc",
    "font/NotoSansJP-ExtraBold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
]

def _set_japanese_font():
    for path in _FONT_PATHS:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            return prop.get_name()
    return None

_set_japanese_font()
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# 定数
# -----------------------------
GEMINI_MODEL = "gemini-2.5-pro"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# -----------------------------
# ページ設定
# -----------------------------
st.set_page_config(layout="wide", page_title="📈 日本株 分析ダッシュボード", page_icon="📈")
st.title("📈 日本株 シャープレシオ分析 + ニュース統合")

# -----------------------------
# AI設定（Gemini優先 / Groqフォールバック）
# -----------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GROQ_API_KEY   = st.secrets.get("GROQ_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)
groq_client  = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

def generate_ai_comment(prompt: str) -> tuple[str, str]:
    """Gemini -> Groq -> OpenRouter の順でフォールバック"""
    # 1) Gemini
    try:
        response = gemini_model.generate_content(prompt)
        return response.text, "Gemini"
    except Exception as e:
        gemini_err = str(e)

    # 2) Groq
    if groq_client:
        try:
            chat = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
            )
            return chat.choices[0].message.content, "Groq"
        except Exception as e:
            groq_err = str(e)
    else:
        groq_err = "GROQ_API_KEY 未設定"

    # 3) OpenRouter
    if OPENROUTER_API_KEY:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://jstock-dashboard.streamlit.app",
                    "X-Title": "JStock Dashboard",
                },
                json={
                    "model": "meta-llama/llama-3.1-8b-instruct:free",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 600,
                },
                timeout=30,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            return text, "OpenRouter"
        except Exception as e:
            or_err = str(e)
    else:
        or_err = "OPENROUTER_API_KEY 未設定"

    raise RuntimeError(
        f"全AIバックエンド失敗 / Gemini: {gemini_err} / Groq: {groq_err} / OpenRouter: {or_err}"
    )

# ================================================================
# 📰 ニュース取得モジュール
# ================================================================

_NEWS_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# ── ① Yahoo!ファイナンス Japan RSS ─────────────────────────────
@st.cache_data(ttl=600)
def fetch_yahoo_jp_news(ticker_code: str, max_items: int = 8) -> list[dict]:
    """
    Yahoo!ファイナンス Japan の銘柄別ニュースRSSを取得。
    ticker_code: '7203' など（.T なし）
    """
    code = ticker_code.replace(".T", "")
    url = f"https://finance.yahoo.co.jp/rss/stocks/{code}"
    try:
        r = requests.get(url, headers=_NEWS_HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        items = []
        for item in root.findall(".//item")[:max_items]:
            title = item.findtext("title", "").strip()
            link  = item.findtext("link", "").strip()
            pubdate = item.findtext("pubDate", "").strip()
            desc  = item.findtext("description", "").strip()
            # HTMLタグ除去
            desc = re.sub(r"<[^>]+>", "", desc)[:100]
            if title:
                items.append({"source": "Yahoo!Finance JP", "title": title,
                              "link": link, "date": pubdate, "summary": desc})
        return items
    except Exception:
        return []


# ── ② 株探（Kabutan）銘柄別ニュース ────────────────────────────
@st.cache_data(ttl=600)
def fetch_kabutan_news(ticker_code: str, max_items: int = 8) -> list[dict]:
    """
    株探の銘柄ニュースページから取得。
    URL: https://kabutan.jp/stock/news?code=XXXX

    HTMLファイル実測による確定構造:
    ─────────────────────────────────────────────────
    <table class="s_news_list mgbt0">
      <tbody>
        <tr>
          <td class="news_time">
            <time datetime="2026-02-19T17:00:03+09:00">26/02/19&nbsp;17:00</time>
          </td>
          <td>
            <div class="newslist_ctg newsctg5_b">特集</div>
          </td>
          <td>
            <a href="https://kabutan.jp/stock/news?code=5803&b=n202602191135">
              レーティング日報【最上位を継続＋目標株価を増額】(2月19日)
            </a>
          </td>
        </tr>
        ...
        <!-- 開示（PDF）の場合 -->
        <tr>
          <td class="news_time"><time ...>26/02/09&nbsp;14:00</time></td>
          <td><div class="newslist_ctg newsctg_kaiji_b">開示</div></td>
          <td class="td_kaiji">
            <a href="https://kabutan.jp/disclosures/pdf/20260209/140120260206550334/" target="pdf">
              2026年３月期通期連結業績予想...
            </a>
          </td>
        </tr>
      </tbody>
    </table>
    ─────────────────────────────────────────────────
    ※ このページ自体が code=XXXX の銘柄専用ページなので
      取得記事はすべて銘柄固有情報。
    ※ リンクURLに &b=n... (ニュース) または /disclosures/pdf/... (開示PDF) の2種類あり。
    ※ プレミアム記事は <img class="vat pdr4"> が挿入される。
    """
    code = ticker_code.replace(".T", "")
    url = f"https://kabutan.jp/stock/news?code={code}"
    headers = {
        **_NEWS_HEADERS,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer": "https://kabutan.jp/",
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return []
        html = r.text

        # ── s_news_list テーブルを抽出 ─────────────────────────
        # テーブル全体を取得
        table_match = re.search(
            r'class="s_news_list[^"]*"[^>]*>(.*?)</table>',
            html, re.DOTALL
        )
        if not table_match:
            return []
        table_html = table_match.group(1)

        # ── tr 行ごとにパース ──────────────────────────────────
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL)

        # カテゴリ判定マップ
        ctg_class_map = {
            "newsctg2_b":    "材料",
            "newsctg3_kk_b": "決算",
            "newsctg4_b":    "テク",
            "newsctg5_b":    "特集",
            "newsctg_kaiji_b": "開示",
        }
        badge_emoji = {
            "材料": "🟢", "決算": "🔵", "テク": "⚪",
            "特集": "🟠", "開示": "🔴",
        }

        items = []
        for row in rows:
            # ① 日時: <time datetime="2026-02-19T17:00:03+09:00">
            time_match = re.search(r'<time[^>]+datetime="([^"]+)"', row)
            if not time_match:
                continue
            # datetime属性から読みやすい形式に変換
            dt_raw = time_match.group(1)  # "2026-02-19T17:00:03+09:00"
            dt_disp = re.search(r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})', dt_raw)
            date_str = f"{dt_disp.group(1)} {dt_disp.group(2)}" if dt_disp else dt_raw[:16]

            # ② カテゴリバッジ: class="newslist_ctg newsctgX_b"
            badge = ""
            for cls, label in ctg_class_map.items():
                if cls in row:
                    badge = label
                    break

            # ③ リンクとタイトル: 2パターン
            #    a) ニュース: href="https://kabutan.jp/stock/news?code=XXXX&b=nXXX"
            #    b) 開示PDF:  href="https://kabutan.jp/disclosures/pdf/..."
            link_match = re.search(
                r'<a\s+href="(https://kabutan\.jp/(?:stock/news\?[^"]+|disclosures/pdf/[^"]+))"'
                r'[^>]*>\s*(.*?)\s*</a>',
                row, re.DOTALL
            )
            if not link_match:
                continue

            link = link_match.group(1).replace("&amp;", "&")
            # タイトルからHTMLタグ（imgなど）を除去
            title = re.sub(r'<[^>]+>', '', link_match.group(2)).strip()

            if len(title) < 3:
                continue

            # ④ プレミアム記事の検出（ロック画像が挿入される）
            is_premium = "🔒 " if "premium" in row.lower() or "pdr4" in row else ""

            emoji = badge_emoji.get(badge, "📰")

            items.append({
                "source": "株探(Kabutan)",
                "title": f"{is_premium}{title}",
                "badge": badge,
                "badge_emoji": emoji,
                "link": link,
                "date": date_str,
                "summary": "",
                "ticker_specific": True,
            })

            if len(items) >= max_items:
                break

        return items

    except Exception:
        return []


# ── 英語社名マッピング（主要銘柄） ───────────────────────────────
_JP_EN_NAME_MAP = {
    "トヨタ": "Toyota", "ホンダ": "Honda", "日産自": "Nissan", "ソニーＧ": "Sony",
    "三菱ＵＦＪ": "Mitsubishi UFJ", "三井住友ＦＧ": "Sumitomo Mitsui",
    "みずほＦＧ": "Mizuho", "ソフトバンク": "SoftBank", "ＳＢＧ": "SoftBank",
    "任天堂": "Nintendo", "パナＨＤ": "Panasonic", "日立": "Hitachi",
    "富士通": "Fujitsu", "ＮＥＣ": "NEC", "キヤノン": "Canon",
    "シャープ": "Sharp", "東エレク": "Tokyo Electron", "信越化": "Shin-Etsu",
    "村田製": "Murata", "京セラ": "Kyocera", "ダイキン": "Daikin",
    "コマツ": "Komatsu", "ファナック": "Fanuc", "キーエンス": "Keyence",
    "ルネサス": "Renesas", "アドテスト": "Advantest", "レーザーテク": "Lasertec",
    "ディスコ": "Disco", "ニデック": "Nidec", "三菱電": "Mitsubishi Electric",
    "伊藤忠": "Itochu", "三菱商": "Mitsubishi Corp", "三井物": "Mitsui",
    "住友商": "Sumitomo Corp", "丸紅": "Marubeni", "武田": "Takeda",
    "エーザイ": "Eisai", "第一三共": "Daiichi Sankyo", "中外薬": "Chugai",
    "アステラス": "Astellas", "リクルート": "Recruit", "メルカリ": "Mercari",
    "楽天グループ": "Rakuten", "ＮＴＴ": "NTT", "ＫＤＤＩ": "KDDI",
    "東京海上": "Tokio Marine", "ＪＴ": "Japan Tobacco",
    "日本製鉄": "Nippon Steel", "ブリヂストン": "Bridgestone",
    "ＪＡＬ": "Japan Airlines", "ＡＮＡＨＤ": "ANA",
}

def _get_en_name(company_name: str) -> str:
    """日本語社名から英語名を推定"""
    for jp, en in _JP_EN_NAME_MAP.items():
        if jp in company_name:
            return en
    # ローマ字っぽい文字列を含む場合はそのまま
    ascii_part = re.sub(r'[^\x20-\x7E]', '', company_name).strip()
    return ascii_part if len(ascii_part) >= 2 else ""


# ── Google News RSS（過去90日）銘柄検索 ──────────────────────────
@st.cache_data(ttl=600)
def _fetch_google_news_rss(query: str, source_filter: str, max_items: int, days: int = 90) -> list[dict]:
    """
    Google News RSS で query を検索し、指定ソースの記事のみ返す。
    days: 過去何日以内の記事のみ返すか
    """
    import datetime as dt
    import urllib.parse

    q_enc = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={q_enc}&hl=ja&gl=JP&ceid=JP:ja"
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)

    try:
        r = requests.get(url, headers=_NEWS_HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        # Google News RSSはUTF-8
        content = r.content
        # namespace宣言が壊れることがあるので前処理
        content = re.sub(rb'<\?xml[^?]*\?>', b'<?xml version="1.0" encoding="UTF-8"?>', content)
        root = ET.fromstring(content)
        items = []
        for item in root.findall(".//item"):
            title   = item.findtext("title", "").strip()
            link    = item.findtext("link", "").strip()
            pubdate = item.findtext("pubDate", "").strip()
            source_elem = item.find("source")
            source_name = source_elem.text.strip() if source_elem is not None else ""

            if not title:
                continue

            # ソースフィルタ（部分一致）
            if source_filter and source_filter.lower() not in source_name.lower():
                continue

            # 日付フィルタ（過去days日以内）
            if pubdate:
                try:
                    from email.utils import parsedate_to_datetime
                    pub_dt = parsedate_to_datetime(pubdate)
                    if pub_dt.tzinfo is None:
                        import datetime as dt2
                        pub_dt = pub_dt.replace(tzinfo=dt2.timezone.utc)
                    if pub_dt < cutoff:
                        continue
                    date_str = pub_dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    date_str = pubdate[:16]
            else:
                date_str = ""

            items.append({
                "title": title,
                "link": link,
                "date": date_str,
                "source_name": source_name,
            })
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


# ── ③ 日経新聞 銘柄別（Google News経由・過去90日） ────────────────
@st.cache_data(ttl=600)
def fetch_nikkei_stock_news(company_name: str, ticker_code: str, max_items: int = 10) -> list[dict]:
    """日経新聞の銘柄関連記事をGoogle News RSS経由で取得（過去90日）"""
    code = ticker_code.replace(".T", "")
    en_name = _get_en_name(company_name)
    company_short = re.sub(r'[　（）()ＨＤホールディングス\s]', '', company_name)

    # 複数クエリを試してマージ
    queries = [f"{company_short} site:nikkei.com", f"{code} 日経"]
    if en_name:
        queries.append(f"{en_name} nikkei")

    all_items = []
    seen = set()
    for q in queries:
        for it in _fetch_google_news_rss(q, "日経", max_items * 2, days=90):
            k = it["title"][:40]
            if k not in seen:
                seen.add(k)
                all_items.append({
                    "source": "日経新聞",
                    "title": it["title"],
                    "link": it["link"],
                    "date": it["date"],
                    "summary": "",
                    "ticker_specific": True,
                })
        if len(all_items) >= max_items:
            break
    return all_items[:max_items]


# ── ④ CNBC 銘柄別（Google News経由・過去90日） ───────────────────
@st.cache_data(ttl=600)
def fetch_cnbc_news(company_name: str, ticker_code: str, max_items: int = 10) -> list[dict]:
    """CNBCの銘柄関連記事をGoogle News RSS経由で取得（過去90日）"""
    code = ticker_code.replace(".T", "")
    en_name = _get_en_name(company_name)

    queries = []
    if en_name:
        queries.append(f"{en_name} site:cnbc.com")
        queries.append(f"{en_name} CNBC")
    queries.append(f"{code} CNBC")

    all_items = []
    seen = set()
    for q in queries:
        for it in _fetch_google_news_rss(q, "CNBC", max_items * 2, days=90):
            k = it["title"][:40]
            if k not in seen:
                seen.add(k)
                all_items.append({
                    "source": "CNBC",
                    "title": it["title"],
                    "link": it["link"],
                    "date": it["date"],
                    "summary": "",
                    "ticker_specific": True,
                })
        if len(all_items) >= max_items:
            break
    return all_items[:max_items]


# ── ④ TDnet（適時開示）銘柄別 ────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_tdnet_news(ticker_code: str, max_items: int = 20, months: int = 3) -> list[dict]:
    """
    株探の開示タブ（nmode=3）から過去N ヶ月分の適時開示を取得。

    ▼ データソース選定の根拠
      TDnet本家 (release.tdnet.info) は日付選択式で過去約1ヶ月分のみ。
      株探の開示タブ (kabutan.jp/stock/news?code=XXXX&nmode=3) は
      複数ページで数年分まで遡れるため、こちらを使用する。

    ▼ 取得戦略
      - 新しい順（デフォルト）で page=1 から順にたどる
      - 各行の datetime を見てカットオフより古くなったら終了
      - PDFリンクは kabutan.jp/disclosures/pdf/... 形式
    """
    import datetime as dt
    code = ticker_code.replace(".T", "")
    cutoff = dt.datetime.now() - dt.timedelta(days=months * 31)
    base_headers = {
        **_NEWS_HEADERS,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "ja,en-US;q=0.9",
        "Referer": "https://kabutan.jp/",
    }
    items = []
    page = 1

    while len(items) < max_items and page <= 10:  # 最大10ページ
        url = f"https://kabutan.jp/stock/news?code={code}&nmode=3&page={page}"
        try:
            r = requests.get(url, headers=base_headers, timeout=15)
            if r.status_code != 200:
                break
            html = r.text

            table_match = re.search(
                r'class="s_news_list[^"]*"[^>]*>(.*?)</table>',
                html, re.DOTALL
            )
            if not table_match:
                break
            table_html = table_match.group(1)
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL)
            if not rows:
                break

            found_on_page = 0
            hit_cutoff = False

            for row in rows:
                time_match = re.search(r'<time[^>]+datetime="([^"]+)"', row)
                if not time_match:
                    continue
                dt_raw = time_match.group(1)
                dt_disp = re.search(r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})', dt_raw)
                if not dt_disp:
                    continue
                date_str = f"{dt_disp.group(1)} {dt_disp.group(2)}"

                # カットオフチェック（新しい順なのでここ以降は全部古い）
                try:
                    row_dt = dt.datetime.strptime(date_str, "%Y-%m-%d %H:%M")
                    if row_dt < cutoff:
                        hit_cutoff = True
                        break
                except Exception:
                    pass

                link_match = re.search(
                    r'<a\s+href="(https://kabutan\.jp/disclosures/[^"]+)"[^>]*>\s*(.*?)\s*</a>',
                    row, re.DOTALL
                )
                if not link_match:
                    continue
                link  = link_match.group(1)
                title = re.sub(r'<[^>]+>', '', link_match.group(2)).strip()
                if len(title) < 3:
                    continue

                items.append({
                    "source": "TDnet（適時開示）",
                    "title": title,
                    "badge": "開示",
                    "badge_emoji": "🔴",
                    "link": link,
                    "date": date_str,
                    "summary": "📄 適時開示PDF",
                    "ticker_specific": True,
                })
                found_on_page += 1
                if len(items) >= max_items:
                    return items

            if hit_cutoff or found_on_page == 0:
                break
            page += 1

        except Exception:
            break

    return items


@st.cache_data(ttl=7200)
def ai_summarize_tdnet_pdf(pdf_url: str, title: str) -> str:
    """
    適時開示の内容をAIで詳細要約。
    株探の開示HTMLページ取得 -> PDFテキスト抽出 -> タイトルのみ の順でフォールバック。
    """
    page_text = ""
    source_desc = ""

    # 1) 株探の開示HTMLページ（/disclosures/pdf/ -> /disclosures/ に変換）
    try:
        html_url = pdf_url.replace("/disclosures/pdf/", "/disclosures/")
        if html_url != pdf_url:
            r = requests.get(html_url, headers={**_NEWS_HEADERS, "Referer": "https://kabutan.jp/"}, timeout=15)
            if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
                raw = re.sub(r'<script[^>]*>.*?</script>', ' ', r.text, flags=re.DOTALL)
                raw = re.sub(r'<style[^>]*>.*?</style>', ' ', raw, flags=re.DOTALL)
                raw = re.sub(r'<[^>]+>', ' ', raw)
                raw = re.sub(r'\s+', ' ', raw).strip()
                page_text = raw[:6000]
                source_desc = "株探開示ページ"
    except Exception:
        pass

    # 2) PDF直接取得（バイナリからテキスト部分を抽出）
    if not page_text:
        try:
            r = requests.get(pdf_url, headers={**_NEWS_HEADERS, "Referer": "https://kabutan.jp/"}, timeout=20)
            if r.status_code == 200:
                ct = r.headers.get("Content-Type", "")
                if "text/html" in ct:
                    raw = re.sub(r'<[^>]+>', ' ', r.text)
                    page_text = re.sub(r'\s+', ' ', raw).strip()[:6000]
                    source_desc = "開示HTML"
                elif "pdf" in ct.lower():
                    pdf_str = r.content.decode("latin-1", errors="ignore")
                    chunks = re.findall(r'BT\s*(.*?)\s*ET', pdf_str, re.DOTALL)
                    parts = []
                    for chunk in chunks:
                        parts.extend(re.findall(r'\(([^)]{1,200})\)', chunk))
                    page_text = " ".join(parts)[:6000]
                    source_desc = "PDFテキスト抽出"
        except Exception:
            pass

    if not page_text or len(page_text.strip()) < 50:
        page_text = "（本文取得不可。タイトルから推定）"
        source_desc = "タイトルのみ"

    prompt = f"""あなたは日本株の機関投資家向けアナリストです。
以下の適時開示情報を分析し、投資判断に役立つ詳細な要約を日本語で作成してください。

【開示タイトル】{title}

【開示内容】{page_text}

【要約フォーマット（合計400〜500文字）】

■ 開示種別: （業績修正 / 配当変更 / 決算発表 / 資本政策 / その他）

■ 主要な変更点:
  - 変更前→変更後の数値を具体的に（例: 営業利益 500億円→620億円、+24%）
  - 配当がある場合は1株あたりの金額も記載
  - 複数項目ある場合はすべて列挙

■ 背景・理由: なぜ修正・発表したか

■ 株価への影響予測:
  - ポジティブ / ネガティブ / 中立 とその理由
  - 市場が注目すべきポイント

■ 投資家へのアドバイス: 短期・中長期の観点で
"""
    try:
        comment, ai_name = generate_ai_comment(prompt)
        return comment + "\n\n_情報源: " + source_desc + " / AI: " + ai_name + "_"
    except Exception as e:
        return f"要約エラー: {e}"




# ── ⑤ Reuters 銘柄別（Google News経由・過去90日） ────────────────
@st.cache_data(ttl=600)
def fetch_reuters_stock_news(company_name: str, ticker_code: str, max_items: int = 10) -> list[dict]:
    """Reutersの銘柄関連記事をGoogle News RSS経由で取得（過去90日）"""
    code = ticker_code.replace(".T", "")
    en_name = _get_en_name(company_name)
    company_short = re.sub(r'[　（）()ＨＤホールディングス\s]', '', company_name)

    queries = []
    if en_name:
        queries.append(f"{en_name} site:reuters.com")
        queries.append(f"{en_name} Reuters")
    queries.append(f"{company_short} ロイター")
    queries.append(f"{code} Reuters")

    all_items = []
    seen = set()
    for q in queries:
        for it in _fetch_google_news_rss(q, "Reuters", max_items * 2, days=90):
            k = it["title"][:40]
            if k not in seen:
                seen.add(k)
                all_items.append({
                    "source": "Reuters JP",
                    "title": it["title"],
                    "link": it["link"],
                    "date": it["date"],
                    "summary": "",
                    "ticker_specific": True,
                })
        if len(all_items) >= max_items:
            break
    return all_items[:max_items]


# ── ⑥ 日経新聞 マーケット RSS（全体・Tab3用）───────────────────────
@st.cache_data(ttl=600)
def fetch_nikkei_market_rss(max_items: int = 8) -> list[dict]:
    """日経新聞マーケットニュース RSS（全体市況）"""
    url = "https://www.nikkei.com/rss/market.xml"
    try:
        r = requests.get(url, headers=_NEWS_HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        items = []
        for item in root.findall(".//item")[:max_items]:
            title   = item.findtext("title", "").strip()
            link    = item.findtext("link", "").strip()
            pubdate = item.findtext("pubDate", "").strip()
            if title:
                items.append({"source": "日経新聞", "title": title,
                              "link": link, "date": pubdate, "summary": ""})
        return items
    except Exception:
        return []


# ── ⑦ Reuters Japan RSS（全体・Tab3用）────────────────────────────
@st.cache_data(ttl=600)
def fetch_reuters_jp_rss(max_items: int = 8) -> list[dict]:
    """Reuters日本語マーケットニュース"""
    url = "https://feeds.reuters.com/reuters/JPBusinessNews"
    try:
        r = requests.get(url, headers=_NEWS_HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        items = []
        for item in root.findall(".//item")[:max_items]:
            title   = item.findtext("title", "").strip()
            link    = item.findtext("link", "").strip()
            pubdate = item.findtext("pubDate", "").strip()
            if title:
                items.append({"source": "Reuters JP", "title": title,
                              "link": link, "date": pubdate, "summary": ""})
        return items
    except Exception:
        return []


# ── ⑦ ソース別並列取得 ─────────────────────────────────────────
def fetch_news_by_source(
    ticker_code: str,
    company_name: str,
    max_per_source: int = 10,
) -> dict:
    """
    各ソースを並列取得し、ソース名をキーにした辞書で返す。
    {
      "Yahoo!Finance JP": [...],
      "株探(Kabutan)":    [...],
      "TDnet（適時開示）": [...],
      "日経新聞":          [...],
      "CNBC":             [...],
      "Reuters JP":       [...],
    }
    全体ニュース（日経・CNBC・Reuters）は銘柄名・コードを含む記事のみ残す。
    """
    import concurrent.futures
    code = ticker_code.replace(".T", "")

    # 銘柄マッチキーワード
    company_short = re.sub(r"[　ＨＤ（）()ホールディングス]", "", company_name)[:6]
    keywords = {code, company_name, company_short,
                company_name[:4], company_name.replace("ＨＤ", "").strip()}
    keywords = {k for k in keywords if len(k) >= 2}

    tasks = {
        "Yahoo!Finance JP":  lambda: fetch_yahoo_jp_news(code, max_per_source),
        "株探(Kabutan)":     lambda: fetch_kabutan_news(code, max_per_source),
        "TDnet（適時開示）": lambda: fetch_tdnet_news(code, max_items=30, months=3),
        "日経新聞":          lambda: fetch_nikkei_stock_news(company_name, code, max_per_source),
        "CNBC":              lambda: fetch_cnbc_news(company_name, code, max_per_source),
        "Reuters JP":        lambda: fetch_reuters_stock_news(company_name, code, max_per_source),
    }

    results_by_source = {k: [] for k in tasks}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(fn): key for key, fn in tasks.items()}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                items = future.result()
                results_by_source[key] = items
            except Exception:
                results_by_source[key] = []

    return results_by_source


# ── ⑧ AI によるニュース要約・センチメント ───────────────────────
def ai_news_summary(news_items, company_name: str, ticker: str) -> str:
    """
    ニュース一覧をAIで日本語要約・センチメント分析。
    news_items: list[dict] または list[str]（見出し文字列）を受け付ける。
    """
    if not news_items:
        return "ニュースが取得できませんでした。"

    if isinstance(news_items[0], str):
        headlines = "\n".join(news_items[:20])
    else:
        headlines = "\n".join(
            f"[{it['source']}] {it['title']}" for it in news_items[:20]
        )
    prompt = f"""
以下は日本株「{company_name}（{ticker}）」に関する最新ニュース・適時開示の見出しです。

{headlines}

投資家向けに以下を日本語300文字以内でまとめてください：
1. センチメント判定: 【強気 / 弱気 / 中立】
2. 注目イベントの要点
3. 株価への影響の可能性
"""
    try:
        comment, ai_name = generate_ai_comment(prompt)
        return f"{comment}\n\n_AI: {ai_name}_"
    except Exception as e:
        return f"AI分析エラー: {e}"


# ================================================================
# サイドバー
# ================================================================
with st.sidebar:
    st.header("⚙️ 分析パラメータ")
    years          = st.number_input("📅 過去何年で分析？", 1, 10, 3)
    risk_free_rate = st.number_input("📉 無リスク金利（%）", 0.0, 10.0, 1.0, step=0.1) / 100
    top_n          = st.number_input("📊 上位何社を表示？", 5, 50, 20, step=5)
    st.divider()

    st.header("📰 ニュース設定")
    news_max_per_source = st.slider("各ソースの最大取得件数", 3, 10, 5)
    show_news_sources = st.multiselect(
        "表示するニュースソース",
        ["Yahoo!Finance JP", "株探(Kabutan)", "TDnet（適時開示）", "日経新聞", "CNBC", "Reuters JP"],
        default=["Yahoo!Finance JP", "株探(Kabutan)", "TDnet（適時開示）", "日経新聞", "CNBC", "Reuters JP"],
    )
    st.divider()
    st.caption("データソース: Yahoo Finance, TDnet, 株探, みんかぶ, 日経, Reuters")

# ================================================================
# 銘柄マスタ
# ================================================================
ticker_name_map = {
    '1332.T': ('ニッスイ', '水産'),
    '1605.T': ('ＩＮＰＥＸ', '鉱業'),
    '1721.T': ('コムシスＨＤ', '建設'),
    '1801.T': ('大成建', '建設'),
    '1802.T': ('大林組', '建設'),
    '1803.T': ('清水建', '建設'),
    '1808.T': ('長谷工', '建設'),
    '1812.T': ('鹿島', '建設'),
    '1925.T': ('ハウス', '建設'),
    '1928.T': ('積ハウス', '建設'),
    '1963.T': ('日揮ＨＤ', '建設'),
    '2002.T': ('日清粉Ｇ', '食品'),
    '2269.T': ('明治ＨＤ', '食品'),
    '2282.T': ('日ハム', '食品'),
    '2413.T': ('エムスリー', 'サービス'),
    '2432.T': ('ディーエヌエ', 'サービス'),
    '2501.T': ('サッポロＨＤ', '食品'),
    '2502.T': ('アサヒ', '食品'),
    '2503.T': ('キリンＨＤ', '食品'),
    '2768.T': ('双日', '商社'),
    '2801.T': ('キッコマン', '食品'),
    '2802.T': ('味の素', '食品'),
    '2871.T': ('ニチレイ', '食品'),
    '2914.T': ('ＪＴ', '食品'),
    '3086.T': ('Ｊフロント', '小売業'),
    '3092.T': ('ＺＯＺＯ', '小売業'),
    '3099.T': ('三越伊勢丹', '小売業'),
    '3289.T': ('東急不ＨＤ', '不動産'),
    '3382.T': ('セブン＆アイ', '小売業'),
    '3401.T': ('帝人', '繊維'),
    '3402.T': ('東レ', '繊維'),
    '3405.T': ('クラレ', '化学'),
    '3407.T': ('旭化成', '化学'),
    '3436.T': ('ＳＵＭＣＯ', '非鉄・金属'),
    '3659.T': ('ネクソン', 'サービス'),
    '3861.T': ('王子ＨＤ', 'パルプ・紙'),
    '4004.T': ('レゾナック', '化学'),
    '4005.T': ('住友化', '化学'),
    '4021.T': ('日産化', '化学'),
    '4042.T': ('東ソー', '化学'),
    '4043.T': ('トクヤマ', '化学'),
    '4061.T': ('デンカ', '化学'),
    '4063.T': ('信越化', '化学'),
    '4151.T': ('協和キリン', '医薬品'),
    '4183.T': ('三井化学', '化学'),
    '4188.T': ('三菱ケミＧ', '化学'),
    '4208.T': ('ＵＢＥ', '化学'),
    '4307.T': ('野村総研', 'サービス'),
    '4324.T': ('電通グループ', 'サービス'),
    '4385.T': ('メルカリ', 'サービス'),
    '4452.T': ('花王', '化学'),
    '4502.T': ('武田', '医薬品'),
    '4503.T': ('アステラス', '医薬品'),
    '4506.T': ('住友ファーマ', '医薬品'),
    '4507.T': ('塩野義', '医薬品'),
    '4519.T': ('中外薬', '医薬品'),
    '4523.T': ('エーザイ', '医薬品'),
    '4543.T': ('テルモ', '精密機器'),
    '4568.T': ('第一三共', '医薬品'),
    '4578.T': ('大塚ＨＤ', '医薬品'),
    '4661.T': ('ＯＬＣ', 'サービス'),
    '4689.T': ('ラインヤフー', 'サービス'),
    '4704.T': ('トレンド', 'サービス'),
    '4751.T': ('サイバー', 'サービス'),
    '4755.T': ('楽天グループ', 'サービス'),
    '4901.T': ('富士フイルム', '化学'),
    '4902.T': ('コニカミノル', '精密機器'),
    '4911.T': ('資生堂', '化学'),
    '5019.T': ('出光興産', '石油'),
    '5020.T': ('ＥＮＥＯＳ', '石油'),
    '5101.T': ('浜ゴム', 'ゴム'),
    '5108.T': ('ブリヂストン', 'ゴム'),
    '5201.T': ('ＡＧＣ', '窯業'),
    '5214.T': ('日電硝', '窯業'),
    '5233.T': ('太平洋セメ', '窯業'),
    '5301.T': ('東海カーボン', '窯業'),
    '5332.T': ('ＴＯＴＯ', '窯業'),
    '5333.T': ('ガイシ', '窯業'),
    '5401.T': ('日本製鉄', '鉄鋼'),
    '5406.T': ('神戸鋼', '鉄鋼'),
    '5411.T': ('ＪＦＥ', '鉄鋼'),
    '5631.T': ('日製鋼', '機械'),
    '5706.T': ('三井金', '非鉄・金属'),
    '5711.T': ('三菱マ', '非鉄・金属'),
    '5713.T': ('住友鉱', '非鉄・金属'),
    '5714.T': ('ＤＯＷＡ', '非鉄・金属'),
    '5801.T': ('古河電', '非鉄・金属'),
    '5802.T': ('住友電', '非鉄・金属'),
    '5803.T': ('フジクラ', '非鉄・金属'),
    '5831.T': ('しずおかＦＧ', '銀行'),
    '6098.T': ('リクルート', 'サービス'),
    '6103.T': ('オークマ', '機械'),
    '6113.T': ('アマダ', '機械'),
    '6146.T': ('ディスコ', '精密機器'),
    '6178.T': ('日本郵政', 'サービス'),
    '6273.T': ('ＳＭＣ', '機械'),
    '6301.T': ('コマツ', '機械'),
    '6302.T': ('住友重', '機械'),
    '6305.T': ('日立建機', '機械'),
    '6326.T': ('クボタ', '機械'),
    '6361.T': ('荏原', '機械'),
    '6367.T': ('ダイキン', '機械'),
    '6471.T': ('日精工', '機械'),
    '6472.T': ('ＮＴＮ', '機械'),
    '6473.T': ('ジェイテクト', '機械'),
    '6479.T': ('ミネベア', '電気機器'),
    '6501.T': ('日立', '電気機器'),
    '6503.T': ('三菱電', '電気機器'),
    '6504.T': ('富士電機', '電気機器'),
    '6506.T': ('安川電', '電気機器'),
    '6526.T': ('ソシオネクス', '電気機器'),
    '6532.T': ('ベイカレント', 'サービス'),
    '6594.T': ('ニデック', '電気機器'),
    '6645.T': ('オムロン', '電気機器'),
    '6674.T': ('ＧＳユアサ', '電気機器'),
    '6701.T': ('ＮＥＣ', '電気機器'),
    '6702.T': ('富士通', '電気機器'),
    '6723.T': ('ルネサス', '電気機器'),
    '6724.T': ('エプソン', '電気機器'),
    '6752.T': ('パナＨＤ', '電気機器'),
    '6753.T': ('シャープ', '電気機器'),
    '6758.T': ('ソニーＧ', '電気機器'),
    '6762.T': ('ＴＤＫ', '電気機器'),
    '6770.T': ('アルプスアル', '電気機器'),
    '6841.T': ('横河電', '電気機器'),
    '6857.T': ('アドテスト', '電気機器'),
    '6861.T': ('キーエンス', '電気機器'),
    '6902.T': ('デンソー', '電気機器'),
    '6920.T': ('レーザーテク', '電気機器'),
    '6952.T': ('カシオ', '電気機器'),
    '6954.T': ('ファナック', '電気機器'),
    '6971.T': ('京セラ', '電気機器'),
    '6976.T': ('太陽誘電', '電気機器'),
    '6981.T': ('村田製', '電気機器'),
    '6988.T': ('日東電', '化学'),
    '7004.T': ('カナデビア', '機械'),
    '7011.T': ('三菱重', '機械'),
    '7012.T': ('川重', '造船'),
    '7013.T': ('ＩＨＩ', '機械'),
    '7186.T': ('コンコルディ', '銀行'),
    '7201.T': ('日産自', '自動車'),
    '7202.T': ('いすゞ', '自動車'),
    '7203.T': ('トヨタ', '自動車'),
    '7205.T': ('日野自', '自動車'),
    '7211.T': ('三菱自', '自動車'),
    '7261.T': ('マツダ', '自動車'),
    '7267.T': ('ホンダ', '自動車'),
    '7269.T': ('スズキ', '自動車'),
    '7270.T': ('ＳＵＢＡＲＵ', '自動車'),
    '7272.T': ('ヤマハ発', '自動車'),
    '7453.T': ('良品計画', '小売業'),
    '7731.T': ('ニコン', '精密機器'),
    '7733.T': ('オリンパス', '精密機器'),
    '7735.T': ('スクリン', '電気機器'),
    '7741.T': ('ＨＯＹＡ', '精密機器'),
    '7751.T': ('キヤノン', '電気機器'),
    '7752.T': ('リコー', '電気機器'),
    '7762.T': ('シチズン', '精密機器'),
    '7832.T': ('バンナムＨＤ', 'その他製造'),
    '7911.T': ('ＴＯＰＰＡＮ', 'その他製造'),
    '7912.T': ('大日印', 'その他製造'),
    '7951.T': ('ヤマハ', 'その他製造'),
    '7974.T': ('任天堂', 'サービス'),
    '8001.T': ('伊藤忠', '商社'),
    '8002.T': ('丸紅', '商社'),
    '8015.T': ('豊田通商', '商社'),
    '8031.T': ('三井物', '商社'),
    '8035.T': ('東エレク', '電気機器'),
    '8053.T': ('住友商', '商社'),
    '8058.T': ('三菱商', '商社'),
    '8233.T': ('高島屋', '小売業'),
    '8252.T': ('丸井Ｇ', '小売業'),
    '8253.T': ('クレセゾン', 'その他金融'),
    '8267.T': ('イオン', '小売業'),
    '8304.T': ('あおぞら銀', '銀行'),
    '8306.T': ('三菱ＵＦＪ', '銀行'),
    '8308.T': ('りそなＨＤ', '銀行'),
    '8309.T': ('三井住友トラ', '銀行'),
    '8316.T': ('三井住友ＦＧ', '銀行'),
    '8331.T': ('千葉銀', '銀行'),
    '8354.T': ('ふくおかＦＧ', '銀行'),
    '8411.T': ('みずほＦＧ', '銀行'),
    '8591.T': ('オリックス', 'その他金融'),
    '8601.T': ('大和', '証券'),
    '8604.T': ('野村', '証券'),
    '8630.T': ('ＳＯＭＰＯ', '保険'),
    '8697.T': ('日本取引所', 'その他金融'),
    '8725.T': ('ＭＳ＆ＡＤ', '保険'),
    '8750.T': ('第一生命ＨＤ', '保険'),
    '8766.T': ('東京海上', '保険'),
    '8795.T': ('Ｔ＆Ｄ', '保険'),
    '8801.T': ('三井不', '不動産'),
    '8802.T': ('菱地所', '不動産'),
    '8804.T': ('東建物', '不動産'),
    '8830.T': ('住友不', '不動産'),
    '9001.T': ('東武', '鉄道・バス'),
    '9005.T': ('東急', '鉄道・バス'),
    '9007.T': ('小田急', '鉄道・バス'),
    '9008.T': ('京王', '鉄道・バス'),
    '9009.T': ('京成', '鉄道・バス'),
    '9020.T': ('ＪＲ東日本', '鉄道・バス'),
    '9021.T': ('ＪＲ西日本', '鉄道・バス'),
    '9022.T': ('ＪＲ東海', '鉄道・バス'),
    '9064.T': ('ヤマトＨＤ', '陸運'),
    '9101.T': ('郵船', '海運'),
    '9104.T': ('商船三井', '海運'),
    '9107.T': ('川崎汽', '海運'),
    '9147.T': ('ＮＸＨＤ', '陸運'),
    '9201.T': ('ＪＡＬ', '空運'),
    '9202.T': ('ＡＮＡＨＤ', '空運'),
    '9432.T': ('ＮＴＴ', '通信'),
    '9433.T': ('ＫＤＤＩ', '通信'),
    '9434.T': ('ＳＢ', '通信'),
    '9501.T': ('東電ＨＤ', '電力'),
    '9502.T': ('中部電', '電力'),
    '9503.T': ('関西電', '電力'),
    '9531.T': ('東ガス', 'ガス'),
    '9532.T': ('大ガス', 'ガス'),
    '9602.T': ('東宝', 'サービス'),
    '9613.T': ('ＮＴＴデータ', '通信'),
    '9735.T': ('セコム', 'サービス'),
    '9766.T': ('コナミＧ', 'サービス'),
    '9843.T': ('ニトリＨＤ', '小売業'),
    '9983.T': ('ファストリ', '小売業'),
    '9984.T': ('ＳＢＧ', '通信'),
}

# ================================================================
# データ取得
# ================================================================
@st.cache_data(ttl=3600)
def get_price(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

@st.cache_data(ttl=3600)
def get_benchmark(start, end):
    df = yf.download("^N225", start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

# ================================================================
# メインタブ
# ================================================================
tab_analysis, tab_news, tab_market_news = st.tabs([
    "📊 パフォーマンス分析",
    "📰 銘柄別ニュース",
    "🌐 市場全体ニュース",
])

# ─── Tab1: パフォーマンス分析（既存機能） ────────────────────────
with tab_analysis:
    if st.button("▶ 分析実行", type="primary"):
        end_date   = datetime.today()
        start_date = end_date - relativedelta(years=int(years))

        with st.spinner("市場データ（日経225）を取得中..."):
            benchmark = get_benchmark(start_date, end_date)

        if benchmark.empty:
            st.error("市場データ取得失敗")
            st.stop()

        market_returns = benchmark["Close"].pct_change().dropna()

        results = []
        progress    = st.progress(0)
        status_text = st.empty()

        for i, (ticker, (name, sector)) in enumerate(ticker_name_map.items()):
            status_text.text(f"取得中: {name} ({ticker})")
            df = get_price(ticker, start_date, end_date)
            progress.progress((i + 1) / len(ticker_name_map))
            if df.empty:
                continue
            returns = df["Close"].pct_change().dropna()
            common  = returns.index.intersection(market_returns.index)
            if len(common) < 30:
                continue
            x = returns.loc[common].values.flatten()
            y = market_returns.loc[common].values.flatten()
            annual_return = x.mean() * 252
            annual_vol    = x.std() * np.sqrt(252)
            beta   = np.cov(x, y)[0][1] / np.var(y)
            sharpe = (annual_return - risk_free_rate) / annual_vol
            results.append({
                "企業名": name, "業種": sector,
                "年間平均リターン(%)": annual_return * 100,
                "年間リスク(%)": annual_vol * 100,
                "シャープレシオ": sharpe, "ベータ": beta,
            })

        progress.empty()
        status_text.empty()

        df_results = pd.DataFrame(results)
        if df_results.empty:
            st.error("データなし")
            st.stop()

        df_results = df_results.sort_values("シャープレシオ", ascending=False)

        st.subheader("📋 分析結果一覧")
        st.dataframe(
            df_results.style.format({
                "年間平均リターン(%)": "{:.2f}",
                "年間リスク(%)": "{:.2f}",
                "シャープレシオ": "{:.2f}",
                "ベータ": "{:.2f}",
            }),
            use_container_width=True,
        )

        top_n_int   = int(top_n)
        top_stocks  = df_results.head(top_n_int)

        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.bar(top_stocks["企業名"], top_stocks["シャープレシオ"], color="green")
        ax1.set_title(f"シャープレシオ 上位{top_n_int}社")
        ax1.set_ylabel("シャープレシオ")
        ax1.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.bar(top_stocks["企業名"], top_stocks["年間平均リターン(%)"], color="steelblue")
        ax2.set_title(f"年間平均リターン(%) 上位{top_n_int}社")
        ax2.set_ylabel("年間平均リターン(%)")
        ax2.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # AI コメント
        summary = top_stocks.head(5).to_string()
        prompt = f"""
以下は日本株のリスク・リターン分析結果です。
投資家向けに簡潔に300文字以内で評価してください。

{summary}
"""
        try:
            comment, ai_name = generate_ai_comment(prompt)
            st.subheader(f"🤖 AIコメント（{ai_name}）")
            st.write(comment)
        except Exception as e:
            st.warning(f"AI APIエラー: {e}")

# ─── Tab2: 銘柄別ニュース（ソース別独立表示）────────────────────
with tab_news:
    st.subheader("📰 銘柄別ニュース")

    # ── session_state 初期化 ──────────────────────────────────────
    # ニュースデータとAI要約結果をページ再レンダリング後も保持する
    if "news_by_src" not in st.session_state:
        st.session_state.news_by_src = {}
    if "news_ticker" not in st.session_state:
        st.session_state.news_ticker = ""
    if "tdnet_summaries" not in st.session_state:
        st.session_state.tdnet_summaries = {}   # key: "{ticker}_{idx}" -> summary str
    if "sentiment_result" not in st.session_state:
        st.session_state.sentiment_result = {}  # key: ticker -> summary str

    # 銘柄選択
    ticker_options = {f"{name}（{t}）": t for t, (name, _) in ticker_name_map.items()}
    default_idx = list(ticker_options.keys()).index("トヨタ（7203.T）") if "トヨタ（7203.T）" in ticker_options else 0
    selected_label  = st.selectbox("銘柄を選択", list(ticker_options.keys()), index=default_idx)
    selected_ticker = ticker_options[selected_label]
    selected_name   = ticker_name_map[selected_ticker][0]
    selected_code   = selected_ticker.replace(".T", "")

    # 銘柄が変わったらキャッシュをリセット
    if st.session_state.news_ticker != selected_ticker:
        st.session_state.news_by_src = {}
        st.session_state.tdnet_summaries = {}
        st.session_state.sentiment_result = {}
        st.session_state.news_ticker = selected_ticker

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_news = st.button("▶ ニュースを取得", type="primary")
    with col_btn2:
        run_ai = st.checkbox("🤖 AIによる総合センチメント分析", value=True)

    # ── ソース設定 ─────────────────────────────────────────────────
    SOURCE_CFG = {
        "Yahoo!Finance JP":  {"icon": "🟦", "label": "Yahoo!ファイナンス",  "desc": "銘柄RSS"},
        "株探(Kabutan)":     {"icon": "🟩", "label": "株探",               "desc": "銘柄専用ページ"},
        "TDnet（適時開示）": {"icon": "🔴", "label": "TDnet 適時開示",     "desc": "過去3ヶ月"},
        "日経新聞":          {"icon": "⬛", "label": "日経新聞",           "desc": "銘柄言及のみ"},
        "CNBC":              {"icon": "🟪", "label": "CNBC",               "desc": "英語・銘柄言及"},
        "Reuters JP":        {"icon": "🟫", "label": "Reuters",            "desc": "銘柄言及のみ"},
    }

    # ── ニュース取得（ボタン押下時のみ実行、結果はsession_stateへ）─
    if run_news:
        with st.spinner(f"{selected_name}（{selected_ticker}）のニュースを全ソースから並列取得中..."):
            st.session_state.news_by_src = fetch_news_by_source(
                selected_ticker, selected_name, news_max_per_source
            )
        st.session_state.tdnet_summaries = {}   # 銘柄再取得したら要約リセット
        st.session_state.sentiment_result = {}

    # ── 取得済みデータがあれば常に表示 ───────────────────────────
    news_by_src = st.session_state.news_by_src
    if not news_by_src:
        st.info("「▶ ニュースを取得」ボタンを押してください")
    else:
        total = sum(len(v) for v in news_by_src.values())
        st.caption(f"取得完了 — 合計 {total} 件")

        # ── サマリーバー ─────────────────────────────────────────
        cols_hdr = st.columns(len(SOURCE_CFG))
        for i, (src_key, cfg) in enumerate(SOURCE_CFG.items()):
            cnt = len(news_by_src.get(src_key, []))
            cols_hdr[i].metric(
                f"{cfg['icon']} {cfg['label']}",
                f"{cnt} 件",
                help=cfg["desc"],
            )

        st.divider()

        # ── ソースごとに独立セクション表示 ────────────────────────
        for src_key, cfg in SOURCE_CFG.items():
            items = news_by_src.get(src_key, [])
            icon  = cfg["icon"]
            label = cfg["label"]

            with st.expander(
                f"{icon} **{label}** — {len(items)} 件  `{cfg['desc']}`",
                expanded=(len(items) > 0),
            ):
                if not items:
                    st.caption("記事が取得できませんでした")
                    if src_key == "TDnet（適時開示）":
                        st.caption("※ 適時開示は決算期（3・6・9・12月）前後に集中します")
                    elif src_key in ("日経新聞", "CNBC", "Reuters JP"):
                        st.caption(f"※ {selected_name} に言及する記事が直近ありませんでした")
                    continue

                for idx_item, item in enumerate(items):
                    title       = item["title"]
                    link        = item.get("link", "")
                    date        = item.get("date", "")
                    badge_emoji = item.get("badge_emoji", "")
                    badge_text  = item.get("badge", "")

                    col_t, col_d = st.columns([5, 1])
                    with col_t:
                        if src_key == "TDnet（適時開示）":
                            # タイトル＋PDFリンク
                            if link:
                                st.markdown(f"🔴 [{title} 📄]({link})")
                            else:
                                st.markdown(f"🔴 {title} 📄")

                            # AI要約ボタン
                            summary_key = f"{selected_code}_{idx_item}"
                            btn_key     = f"btn_tdnet_{summary_key}"

                            if st.button("🤖 AIで要約", key=btn_key):
                                with st.spinner("PDF内容を取得・要約中..."):
                                    result = ai_summarize_tdnet_pdf(link, title)
                                # session_state に保存 → ボタン再押しでも消えない
                                st.session_state.tdnet_summaries[summary_key] = result

                            # 要約結果を表示（session_stateから読む）
                            if summary_key in st.session_state.tdnet_summaries:
                                st.info(st.session_state.tdnet_summaries[summary_key])

                        elif src_key == "株探(Kabutan)":
                            prefix = f"{badge_emoji}{badge_text} " if badge_text else ""
                            if link:
                                st.markdown(f"{prefix}[{title}]({link})")
                            else:
                                st.markdown(f"{prefix}{title}")

                            # 株探 AI要約ボタン
                            summary_key = f"kabutan_{selected_code}_{idx_item}"
                            btn_key     = f"btn_kabutan_{summary_key}"
                            if st.button("🤖 AIで要約", key=btn_key):
                                with st.spinner("記事内容を取得・要約中..."):
                                    result = ai_summarize_tdnet_pdf(link, title)
                                st.session_state.tdnet_summaries[summary_key] = result
                            if summary_key in st.session_state.tdnet_summaries:
                                st.info(st.session_state.tdnet_summaries[summary_key])

                        else:
                            prefix = f"{badge_emoji}{badge_text} " if badge_text else ""
                            if link:
                                st.markdown(f"{prefix}[{title}]({link})")
                            else:
                                st.markdown(f"{prefix}{title}")

                    with col_d:
                        if date:
                            date_short = date[:10] if len(date) >= 10 else date
                            st.caption(date_short)

                    if src_key != "TDnet（適時開示）":
                        st.markdown("---")

        # ── AI 総合センチメント ───────────────────────────────────
        if run_ai:
            st.divider()
            st.subheader(f"🤖 {selected_name} AI センチメント分析")

            # 既にsession_stateに結果があればそのまま表示
            if selected_ticker in st.session_state.sentiment_result:
                st.info(st.session_state.sentiment_result[selected_ticker])
            elif total > 0:
                all_headlines = [
                    f"[{src}] {it['title']}"
                    for src, its in news_by_src.items()
                    for it in its
                ]
                with st.spinner("AI分析中..."):
                    ai_result = ai_news_summary(all_headlines, selected_name, selected_ticker)
                st.session_state.sentiment_result[selected_ticker] = ai_result
                st.info(ai_result)


# ─── Tab3: 市場全体ニュース ──────────────────────────────────────
with tab_market_news:
    st.subheader("🌐 市場全体ニュース（日経・Reuters）")

    if st.button("▶ 市場ニュースを取得", type="primary"):
        import concurrent.futures

        with st.spinner("市場ニュースを取得中..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                f_nikkei  = ex.submit(fetch_nikkei_market_rss, 10)
                f_reuters = ex.submit(fetch_reuters_jp_rss, 10)
                nikkei_news  = f_nikkei.result()
                reuters_news = f_reuters.result()

        col_n, col_r = st.columns(2)

        with col_n:
            st.markdown("### ⬛ 日経新聞 マーケットニュース")
            if nikkei_news:
                for item in nikkei_news:
                    st.markdown(f"- [{item['title']}]({item['link']})")
                    if item.get("date"):
                        st.caption(f"  📅 {item['date']}")
            else:
                st.info("取得できませんでした（日経新聞RSSは会員制の場合があります）")

        with col_r:
            st.markdown("### 🟫 Reuters Japan ビジネスニュース")
            if reuters_news:
                for item in reuters_news:
                    st.markdown(f"- [{item['title']}]({item['link']})")
                    if item.get("date"):
                        st.caption(f"  📅 {item['date']}")
            else:
                st.info("取得できませんでした")

        # 全市場ニュースをAIで要約
        all_market = nikkei_news + reuters_news
        if all_market and st.checkbox("🤖 市場全体のAI要約を表示", value=True):
            headlines = "\n".join(f"[{n['source']}] {n['title']}" for n in all_market[:12])
            prompt = f"""
以下は本日の日本株マーケット関連ニュースです。

{headlines}

投資家向けに以下を日本語300文字以内でまとめてください：
1. 本日の市場全体のセンチメント（強気/弱気/中立）
2. 注目テーマ・セクター
3. 今後の注意点
"""
            with st.spinner("AI要約中..."):
                try:
                    comment, ai_name = generate_ai_comment(prompt)
                    st.subheader(f"🤖 市場全体AI要約（{ai_name}）")
                    st.info(comment)
                except Exception as e:
                    st.warning(f"AI APIエラー: {e}")
