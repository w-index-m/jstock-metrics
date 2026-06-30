import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from dateutil.relativedelta import relativedelta
from groq import Groq
import requests
import xml.etree.ElementTree as ET
import re
from io import StringIO

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
GROQ_MODEL = "llama-3.1-8b-instant"

# -----------------------------
# ページ設定
# -----------------------------
st.set_page_config(layout="wide", page_title="📈 日本株 分析ダッシュボード", page_icon="📈", initial_sidebar_state="expanded")
st.title("📈 日本株 シャープレシオ分析 + ニュース統合")

# ── アクセス計測 ────────────────────────────────────────────────
try:
    from analytics import track_pageview as _track_pv
    _track_pv("jstock")
except Exception:
    pass

# ── Google翻訳ウィジェット ────────────────────────────────────────
import streamlit.components.v1 as components
components.html(
    """
    <div id="google_translate_element" style="margin:4px 0 8px 0;display:inline-block;"></div>
    <script type="text/javascript">
    function googleTranslateElementInit(){
        new google.translate.TranslateElement({
            pageLanguage:'ja',includedLanguages:'ja,en,zh-TW',
            layout:google.translate.TranslateElement.InlineLayout.SIMPLE,
            autoDisplay:false
        },'google_translate_element');
    }
    </script>
    <script src="//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
    <style>
    .goog-te-gadget-simple{border:1px solid #d0d7de!important;border-radius:6px!important;padding:4px 8px!important;font-size:13px!important;background:#f6f8fa!important;}
    .goog-te-banner-frame{display:none!important;}body{top:0!important;}
    </style>
    """, height=50,
)

# ── 関連ダッシュボード リンクバー ────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #e8eaf6 0%, #e8f5e9 100%);
        border: 1px solid #c5cae9;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 14px;
        flex-wrap: wrap;
    ">
        <span style="font-weight:700;font-size:13px;color:#3949ab;white-space:nowrap;">
            🔗 関連ダッシュボード
        </span>
        <a href="https://usstock-metrics.streamlit.app/" target="_blank" rel="noopener noreferrer" style="
            display:inline-flex;align-items:center;gap:6px;
            background:linear-gradient(135deg,#1565c0,#1976d2);
            color:#fff;padding:7px 16px;border-radius:7px;text-decoration:none;
            font-size:13px;font-weight:700;
            box-shadow:0 2px 8px rgba(21,101,192,0.35);
            white-space:nowrap;
        ">🇺🇸&nbsp;USStockMetrics</a>
        <a href="https://windex.streamlit.app/" target="_blank" rel="noopener noreferrer" style="
            display:inline-flex;align-items:center;gap:6px;
            background:linear-gradient(135deg,#2e7d32,#43a047);
            color:#fff;padding:7px 16px;border-radius:7px;text-decoration:none;
            font-size:13px;font-weight:700;
            box-shadow:0 2px 8px rgba(46,125,50,0.35);
            white-space:nowrap;
        ">📊&nbsp;Market Dashboard</a>
        <span style="font-size:11px;color:#888;">
            各ダッシュボードで詳細な銘柄分析・指標をご覧いただけます
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# AI設定（Gemini優先 / Groqフォールバック）
# -----------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GROQ_API_KEY   = st.secrets.get("GROQ_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)
groq_client  = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def generate_ai_comment(prompt: str) -> tuple[str, str]:
    """Gemini → Groq フォールバック（安定版）"""
    # ---- Gemini ----
    try:
        response = gemini_model.generate_content(prompt)
        text = getattr(response, "text", None)
        if not text and hasattr(response, "candidates") and response.candidates:
            text = response.candidates[0].content.parts[0].text
        if text:
            return text, "Gemini"
    except Exception as e:
        print("Gemini Error:", e)

    # ---- Groq ----
    if groq_client is None:
        return "AIエラー（Gemini失敗・Groq未設定）", "Error"
    try:
        chat = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return chat.choices[0].message.content, "Groq"
    except Exception as e:
        print("Groq Error:", e)
        return f"Groqも失敗: {e}", "Error"


# ================================================================
# yfinance ユーティリティ（MultiIndex対応）
# ================================================================

def _yfdownload(ticker, start=None, end=None, period=None, progress=False, **kwargs):
    """yfinance v0.2以降のMultiIndex列を自動フラット化"""
    try:
        params = dict(progress=progress, auto_adjust=True)
        params.update(kwargs)
        if period:
            params["period"] = period
        else:
            params["start"] = start
            params["end"]   = end
        df = yf.download(ticker, **params)
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except Exception as e:
        import logging as _lg
        _lg.getLogger(__name__).warning(f"_yfdownload({ticker}): {e}")
        return pd.DataFrame()


def _to_series(col):
    """DataFrame列またはSeriesを確実に1次元Seriesに変換"""
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0]
    return col


# ================================================================
# 📰 ニュース取得モジュール
# ================================================================

_NEWS_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

@st.cache_data(ttl=600)
def fetch_yahoo_jp_news(ticker_code: str, max_items: int = 8) -> list[dict]:
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
            desc = re.sub(r"<[^>]+>", "", desc)[:100]
            if title:
                items.append({"source": "Yahoo!Finance JP", "title": title,
                              "link": link, "date": pubdate, "summary": desc})
        return items
    except Exception:
        return []


@st.cache_data(ttl=600)
def fetch_kabutan_news(ticker_code: str, max_items: int = 8) -> list[dict]:
    code = ticker_code.replace(".T", "")
    url = f"https://kabutan.jp/stock/news?code={code}"
    try:
        r = requests.get(url, headers=_NEWS_HEADERS, timeout=12)
        if r.status_code != 200:
            return []
        titles = re.findall(
            r'<a href="(/news/[^"]+)"[^>]*>([^<]{5,120})</a>', r.text
        )
        times  = re.findall(r'<time[^>]*>([^<]+)</time>', r.text)
        items = []
        for i, (path, title) in enumerate(titles[:max_items]):
            title = title.strip()
            if len(title) < 5 or "株探" in title:
                continue
            date = times[i].strip() if i < len(times) else ""
            items.append({
                "source": "株探(Kabutan)",
                "title": title,
                "link": f"https://kabutan.jp{path}",
                "date": date,
                "summary": "",
            })
        return items
    except Exception:
        return []


@st.cache_data(ttl=600)
def fetch_minkabu_news(ticker_code: str, max_items: int = 6) -> list[dict]:
    code = ticker_code.replace(".T", "")
    url = f"https://minkabu.jp/stock/{code}/news"
    try:
        r = requests.get(url, headers=_NEWS_HEADERS, timeout=12)
        if r.status_code != 200:
            return []
        titles = re.findall(
            r'<a[^>]+href="(/stock/[^"]+/news/[^"]+)"[^>]*>\s*<[^>]+>\s*([^<]{5,120})\s*</[^>]+>',
            r.text,
        )
        if not titles:
            titles = re.findall(
                r'class="[^"]*news[^"]*"[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>([^<]{5,120})</a>',
                r.text, re.DOTALL
            )
        dates = re.findall(r'\d{4}/\d{2}/\d{2}', r.text)
        items = []
        for i, (path, title) in enumerate(titles[:max_items]):
            title = title.strip()
            if len(title) < 5:
                continue
            link = f"https://minkabu.jp{path}" if path.startswith("/") else path
            date = dates[i] if i < len(dates) else ""
            items.append({
                "source": "みんかぶ",
                "title": title,
                "link": link,
                "date": date,
                "summary": "",
            })
        return items
    except Exception:
        return []


@st.cache_data(ttl=900)
def fetch_tdnet_news(ticker_code: str, max_items: int = 6) -> list[dict]:
    code = ticker_code.replace(".T", "")
    search_url = f"https://www.release.tdnet.info/inbs/I_main_00.html?target-code={code}"
    try:
        r = requests.get(search_url, headers=_NEWS_HEADERS, timeout=12)
        if r.status_code != 200:
            return []
        rows = re.findall(
            r'<td[^>]*class="[^"]*kjTitle[^"]*"[^>]*>(.*?)</td>.*?'
            r'href="([^"]+\.pdf)"',
            r.text, re.DOTALL
        )
        items = []
        for title_raw, pdf_path in rows[:max_items]:
            title = re.sub(r"<[^>]+>", "", title_raw).strip()
            if not title:
                continue
            link = f"https://www.release.tdnet.info{pdf_path}" if pdf_path.startswith("/") else pdf_path
            items.append({
                "source": "TDnet（適時開示）",
                "title": title,
                "link": link,
                "date": "",
                "summary": "📄 PDF",
            })
        return items
    except Exception:
        return []


@st.cache_data(ttl=600)
def fetch_nikkei_market_rss(max_items: int = 8) -> list[dict]:
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


@st.cache_data(ttl=600)
def fetch_reuters_jp_rss(max_items: int = 8) -> list[dict]:
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


MEMORY_KEYWORDS_JA = [
    "半導体", "メモリ", "DRAM", "NAND", "HBM", "LPDDR", "フラッシュメモリ",
    "シリコンウエハ", "ウエハ", "キオクシア", "マイクロン", "サムスン", "ハイニックス",
    "Western Digital", "AIチップ", "GPU", "積層", "露光装置", "エッチング",
    "TSMCニコン", "チップ", "製造装置", "フォトレジスト", "CMP",
]
MEMORY_KEYWORDS_EN = [
    "memory", "DRAM", "NAND", "HBM", "semiconductor", "Micron", "Samsung",
    "Hynix", "Kioxia", "flash memory", "wafer", "chip", "AI chip", "GPU memory",
    "3D NAND", "stacked memory", "TSMC", "fab", "chipmaker",
]

MEMORY_TICKERS = {
    '285A.T': ('キオクシアHD',    'NANDフラッシュ'),
    '8035.T': ('東京エレクトロン', '半導体製造装置'),
    '6857.T': ('アドバンテスト',   'メモリテスト装置'),
    '6920.T': ('レーザーテック',   '半導体検査'),
    '4063.T': ('信越化学',         'シリコンウエハ・材料'),
    '3436.T': ('SUMCO',            'シリコンウエハ'),
    '6723.T': ('ルネサス',         'MCU・半導体'),
    '6526.T': ('ソシオネクスト',   'SoC設計'),
    '6146.T': ('ディスコ',         '半導体切断装置'),
    '6758.T': ('ソニーＧ',         'CMOSセンサー'),
    '6762.T': ('ＴＤＫ',           '電子部品'),
    '6504.T': ('富士電機',         'パワー半導体'),
    '6702.T': ('富士通',           '半導体・IT'),
}

# テーマ別銘柄グループ（テーマ市場サマリー用）
THEME_GROUPS = {
    "半導体（主要）":   ["8035.T","6857.T","6920.T","6723.T","6526.T"],
    "DRAM・メモリ":    ["285A.T","8035.T","6857.T","3436.T"],
    "フラッシュメモリ": ["285A.T","6857.T"],
    "半導体製造装置":   ["8035.T","6920.T","6146.T"],
    "半導体露光装置":   ["8035.T","6920.T"],
    "パワー半導体":     ["6504.T","6723.T"],
    "ウエハ":          ["3436.T","4063.T"],
    "電子材料":        ["4063.T","4183.T","4188.T"],
    "AI・データセンター":["6702.T","6701.T","8035.T","6857.T"],
    "EV・電池":        ["6674.T","6752.T","6594.T"],
    "ロボット・FA":    ["6506.T","6861.T","6954.T"],
    "自動車・トヨタ系": ["7203.T","7267.T","6902.T","7269.T"],
    "防衛・宇宙":      ["7011.T","7013.T"],
    "インバウンド":    ["9602.T","9603.T"],
    "銀行":            ["8306.T","8316.T","8411.T","8354.T"],
    "不動産":          ["8802.T","8801.T"],
    "医療機器":        ["4543.T"],
    "商社":            ["8001.T","8002.T","8058.T"],
    "通信":            ["9432.T","9433.T","9984.T"],
    "エネルギー":      ["1605.T","5020.T"],
    "外食・食品":      ["2801.T","2802.T","2802.T"],
    "化学":            ["4063.T","4188.T","4005.T"],
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_memory_news_domestic(max_items: int = 15) -> list[dict]:
    """国内RSSからメモリ・半導体関連ニュースをキーワードフィルタで取得"""
    import xml.etree.ElementTree as _ET
    feeds = [
        ("https://www.nikkei.com/rss/market.xml",            "日経新聞"),
        ("https://feeds.reuters.com/reuters/JPBusinessNews",  "Reuters JP"),
        ("https://feeds.reuters.com/reuters/JPTechnologyNews","Reuters JP Tech"),
    ]
    results = []
    for url, source in feeds:
        try:
            r = requests.get(url, headers=_NEWS_HEADERS, timeout=10)
            if r.status_code != 200:
                continue
            root = _ET.fromstring(r.content)
            for item in root.findall(".//item"):
                title   = (item.findtext("title", "") or "").strip()
                link    = (item.findtext("link",  "") or "").strip()
                pubdate = (item.findtext("pubDate", "") or "").strip()
                desc    = (item.findtext("description", "") or "").strip()
                text    = title + " " + desc
                if any(kw in text for kw in MEMORY_KEYWORDS_JA):
                    results.append({
                        "source": source, "title": title,
                        "link": link, "date": pubdate,
                    })
        except Exception:
            continue
    return results[:max_items]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_memory_news_overseas(max_items: int = 15) -> list[dict]:
    """Reuters Technology RSS（英語）からメモリ業界ニュースを取得"""
    import xml.etree.ElementTree as _ET
    feeds = [
        ("https://feeds.reuters.com/reuters/technologyNews", "Reuters Technology"),
        ("https://feeds.reuters.com/reuters/businessNews",   "Reuters Business"),
    ]
    results = []
    for url, source in feeds:
        try:
            r = requests.get(url, headers=_NEWS_HEADERS, timeout=10)
            if r.status_code != 200:
                continue
            root = _ET.fromstring(r.content)
            for item in root.findall(".//item"):
                title   = (item.findtext("title", "") or "").strip()
                link    = (item.findtext("link",  "") or "").strip()
                pubdate = (item.findtext("pubDate", "") or "").strip()
                desc    = (item.findtext("description", "") or "").strip()
                text    = (title + " " + desc).lower()
                if any(kw.lower() in text for kw in MEMORY_KEYWORDS_EN):
                    results.append({
                        "source": source, "title": title,
                        "link": link, "date": pubdate,
                    })
        except Exception:
            continue
    return results[:max_items]


def fetch_all_news(ticker_code: str, max_per_source: int = 5) -> list[dict]:
    import concurrent.futures
    code = ticker_code.replace(".T", "")
    tasks = {
        "yahoo_jp":  lambda: fetch_yahoo_jp_news(code, max_per_source),
        "kabutan":   lambda: fetch_kabutan_news(code, max_per_source),
        "minkabu":   lambda: fetch_minkabu_news(code, max_per_source),
        "tdnet":     lambda: fetch_tdnet_news(code, max_per_source),
        "nikkei":    lambda: fetch_nikkei_market_rss(max_per_source),
        "reuters":   lambda: fetch_reuters_jp_rss(max_per_source),
    }
    all_items = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(fn): key for key, fn in tasks.items()}
        for future in concurrent.futures.as_completed(futures):
            try:
                all_items.extend(future.result())
            except Exception:
                pass
    seen, unique = set(), []
    for item in all_items:
        if item["title"] not in seen:
            seen.add(item["title"])
            unique.append(item)
    return unique


def ai_news_summary(news_items: list[dict], company_name: str, ticker: str) -> str:
    if not news_items:
        return "ニュースが取得できませんでした。"
    headlines = "\n".join(
        f"[{it['source']}] {it['title']}" for it in news_items[:15]
    )
    prompt = (
        f"以下は日本株「{company_name}({ticker})」に関する最新ニュースです。\n\n"
        f"{headlines}\n\n"
        "投資家向けに300文字以内でまとめてください:\n"
        "1. センチメント判定: 強気 / 弱気 / 中立\n"
        "2. 注目イベントの要点\n"
        "3. 株価への影響の可能性\n"
    )
    try:
        comment, ai_name = generate_ai_comment(prompt)
        return f"{comment}\n\n_AI: {ai_name}_"
    except Exception as e:
        return f"AI分析エラー: {e}"


# ================================================================
# 🔄 セクターローテーション分析モジュール
# ================================================================

def _batch_close(tickers: list, start_dt, end_dt) -> pd.DataFrame:
    """全銘柄を yf.download 1回のバッチ取得（Close列のみ返す）"""
    try:
        raw = yf.download(
            tickers,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True, threads=True,
        )
        if raw.empty:
            return pd.DataFrame()
        return raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    except Exception:
        return pd.DataFrame()


def _batch_ohlcv(tickers: list, start_dt, end_dt) -> dict:
    """Close / High / Low / Volume を1回のバッチ取得でまとめて返す"""
    try:
        raw = yf.download(
            tickers,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True, threads=True,
        )
        if raw.empty:
            return {}
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0).unique()
            return {
                "Close":  raw["Close"]  if "Close"  in lvl0 else pd.DataFrame(),
                "Volume": raw["Volume"] if "Volume" in lvl0 else pd.DataFrame(),
                "High":   raw["High"]   if "High"   in lvl0 else pd.DataFrame(),
                "Low":    raw["Low"]    if "Low"    in lvl0 else pd.DataFrame(),
            }
        return {"Close": raw, "Volume": pd.DataFrame(), "High": pd.DataFrame(), "Low": pd.DataFrame()}
    except Exception:
        return {}


@st.cache_data(ttl=1800)
def get_sector_performance(ticker_name_map: dict, period_days: int = 20) -> pd.DataFrame:
    from datetime import timedelta
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=period_days + 10)
    close_all = _batch_close(list(ticker_name_map.keys()), start_dt, end_dt)
    if close_all.empty:
        return pd.DataFrame()
    sector_returns: dict = {}
    for ticker, (name, sector) in ticker_name_map.items():
        if ticker not in close_all.columns:
            continue
        close = close_all[ticker].dropna()
        if len(close) < 2:
            continue
        ret = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
        sector_returns.setdefault(sector, []).append(float(ret))
    rows = []
    for sector, rets in sector_returns.items():
        rows.append({
            "業種": sector,
            "平均リターン(%)": np.mean(rets),
            "中央値リターン(%)": np.median(rets),
            "銘柄数": len(rets),
            "上昇銘柄数": sum(1 for r in rets if r > 0),
            "下落銘柄数": sum(1 for r in rets if r < 0),
        })
    df_result = pd.DataFrame(rows).sort_values("平均リターン(%)", ascending=False).reset_index(drop=True)
    df_result["騰落率(%)"] = df_result["平均リターン(%)"].round(2)
    df_result["上昇率(%)"] = (df_result["上昇銘柄数"] / df_result["銘柄数"] * 100).round(1)
    return df_result


@st.cache_data(ttl=1800)
def get_sector_timeseries(ticker_name_map: dict, days: int = 60) -> pd.DataFrame:
    from datetime import timedelta
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=days + 5)
    close_all = _batch_close(list(ticker_name_map.keys()), start_dt, end_dt)
    if close_all.empty:
        return pd.DataFrame()
    sector_price_data: dict = {}
    for ticker, (name, sector) in ticker_name_map.items():
        if ticker not in close_all.columns:
            continue
        close = close_all[ticker].dropna()
        if len(close) < 5:
            continue
        sector_price_data.setdefault(sector, []).append(close / close.iloc[0] * 100)
    sector_avg = {}
    for sector, series_list in sector_price_data.items():
        combined = pd.concat(series_list, axis=1)
        sector_avg[sector] = combined.mean(axis=1)
    df_ts = pd.DataFrame(sector_avg)
    df_ts.index = pd.to_datetime(df_ts.index)
    return df_ts.sort_index()


def plot_sector_bar(df_sector: pd.DataFrame, title: str) -> plt.Figure:
    df_sorted = df_sector.sort_values("平均リターン(%)", ascending=True)
    colors = ["#d32f2f" if v < 0 else "#388e3c" for v in df_sorted["平均リターン(%)"]]
    fig, ax = plt.subplots(figsize=(10, max(5, len(df_sorted) * 0.45)))
    bars = ax.barh(df_sorted["業種"], df_sorted["平均リターン(%)"], color=colors, edgecolor="none")
    for bar, val in zip(bars, df_sorted["平均リターン(%)"]):
        xpos = bar.get_width() + (0.05 if val >= 0 else -0.05)
        ha   = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}%", va="center", ha=ha, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("平均リターン (%)")
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    green_patch = mpatches.Patch(color="#388e3c", label="買われている（上昇）")
    red_patch   = mpatches.Patch(color="#d32f2f", label="売られている（下落）")
    ax.legend(handles=[green_patch, red_patch], loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


def plot_sector_timeseries(df_ts: pd.DataFrame, top_sectors: list, bottom_sectors: list) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    ax = axes[0]
    cmap = plt.colormaps["Greens"].resampled(len(top_sectors) + 2)
    for i, sec in enumerate(top_sectors):
        if sec in df_ts.columns:
            series = df_ts[sec].dropna()
            ax.plot(series.index, series - 100, label=sec, color=cmap(i + 2), linewidth=2.2)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_title("買われているセクター（累積リターン）", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("累積リターン (%)", fontsize=12)
    ax.legend(fontsize=12, loc="upper left", framealpha=0.9,
              bbox_to_anchor=(0, 1), borderaxespad=0)
    ax.tick_params(axis="x", rotation=30, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax = axes[1]
    cmap2 = plt.colormaps["Reds"].resampled(len(bottom_sectors) + 2)
    for i, sec in enumerate(bottom_sectors):
        if sec in df_ts.columns:
            series = df_ts[sec].dropna()
            ax.plot(series.index, series - 100, label=sec, color=cmap2(i + 2), linewidth=2.2)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_title("売られているセクター（累積リターン）", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("累積リターン (%)", fontsize=12)
    ax.legend(fontsize=12, loc="upper left", framealpha=0.9,
              bbox_to_anchor=(0, 1), borderaxespad=0)
    ax.tick_params(axis="x", rotation=30, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=2.0)
    return fig


def plot_sector_heatmap(df_multi: pd.DataFrame) -> plt.Figure:
    df_heat = df_multi.set_index("業種")[["1週間", "1ヶ月", "3ヶ月"]]
    df_heat = df_heat.sort_values("1ヶ月", ascending=False)
    vmax = max(abs(df_heat.values.max()), abs(df_heat.values.min()), 3)
    fig, ax = plt.subplots(figsize=(10, max(7, len(df_heat) * 0.48)))
    im = ax.imshow(df_heat.values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    # ── 上部ラベル（通常のxticks）
    ax.set_xticks(range(len(df_heat.columns)))
    ax.set_xticklabels(df_heat.columns, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", which="both", top=False, bottom=True,
                   labeltop=False, labelbottom=True, labelsize=12, pad=6)

    # ── 下部ラベル（ax2xaxisで上部にも表示）
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(len(df_heat.columns)))
    ax2.set_xticklabels(df_heat.columns, fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", which="both", top=True, bottom=False,
                    labeltop=True, labelbottom=False, labelsize=12, pad=6)

    # ── Y軸（業種名）
    ax.set_yticks(range(len(df_heat.index)))
    ax.set_yticklabels(df_heat.index, fontsize=10)

    # ── セル内テキスト
    for i in range(len(df_heat.index)):
        for j in range(len(df_heat.columns)):
            val = df_heat.values[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="リターン (%)", shrink=0.8)
    ax.set_title("セクター別リターン ヒートマップ（期間比較）",
                 fontsize=13, fontweight="bold", pad=36)
    plt.tight_layout(pad=2.0)
    return fig


# ================================================================
# 🔥 需給系モジュール
# ================================================================

@st.cache_data(ttl=1800)
def get_volume_surge(ticker_name_map: dict, surge_ratio: float = 2.0,
                     short_days: int = 5, base_days: int = 20) -> pd.DataFrame:
    from datetime import timedelta
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=base_days + 10)
    cols = _batch_ohlcv(list(ticker_name_map.keys()), start_dt, end_dt)
    close_all  = cols.get("Close",  pd.DataFrame())
    volume_all = cols.get("Volume", pd.DataFrame())
    if close_all.empty:
        return pd.DataFrame()
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            if ticker not in close_all.columns:
                continue
            close = close_all[ticker].dropna()
            vol   = volume_all[ticker].dropna() if not volume_all.empty and ticker in volume_all.columns else pd.Series()
            if len(close) < base_days or len(vol) < base_days:
                continue
            recent_avg = float(vol.iloc[-short_days:].mean())
            base_avg   = float(vol.iloc[-base_days:-short_days].mean())
            if base_avg == 0:
                continue
            ratio     = recent_avg / base_avg
            price_chg = (close.iloc[-1] - close.iloc[-short_days]) / close.iloc[-short_days] * 100
            if ratio >= surge_ratio:
                results.append({
                    "企業名": name, "業種": sector, "ティッカー": ticker,
                    "出来高倍率": round(ratio, 2),
                    "直近5日平均出来高": int(recent_avg),
                    "基準平均出来高": int(base_avg),
                    "株価変化率(5日%)": round(float(price_chg), 2),
                    "最新株価": round(float(close.iloc[-1]), 1),
                })
        except Exception:
            continue
    df_r = pd.DataFrame(results)
    if not df_r.empty:
        df_r = df_r.sort_values("出来高倍率", ascending=False).reset_index(drop=True)
    return df_r


@st.cache_data(ttl=1800)
def get_vwap_deviation(ticker_name_map: dict, days: int = 20) -> pd.DataFrame:
    from datetime import timedelta
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=days + 5)
    cols = _batch_ohlcv(list(ticker_name_map.keys()), start_dt, end_dt)
    close_all  = cols.get("Close",  pd.DataFrame())
    volume_all = cols.get("Volume", pd.DataFrame())
    if close_all.empty:
        return pd.DataFrame()
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            if ticker not in close_all.columns:
                continue
            close = close_all[ticker].dropna()
            vol   = volume_all[ticker].dropna() if not volume_all.empty and ticker in volume_all.columns else pd.Series()
            if len(close) < 5 or len(vol) < 5:
                continue
            idx   = close.index.intersection(vol.index)
            c, v  = close.loc[idx], vol.loc[idx]
            vwap  = (c * v).sum() / v.sum()
            cur   = float(c.iloc[-1])
            results.append({
                "企業名": name, "業種": sector, "ティッカー": ticker,
                "現在値": round(cur, 1),
                "VWAP": round(float(vwap), 1),
                "VWAP乖離率(%)": round((cur - float(vwap)) / float(vwap) * 100, 2),
            })
        except Exception:
            continue
    df_r = pd.DataFrame(results)
    if not df_r.empty:
        df_r = df_r.sort_values("VWAP乖離率(%)", ascending=False).reset_index(drop=True)
    return df_r


@st.cache_data(ttl=1800)
def get_price_volume_scatter(ticker_name_map: dict, days: int = 20) -> pd.DataFrame:
    from datetime import timedelta
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=days + 10)
    cols = _batch_ohlcv(list(ticker_name_map.keys()), start_dt, end_dt)
    close_all  = cols.get("Close",  pd.DataFrame())
    volume_all = cols.get("Volume", pd.DataFrame())
    if close_all.empty:
        return pd.DataFrame()
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            if ticker not in close_all.columns:
                continue
            close  = close_all[ticker].dropna()
            volume = volume_all[ticker].dropna() if not volume_all.empty and ticker in volume_all.columns else pd.Series()
            if len(close) < 5 or len(volume) < 10:
                continue
            price_chg = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
            vol_chg   = (volume.iloc[-5:].mean() - volume.iloc[:5].mean()) / (volume.iloc[:5].mean() + 1) * 100
            results.append({
                "企業名": name, "業種": sector,
                "株価騰落率(%)": round(float(price_chg), 2),
                "出来高変化率(%)": round(float(vol_chg), 2),
            })
        except Exception:
            continue
    return pd.DataFrame(results)


def plot_pv_scatter(df: pd.DataFrame) -> None:
    """Price x Volume 散布図（Plotly・ホバーで銘柄名・数値表示）"""
    if df.empty:
        st.warning("データなし")
        return

    try:
        import plotly.graph_objects as go
        import plotly.express as px

        df = df.copy()
        # 原点からの距離で外れ値上位25銘柄にフロートラベルを付与
        # さらに注目銘柄は距離に関わらず必ずラベル表示
        _PINNED_LABELS = {"フジクラ", "JX金属", "太陽誘電"}
        df["_dist"] = np.sqrt(df["株価騰落率(%)"]**2 + df["出来高変化率(%)"]**2)
        top_idx = df.nlargest(min(25, len(df)), "_dist").index
        df["_label"] = ""
        df.loc[top_idx, "_label"] = df.loc[top_idx, "企業名"]
        pinned_idx = df[df["企業名"].isin(_PINNED_LABELS)].index
        df.loc[pinned_idx, "_label"] = df.loc[pinned_idx, "企業名"]

        x_max = df["出来高変化率(%)"].max()
        x_min = df["出来高変化率(%)"].min()
        y_max = df["株価騰落率(%)"].max()
        y_min = df["株価騰落率(%)"].min()

        fig = px.scatter(
            df,
            x="出来高変化率(%)",
            y="株価騰落率(%)",
            color="業種",
            text="_label",
            hover_name="企業名",
            hover_data={
                "業種": True,
                "株価騰落率(%)":  ":.2f",
                "出来高変化率(%)": ":.2f",
                "_label": False,
                "_dist": False,
            },
            title="Price x Volume マップ（セクター別）― ホバーで銘柄名・数値表示",
            height=650,
            color_discrete_sequence=px.colors.qualitative.Light24,
        )

        # 軸線
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                      line_width=1, opacity=0.6)
        fig.add_vline(x=0, line_dash="dash", line_color="gray",
                      line_width=1, opacity=0.6)

        # 4象限ラベル
        quad_labels = [
            (x_max * 0.65, y_max * 0.85, "株高+出来高増<br>（本命上昇）",    "#388e3c", "rgba(232,245,233,0.85)"),
            (x_min * 0.65, y_max * 0.85, "株高+出来高減<br>（戻り弱い）",    "#f57c00", "rgba(255,243,224,0.85)"),
            (x_max * 0.65, y_min * 0.85, "株安+出来高増<br>（売り圧力）",    "#d32f2f", "rgba(255,235,238,0.85)"),
            (x_min * 0.65, y_min * 0.85, "株安+出来高減<br>（静かな下落）",  "#9e9e9e", "rgba(245,245,245,0.85)"),
        ]
        for x, y, text, color, bgcolor in quad_labels:
            fig.add_annotation(
                x=x, y=y, text=text,
                showarrow=False,
                font=dict(color=color, size=12, family="Arial"),
                bgcolor=bgcolor,
                bordercolor=color,
                borderwidth=1,
                borderpad=6,
            )

        fig.update_traces(
            marker=dict(size=9, opacity=0.82, line=dict(width=0.5, color="gray")),
            textposition="top center",
            textfont=dict(size=8, color="rgba(0,0,0,0.72)"),
        )
        fig.update_layout(
            xaxis_title="出来高変化率 (%)",
            yaxis_title="株価騰落率 (%)",
            legend=dict(
                orientation="v", x=1.01, y=1,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="lightgray", borderwidth=1,
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=13,
                font_family="Arial",
                namelength=-1,
            ),
            margin=dict(r=180),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(gridcolor="rgba(0,0,0,0.07)", zeroline=False)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.07)", zeroline=False)

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # plotlyが無い場合はmatplotlibにフォールバック
        pass
    except Exception as _pv_err:
        st.error(f"チャート描画エラー: {_pv_err}")
        return
    else:
        return  # plotly成功時はここで終了

    # matplotlib fallback（plotly未インストール時のみ到達）
        sectors = df["業種"].unique()
        cmap = plt.colormaps["tab20"].resampled(len(sectors))
        sector_color = {sec: cmap(i) for i, sec in enumerate(sectors)}
        fig2, ax = plt.subplots(figsize=(14, 8))
        for sec in sectors:
            sub = df[df["業種"] == sec]
            ax.scatter(sub["出来高変化率(%)"], sub["株価騰落率(%)"],
                       label=sec, color=sector_color[sec], s=60, alpha=0.8)
            for _, row in sub.iterrows():
                if abs(row["株価騰落率(%)"]) > df["株価騰落率(%)"].std() * 1.2 or \
                   abs(row["出来高変化率(%)"]) > df["出来高変化率(%)"].std() * 1.2:
                    ax.annotate(row["企業名"],
                                (row["出来高変化率(%)"], row["株価騰落率(%)"]),
                                fontsize=7, alpha=0.85,
                                xytext=(4, 4), textcoords="offset points")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("出来高変化率 (%)", fontsize=12)
        ax.set_ylabel("株価騰落率 (%)", fontsize=12)
        ax.set_title("Price x Volume マップ（セクター別）", fontsize=13, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)


# ================================================================
# 📊 価格パターン系モジュール
# ================================================================

@st.cache_data(ttl=1800)
def get_52week_highlow(ticker_name_map: dict) -> pd.DataFrame:
    from datetime import timedelta
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=365)
    cols = _batch_ohlcv(list(ticker_name_map.keys()), start_dt, end_dt)
    close_all = cols.get("Close", pd.DataFrame())
    high_all  = cols.get("High",  pd.DataFrame())
    low_all   = cols.get("Low",   pd.DataFrame())
    if close_all.empty:
        return pd.DataFrame()
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            if ticker not in close_all.columns:
                continue
            close = close_all[ticker].dropna()
            if len(close) < 50:
                continue
            high_52w = float(high_all[ticker].dropna().max()) if not high_all.empty and ticker in high_all.columns else float(close.max())
            low_52w  = float(low_all[ticker].dropna().min())  if not low_all.empty  and ticker in low_all.columns  else float(close.min())
            current  = float(close.iloc[-1])
            results.append({
                "企業名": name, "業種": sector,
                "現在値": round(current, 1),
                "52週高値": round(high_52w, 1),
                "52週安値": round(low_52w, 1),
                "高値からの乖離(%)": round((current - high_52w) / high_52w * 100, 2),
                "安値からの乖離(%)": round((current - low_52w)  / low_52w  * 100, 2),
                "新高値": "新高値" if current >= high_52w * 0.995 else "",
                "新安値": "新安値" if current <= low_52w  * 1.005 else "",
            })
        except Exception:
            continue
    return pd.DataFrame(results)


@st.cache_data(ttl=1800)
def get_ma_deviation(ticker_name_map: dict) -> pd.DataFrame:
    from datetime import timedelta
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=250)
    close_all = _batch_close(list(ticker_name_map.keys()), start_dt, end_dt)
    if close_all.empty:
        return pd.DataFrame()
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            if ticker not in close_all.columns:
                continue
            close = close_all[ticker].dropna()
            if len(close) < 200:
                continue
            cur  = float(close.iloc[-1])
            ma25 = float(close.rolling(25).mean().iloc[-1])
            ma75 = float(close.rolling(75).mean().iloc[-1])
            ma200= float(close.rolling(200).mean().iloc[-1])
            results.append({
                "企業名": name, "業種": sector,
                "現在値": round(cur, 1),
                "25日MA乖離(%)":  round((cur - ma25)  / ma25  * 100, 2),
                "75日MA乖離(%)":  round((cur - ma75)  / ma75  * 100, 2),
                "200日MA乖離(%)": round((cur - ma200) / ma200 * 100, 2),
            })
        except Exception:
            continue
    df_r = pd.DataFrame(results)
    if not df_r.empty:
        df_r = df_r.sort_values("25日MA乖離(%)", ascending=False).reset_index(drop=True)
    return df_r


@st.cache_data(ttl=1800)
def get_cross_signals(ticker_name_map: dict, lookback_days: int = 10) -> pd.DataFrame:
    from datetime import timedelta
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=120)
    close_all = _batch_close(list(ticker_name_map.keys()), start_dt, end_dt)
    if close_all.empty:
        return pd.DataFrame()
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            if ticker not in close_all.columns:
                continue
            close = close_all[ticker].dropna()
            if len(close) < 75:
                continue
            diff = close.rolling(25).mean() - close.rolling(75).mean()
            for i in range(max(1, len(diff) - lookback_days), len(diff)):
                if pd.isna(diff.iloc[i]) or pd.isna(diff.iloc[i-1]):
                    continue
                if diff.iloc[i-1] < 0 and diff.iloc[i] >= 0:
                    results.append({"企業名": name, "業種": sector, "シグナル": "ゴールデンクロス",
                                    "発生日": str(diff.index[i])[:10], "現在値": round(float(close.iloc[-1]), 1)})
                    break
                elif diff.iloc[i-1] > 0 and diff.iloc[i] <= 0:
                    results.append({"企業名": name, "業種": sector, "シグナル": "デッドクロス",
                                    "発生日": str(diff.index[i])[:10], "現在値": round(float(close.iloc[-1]), 1)})
                    break
        except Exception:
            continue
    return pd.DataFrame(results)


# ================================================================
# 💡 モメンタム・相関分析モジュール
# ================================================================

@st.cache_data(ttl=1800)
def get_dow_of_week_pattern(ticker_name_map: dict, days: int = 180) -> pd.DataFrame:
    from datetime import timedelta
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=days)
    close_all = _batch_close(list(ticker_name_map.keys()), start_dt, end_dt)
    if close_all.empty:
        return pd.DataFrame()
    dow_map = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金"}
    sector_dow: dict = {}
    for ticker, (name, sector) in ticker_name_map.items():
        if ticker not in close_all.columns:
            continue
        try:
            close = close_all[ticker].dropna()
            if len(close) < 20:
                continue
            ret = close.pct_change().dropna() * 100
            ret.index = pd.to_datetime(ret.index)
            for dow_num, dow_label in dow_map.items():
                avg = float(ret[ret.index.dayofweek == dow_num].mean())
                sector_dow.setdefault((sector, dow_label), []).append(avg)
        except Exception:
            continue
    rows = [{"業種": s, "曜日": d, "平均リターン(%)": round(np.mean(vals), 4)}
            for (s, d), vals in sector_dow.items()]
    df_long = pd.DataFrame(rows)
    if df_long.empty:
        return df_long
    df_pivot = df_long.pivot(index="業種", columns="曜日", values="平均リターン(%)")
    return df_pivot.reindex(columns=[d for d in ["月","火","水","木","金"] if d in df_pivot.columns])


@st.cache_data(ttl=1800)
def get_correlation_divergence(ticker_name_map: dict, days: int = 60,
                                corr_window: int = 20) -> pd.DataFrame:
    from datetime import timedelta
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=days + 10)
    tickers  = list(ticker_name_map.keys()) + ["^N225"]
    close_all = _batch_close(tickers, start_dt, end_dt)
    if close_all.empty or "^N225" not in close_all.columns:
        return pd.DataFrame()
    market_ret = close_all["^N225"].pct_change().dropna()
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            if ticker not in close_all.columns:
                continue
            close = close_all[ticker].dropna()
            if len(close) < corr_window + 5:
                continue
            ret    = close.pct_change().dropna()
            common = ret.index.intersection(market_ret.index)
            if len(common) < corr_window + 5:
                continue
            r, m        = ret.loc[common], market_ret.loc[common]
            corr_long   = float(r.corr(m))
            corr_recent = float(r.iloc[-corr_window:].corr(m.iloc[-corr_window:]))
            price_chg   = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100
            results.append({
                "企業名": name, "業種": sector,
                "長期相関": round(corr_long, 3),
                "直近相関": round(corr_recent, 3),
                "相関乖離度": round(corr_long - corr_recent, 3),
                "直近5日株価変化(%)": round(float(price_chg), 2),
            })
        except Exception:
            continue
    df_r = pd.DataFrame(results)
    if not df_r.empty:
        df_r = df_r.sort_values("相関乖離度", ascending=False).reset_index(drop=True)
    return df_r


@st.cache_data(ttl=1800)
def get_momentum_score(ticker_name_map: dict) -> pd.DataFrame:
    from datetime import timedelta
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=30)
    cols = _batch_ohlcv(list(ticker_name_map.keys()), start_dt, end_dt)
    close_all  = cols.get("Close",  pd.DataFrame())
    volume_all = cols.get("Volume", pd.DataFrame())
    if close_all.empty:
        return pd.DataFrame()
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            if ticker not in close_all.columns:
                continue
            close = close_all[ticker].dropna()
            if len(close) < 10:
                continue
            vol = volume_all[ticker].dropna() if not volume_all.empty and ticker in volume_all.columns else pd.Series()
            price_chg = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
            vol_chg   = (vol.iloc[-5:].mean() - vol.mean()) / (vol.mean() + 1) * 100 if len(vol) >= 5 else 0.0
            score     = float(price_chg) * np.log1p(max(float(vol_chg), 0) / 100 + 1)
            results.append({
                "企業名": name, "業種": sector,
                "モメンタムスコア": round(score, 3),
                "株価騰落率(%)": round(float(price_chg), 2),
                "出来高変化率(%)": round(float(vol_chg), 2),
                "現在値": round(float(close.iloc[-1]), 1),
            })
        except Exception:
            continue
    df_r = pd.DataFrame(results)
    if not df_r.empty:
        df_r = df_r.sort_values("モメンタムスコア", ascending=False).reset_index(drop=True)
    return df_r


def plot_dow_heatmap(df_pivot: pd.DataFrame) -> plt.Figure:
    vmax = max(abs(df_pivot.values[~np.isnan(df_pivot.values)]).max(), 0.1)
    fig, ax = plt.subplots(figsize=(8, max(5, len(df_pivot) * 0.4)))
    im = ax.imshow(df_pivot.values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(df_pivot.columns)))
    ax.set_xticklabels(df_pivot.columns, fontsize=11)
    ax.set_yticks(range(len(df_pivot.index)))
    ax.set_yticklabels(df_pivot.index, fontsize=9)
    for i in range(len(df_pivot.index)):
        for j in range(len(df_pivot.columns)):
            val = df_pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, label="平均リターン (%)", shrink=0.8)
    ax.set_title("曜日別平均リターン ヒートマップ（セクター別）", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig


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
        ["Yahoo!Finance JP", "株探(Kabutan)", "みんかぶ", "TDnet（適時開示）", "日経新聞", "Reuters JP"],
        default=["Yahoo!Finance JP", "株探(Kabutan)", "TDnet（適時開示）", "日経新聞", "Reuters JP"],
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
    '285A.T': ('キオクシアＨＤ', '電気機器'),
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
@st.cache_data(ttl=3600, show_spinner=False)
def get_price(ticker, start, end):
    return _yfdownload(ticker, start=start, end=end)

@st.cache_data(ttl=3600, show_spinner=False)
def get_benchmark(start, end):
    return _yfdownload("^N225", start=start, end=end)


@st.cache_data(ttl=1800, show_spinner=False)
def compute_sharpe_all(
    ticker_map_items: tuple, start_str: str, end_str: str, rfr: float
) -> pd.DataFrame:
    """全銘柄のシャープレシオ・ベータ・アルファをバッチダウンロードで一括計算。
    225銘柄の個別ダウンロードを1回のyf.downloadに集約し初回ロードを高速化する。"""
    tickers  = [t for t, _ in ticker_map_items]
    name_map = {t: info for t, info in ticker_map_items}
    try:
        raw = yf.download(
            tickers + ["^N225"], start=start_str, end=end_str,
            progress=False, auto_adjust=True, threads=True,
        )
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    close_all = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if "^N225" not in close_all.columns:
        return pd.DataFrame()
    market_ret = close_all["^N225"].dropna().pct_change().dropna()
    market_annual = float(market_ret.mean()) * 252
    results = []
    for ticker in tickers:
        if ticker not in close_all.columns:
            continue
        name, sector = name_map[ticker]
        close = close_all[ticker].dropna()
        if len(close) < 2:
            continue
        ret = close.pct_change().dropna()
        common = ret.index.intersection(market_ret.index)
        if len(common) < 30:
            continue
        x = ret.loc[common].to_numpy(dtype=float)
        y = market_ret.loc[common].to_numpy(dtype=float)
        annual_return = x.mean() * 252
        annual_vol    = x.std() * np.sqrt(252)
        if annual_vol == 0:
            continue
        try:
            beta = np.cov(x, y)[0][1] / np.var(y)
        except Exception:
            beta = 0.0
        results.append({
            "企業名":              name,
            "業種":                sector,
            "年間平均リターン(%)": round(annual_return * 100, 2),
            "年間リスク(%)":       round(annual_vol * 100, 2),
            "シャープレシオ":      round((annual_return - rfr) / annual_vol, 4),
            "ベータ":              round(beta, 4),
            "アルファ(%)":         round((annual_return - beta * market_annual) * 100, 2),
        })
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("シャープレシオ", ascending=False).reset_index(drop=True)
    return df

# ================================================================
# メインタブ
# ================================================================

# ================================================================
# デフォルトパラメータで事前計算（自動実行用）
# ================================================================
end_date   = datetime.today()
start_date = end_date - relativedelta(years=3)  # デフォルト3年

# ================================================================
# 起動時 並列プリフェッチ（キャッシュウォームアップ）
# compute_sharpe_all と fetch_all_ticker_info_bulk を同時実行し
# 後続セクションがキャッシュ済みデータを即座に取得できるようにする
# ================================================================
import concurrent.futures as _cf_boot
_boot_tuple = tuple(ticker_name_map.items())
_boot_s     = start_date.strftime("%Y-%m-%d")
_boot_e     = end_date.strftime("%Y-%m-%d")
# compute_sharpe_all をここでキャッシュウォーム（fetch_all_ticker_info_bulk は後段で定義後に別途実行）
with st.spinner("📊 シャープレシオデータを取得中（初回のみ）..."):
    compute_sharpe_all(_boot_tuple, _boot_s, _boot_e, risk_free_rate)


# ─── Tab1: パフォーマンス分析 ────────────────────────────────────

st.markdown("""<div style="background:#1565c0;color:white;padding:12px 22px;border-radius:8px;margin:28px 0 4px 0;font-size:18px;font-weight:bold;">
📊 A.&nbsp;コア分析 &nbsp;<span style="font-size:12px;font-weight:400;opacity:.88">シャープレシオ・アルファ・ベータ</span></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
st.header("📊 パフォーマンス分析")
st.divider()
with st.spinner("全銘柄データを一括取得・計算中（初回のみ時間がかかります）..."):
    df_results = compute_sharpe_all(
        tuple(ticker_name_map.items()),
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        risk_free_rate,
    )

if df_results.empty:
    st.warning("分析データが取得できませんでした。しばらく後に再度お試しください。")
else:
    st.subheader("📋 分析結果一覧")

    def _color_alpha_cell(val):
        if isinstance(val, float):
            if val > 5:  return "color:#1a7f37;font-weight:bold"
            elif val > 0: return "color:#388e3c"
            elif val < 0: return "color:#d1242f"
        return ""

    st.dataframe(
        df_results.style.format({
            "年間平均リターン(%)": "{:.2f}",
            "年間リスク(%)": "{:.2f}",
            "シャープレシオ": "{:.2f}",
            "ベータ": "{:.2f}",
            "アルファ(%)": "{:+.2f}",
        }).map(_color_alpha_cell, subset=["アルファ(%)"]),
        use_container_width=True,
    )

    top_n_disp = int(top_n)
    top_stocks = df_results.head(top_n_disp)

    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(top_stocks["企業名"], top_stocks["シャープレシオ"], color="green")
    ax1.set_title(f"シャープレシオ 上位{top_n_disp}社")
    ax1.set_ylabel("シャープレシオ")
    ax1.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.bar(top_stocks["企業名"], top_stocks["年間平均リターン(%)"], color="steelblue")
    ax2.set_title(f"年間平均リターン(%) 上位{top_n_disp}社")
    ax2.set_ylabel("年間平均リターン(%)")
    ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    summary = top_stocks.head(5).to_string()
    prompt = (
        "以下は日本株のリスク・リターン分析結果です。\n"
        "投資家向けに簡潔に300文字以内で評価してください。\n\n"
        f"{summary}\n"
    )
    try:
        comment, ai_name = generate_ai_comment(prompt)
        st.subheader(f"🤖 AIコメント（{ai_name}）")
        st.write(comment)
    except Exception as e:
        st.warning(f"AI APIエラー: {e}")


# ─────────────────────────────────────────────────────────────────
st.header("🎯 アルファ・ベータ分析")
st.divider()
st.caption(
    "**α（アルファ）**= 市場平均を超えた銘柄固有の超過リターン。"
    "**β（ベータ）**= 市場との連動性。"
    "理想は「高α × 低β」＝市場に左右されず独自に稼ぐ銘柄。"
)

# df_results が存在するときのみ表示
try:
    _ab_ok = not df_results.empty
except Exception:
    _ab_ok = False

if not _ab_ok:
    st.info("パフォーマンス分析を先に実行してください（上のセクションで自動実行されます）")
else:
    # ── アルファ計算（df_resultsに既に含まれている）────────────────
    df_ab = df_results.copy()
    # アルファ列が無い場合のみ計算（念のため）
    if "アルファ(%)" not in df_ab.columns:
        _bench_close2 = _to_series(benchmark["Close"])
        _market_annual = float(
            (_bench_close2.iloc[-1] - _bench_close2.iloc[0]) / _bench_close2.iloc[0]
        )
        df_ab["アルファ(%)"] = (
            df_ab["年間平均リターン(%)"] / 100
            - df_ab["ベータ"] * _market_annual
        ) * 100
        df_ab["アルファ(%)"] = df_ab["アルファ(%)"].round(2)

    # ── メトリクス ────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("高アルファ銘柄数（α>0）",
              f"{(df_ab['アルファ(%)'] > 0).sum()}社")
    m2.metric("平均アルファ",
              f"{df_ab['アルファ(%)'].mean():.2f}%")
    m3.metric("最大アルファ",
              f"{df_ab['アルファ(%)'].max():.2f}%",
              df_ab.loc[df_ab['アルファ(%)'].idxmax(), '企業名'])
    m4.metric("低β高α銘柄数（β<1 & α>0）",
              f"{((df_ab['ベータ'] < 1) & (df_ab['アルファ(%)'] > 0)).sum()}社")

    st.divider()

    ab_t1, ab_t2, ab_t3 = st.tabs([
        "🏆 高アルファランキング",
        "🔵 α vs β 散布図",
        "💎 低β・高α スクリーナー",
    ])

    # ── Tab1: 高アルファランキング ───────────────────────────────
    with ab_t1:
        st.markdown("#### 🏆 アルファランキング（市場超過リターン上位）")
        st.caption("αが高い = 日経平均の動きに関係なく独自に上昇している銘柄")

        top_alpha = df_ab.sort_values("アルファ(%)", ascending=False).head(30)
        bot_alpha = df_ab.sort_values("アルファ(%)", ascending=True).head(10)

        col_a1, col_a2 = st.columns([2, 1])
        with col_a1:
            st.markdown("**上位30銘柄（高アルファ）**")
            def _color_alpha(val):
                if isinstance(val, float):
                    if val > 10: return "color:#1a7f37;font-weight:bold;font-size:14px"
                    elif val > 0: return "color:#1a7f37;font-weight:bold"
                    elif val < 0: return "color:#d1242f"
                return ""
            st.dataframe(
                top_alpha[["企業名","業種","アルファ(%)","ベータ",
                            "年間平均リターン(%)","シャープレシオ"]]
                .style
                .format({"アルファ(%)":"{:+.2f}","ベータ":"{:.2f}",
                         "年間平均リターン(%)":"{:.2f}","シャープレシオ":"{:.2f}"})
                .map(_color_alpha, subset=["アルファ(%)"]),
                use_container_width=True, hide_index=True
            )
        with col_a2:
            st.markdown("**下位10銘柄（低アルファ・市場負け）**")
            st.dataframe(
                bot_alpha[["企業名","業種","アルファ(%)","ベータ"]]
                .style
                .format({"アルファ(%)":"{:+.2f}","ベータ":"{:.2f}"})
                .map(_color_alpha, subset=["アルファ(%)"]),
                use_container_width=True, hide_index=True
            )

        # アルファ棒グラフ
        fig_alpha, ax_alpha = plt.subplots(figsize=(14, 6))
        top20 = df_ab.sort_values("アルファ(%)", ascending=False).head(20)
        colors_a = ["#1a7f37" if v >= 0 else "#d1242f" for v in top20["アルファ(%)"]]
        ax_alpha.bar(top20["企業名"], top20["アルファ(%)"],
                     color=colors_a, alpha=0.85)
        ax_alpha.axhline(0, color="black", linewidth=0.8)
        ax_alpha.set_title("アルファ上位20銘柄（年間・市場超過リターン）",
                            fontsize=12, fontweight="bold")
        ax_alpha.set_ylabel("アルファ (%)")
        ax_alpha.tick_params(axis="x", rotation=45, labelsize=9)
        ax_alpha.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_alpha, clear_figure=True)

    # ── Tab2: α vs β 散布図 ─────────────────────────────────────
    with ab_t2:
        st.markdown("#### 🔵 アルファ vs ベータ 散布図（全銘柄）")
        st.caption(
            "**右上**（高β・高α）= 積極的成長株 | "
            "**左上**（低β・高α）= 理想的な優良株 | "
            "**右下**（高β・低α）= 市場連動だが割高 | "
            "**左下**（低β・低α）= 市場負け・ディフェンシブ"
        )

        sectors_ab = df_ab["業種"].unique()
        cmap_ab = plt.colormaps["tab20"].resampled(len(sectors_ab))
        sec_color_ab = {s: cmap_ab(i) for i, s in enumerate(sectors_ab)}

        fig_ab, ax_ab = plt.subplots(figsize=(14, 9))

        for sec in sectors_ab:
            sub = df_ab[df_ab["業種"] == sec]
            ax_ab.scatter(
                sub["ベータ"], sub["アルファ(%)"],
                label=sec, color=sec_color_ab[sec],
                s=70, alpha=0.8, zorder=3
            )
            # 注目銘柄にラベル
            for _, row in sub.iterrows():
                if row["アルファ(%)"] > df_ab["アルファ(%)"].quantile(0.85) or \
                   row["アルファ(%)"] < df_ab["アルファ(%)"].quantile(0.10):
                    ax_ab.annotate(
                        row["企業名"],
                        (row["ベータ"], row["アルファ(%)"]),
                        fontsize=7, alpha=0.9,
                        xytext=(4, 4), textcoords="offset points",
                    )

        # 軸線
        ax_ab.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=2)
        ax_ab.axvline(1, color="orange", linewidth=1.0,
                      linestyle="--", alpha=0.6, zorder=2, label="β=1（市場平均）")

        # 象限ラベル
        x_lim = ax_ab.get_xlim()
        y_lim = ax_ab.get_ylim()
        ax_ab.text(0.3, df_ab["アルファ(%)"].max() * 0.8,
                   "低β・高α\n💎 理想優良株",
                   color="#1a7f37", fontsize=10, fontweight="bold", ha="center",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", alpha=0.8))
        ax_ab.text(1.5, df_ab["アルファ(%)"].max() * 0.8,
                   "高β・高α\n🚀 積極成長株",
                   color="#1565c0", fontsize=10, fontweight="bold", ha="center",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#e3f2fd", alpha=0.8))
        ax_ab.text(0.3, df_ab["アルファ(%)"].min() * 0.8,
                   "低β・低α\n😴 市場負け",
                   color="#9e9e9e", fontsize=10, fontweight="bold", ha="center",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", alpha=0.8))
        ax_ab.text(1.5, df_ab["アルファ(%)"].min() * 0.8,
                   "高β・低α\n⚠️ 市場連動・割高",
                   color="#d1242f", fontsize=10, fontweight="bold", ha="center",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", alpha=0.8))

        ax_ab.set_xlabel("ベータ（β）― 市場連動性", fontsize=12)
        ax_ab.set_ylabel("アルファ（α）(%) ― 市場超過リターン", fontsize=12)
        ax_ab.set_title("アルファ vs ベータ 分析マップ", fontsize=13, fontweight="bold")
        ax_ab.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
                     fontsize=8, framealpha=0.9, ncol=1)
        ax_ab.grid(True, alpha=0.2, zorder=1)
        ax_ab.spines["top"].set_visible(False)
        ax_ab.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_ab, clear_figure=True)

    # ── Tab3: 低β・高α スクリーナー ────────────────────────────
    with ab_t3:
        st.markdown("#### 💎 低ベータ・高アルファ スクリーナー")
        st.caption("市場リスクを抑えながら超過リターンを稼いでいる銘柄を抽出")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            beta_max  = st.slider("最大β（低いほど市場影響小）",
                                  0.3, 2.0, 1.0, 0.1, key="ab_beta_max")
        with col_s2:
            alpha_min = st.slider("最小α（%）（高いほど超過収益大）",
                                  -10.0, 30.0, 0.0, 0.5, key="ab_alpha_min")
        with col_s3:
            sharpe_min = st.slider("最小シャープレシオ",
                                   0.0, 3.0, 0.5, 0.1, key="ab_sharpe_min")

        df_screen = df_ab[
            (df_ab["ベータ"] <= beta_max) &
            (df_ab["アルファ(%)"] >= alpha_min) &
            (df_ab["シャープレシオ"] >= sharpe_min)
        ].sort_values("アルファ(%)", ascending=False)

        st.markdown(f"**{len(df_screen)}銘柄が条件を満たしています**"
                    f"（β≤{beta_max} & α≥{alpha_min}% & SR≥{sharpe_min}）")

        if df_screen.empty:
            st.info("条件を緩めてみてください")
        else:
            # スコア計算（α/β比）
            df_screen = df_screen.copy()
            df_screen["α/β比"] = (
                df_screen["アルファ(%)"] / (df_screen["ベータ"].abs() + 0.01)
            ).round(2)

            disp_cols = ["企業名", "業種", "アルファ(%)", "ベータ",
                         "α/β比", "シャープレシオ", "年間平均リターン(%)", "年間リスク(%)"]

            def _color_score(val):
                if isinstance(val, float):
                    if val > 10: return "color:#1a7f37;font-weight:bold;font-size:14px"
                    elif val > 5: return "color:#1a7f37;font-weight:bold"
                    elif val > 0: return "color:#388e3c"
                return ""

            st.dataframe(
                df_screen[disp_cols].style
                .format({
                    "アルファ(%)": "{:+.2f}",
                    "ベータ": "{:.2f}",
                    "α/β比": "{:.2f}",
                    "シャープレシオ": "{:.2f}",
                    "年間平均リターン(%)": "{:.2f}",
                    "年間リスク(%)": "{:.2f}",
                })
                .map(_color_score, subset=["α/β比"]),
                use_container_width=True, hide_index=True
            )

            # CSVダウンロード
            csv_ab = df_screen[disp_cols].to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "⬇️ スクリーニング結果をCSVダウンロード",
                data=csv_ab,
                file_name=f"alpha_beta_screen_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="ab_dl"
            )

            # バブルチャート（サイズ=シャープレシオ）
            fig_sc, ax_sc = plt.subplots(figsize=(12, 7))
            sc = ax_sc.scatter(
                df_screen["ベータ"],
                df_screen["アルファ(%)"],
                s=df_screen["シャープレシオ"].clip(lower=0.1) * 200,
                c=df_screen["α/β比"],
                cmap="YlGn",
                alpha=0.8, edgecolors="gray", linewidth=0.5, zorder=3
            )
            for _, row in df_screen.head(20).iterrows():
                ax_sc.annotate(
                    row["企業名"],
                    (row["ベータ"], row["アルファ(%)"]),
                    fontsize=8,
                    xytext=(5, 5), textcoords="offset points",
                )
            plt.colorbar(sc, ax=ax_sc, label="α/β比")
            ax_sc.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax_sc.axvline(1, color="orange", linewidth=0.8,
                          linestyle="--", alpha=0.6)
            ax_sc.set_xlabel("ベータ（β）", fontsize=12)
            ax_sc.set_ylabel("アルファ（α）(%)", fontsize=12)
            ax_sc.set_title(
                "低β・高α スクリーニング結果\n（バブルサイズ = シャープレシオ、色 = α/β比）",
                fontsize=12, fontweight="bold"
            )
            ax_sc.grid(True, alpha=0.2)
            ax_sc.spines["top"].set_visible(False)
            ax_sc.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_sc, clear_figure=True)


# ─── Tab2: セクターローテーション ────────────────────────────────

st.markdown("""<div style="background:#2e7d32;color:white;padding:12px 22px;border-radius:8px;margin:28px 0 4px 0;font-size:18px;font-weight:bold;">
📈 B.&nbsp;テクニカル・需給分析 &nbsp;<span style="font-size:12px;font-weight:400;opacity:.88">価格パターン / 需給 / モメンタム / セクター / J-Quants / ニュース</span></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
st.header("🔄 セクターローテーション")
st.divider()
st.subheader("🔄 セクターローテーション分析")
st.caption("各業種に属する銘柄の平均リターンを集計し、資金が流入・流出しているセクターを可視化します。")

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 3])
with col_ctrl1:
    rotation_period = st.selectbox(
        "分析期間",
        options=[5, 10, 20, 60, 90],
        index=2,
        format_func=lambda x: {5: "1週間(5日)", 10: "2週間(10日)",
                                20: "1ヶ月(20日)", 60: "3ヶ月(60日)",
                                90: "約半年(90日)"}[x],
    )
with col_ctrl2:
    top_bottom_n = st.slider("上位・下位 表示セクター数", 3, 8, 5)
with col_ctrl3:
    run_rotation = True  # 自動実行

st.divider()

if run_rotation:
    with st.spinner(f"全銘柄の株価データを取得中（{len(ticker_name_map)}銘柄）..."):
        df_sector = get_sector_performance(ticker_name_map, period_days=rotation_period)

    if df_sector.empty:
        st.error("データの取得に失敗しました。しばらくしてから再試行してください。")
    else:
        top_sec    = df_sector.iloc[0]
        bottom_sec = df_sector.iloc[-1]
        rising     = (df_sector["平均リターン(%)"] > 0).sum()
        falling    = (df_sector["平均リターン(%)"] < 0).sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📈 最強セクター",  top_sec["業種"],    f"{top_sec['騰落率(%)']:+.2f}%")
        k2.metric("📉 最弱セクター",  bottom_sec["業種"], f"{bottom_sec['騰落率(%)']:+.2f}%")
        k3.metric("🟢 上昇セクター数", f"{rising} 業種")
        k4.metric("🔴 下落セクター数", f"{falling} 業種")

        st.divider()

        period_label = {5: "1週間", 10: "2週間", 20: "1ヶ月", 60: "3ヶ月", 90: "約半年"}[rotation_period]
        fig_bar = plot_sector_bar(
            df_sector,
            title=f"セクター別平均リターン（{period_label}） 買われ / 売られ",
        )
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        st.divider()
        st.subheader("📈 買われセクター vs 📉 売られセクター の値動き比較")
        top_sectors    = df_sector.head(top_bottom_n)["業種"].tolist()
        bottom_sectors = df_sector.tail(top_bottom_n)["業種"].tolist()

        with st.spinner("時系列データ取得中..."):
            df_ts = get_sector_timeseries(ticker_name_map, days=max(rotation_period + 10, 30))

        if not df_ts.empty:
            fig_ts = plot_sector_timeseries(df_ts, top_sectors, bottom_sectors)
            st.pyplot(fig_ts)
            plt.close(fig_ts)

        st.divider()
        st.subheader("🌡️ セクター別ヒートマップ（期間比較）")
        with st.spinner("複数期間データを取得中..."):
            df_1w = get_sector_performance(ticker_name_map, period_days=5)
            df_1m = get_sector_performance(ticker_name_map, period_days=20)
            df_3m = get_sector_performance(ticker_name_map, period_days=60)

        df_heat_base = df_1m[["業種"]].copy()
        df_heat_base = df_heat_base.merge(
            df_1w[["業種", "平均リターン(%)"]].rename(columns={"平均リターン(%)": "1週間"}), on="業種", how="left"
        ).merge(
            df_1m[["業種", "平均リターン(%)"]].rename(columns={"平均リターン(%)": "1ヶ月"}), on="業種", how="left"
        ).merge(
            df_3m[["業種", "平均リターン(%)"]].rename(columns={"平均リターン(%)": "3ヶ月"}), on="業種", how="left"
        )

        fig_heat = plot_sector_heatmap(df_heat_base)
        st.pyplot(fig_heat)
        plt.close(fig_heat)

        st.divider()
        st.subheader("📋 セクター別詳細データ")
        df_display = df_sector[["業種", "平均リターン(%)", "中央値リターン(%)",
                                 "銘柄数", "上昇銘柄数", "下落銘柄数", "上昇率(%)"]].copy()

        def color_return(val):
            if isinstance(val, float):
                if val > 2:    return "background-color: rgba(56,142,60,0.45); color: white; font-weight:bold"
                elif val > 0:  return "color: #388e3c; font-weight:bold"
                elif val < -2: return "background-color: rgba(211,47,47,0.45); color: white; font-weight:bold"
                elif val < 0:  return "color: #d32f2f; font-weight:bold"
            return ""

        styled = df_display.style.format({
            "平均リターン(%)": "{:+.2f}",
            "中央値リターン(%)": "{:+.2f}",
            "上昇率(%)": "{:.1f}",
        }).map(color_return, subset=["平均リターン(%)", "中央値リターン(%)"])
        st.dataframe(styled, use_container_width=True, height=500)

        st.divider()
        st.subheader("🤖 AIによるセクターローテーション解説")
        top5_str    = df_sector.head(5)[["業種", "騰落率(%)"]].to_string(index=False)
        bottom5_str = df_sector.tail(5)[["業種", "騰落率(%)"]].to_string(index=False)
        prompt_rotation = (
            "あなたは日本株の機関投資家向けストラテジストです。\n"
            f"以下は直近{period_label}のJPX上場主要銘柄のセクター別平均リターンです。\n\n"
            f"【買われているセクター上位5】\n{top5_str}\n\n"
            f"【売られているセクター下位5】\n{bottom5_str}\n\n"
            "以下の観点で400文字以内で分析してください:\n"
            "1. 現在のセクターローテーションの特徴\n"
            "2. 買われているセクターの背景・理由\n"
            "3. 売られているセクターの背景・理由\n"
            "4. 投資家へのアドバイス\n"
        )
        with st.spinner("AI分析中..."):
            try:
                comment, ai_name = generate_ai_comment(prompt_rotation)
                st.info(f"{comment}\n\n_AI: {ai_name}_")
            except Exception as e:
                st.warning(f"AI APIエラー: {e}")

else:
    st.info(
        "「▶ セクターローテーション分析を実行」ボタンを押すと分析が始まります。\n\n"
        "表示されるグラフ:\n"
        "- セクター別平均リターン棒グラフ（買われ・売られ色分け）\n"
        "- 上位・下位セクターの累積リターン時系列グラフ\n"
        "- 1週間 / 1ヶ月 / 3ヶ月 ヒートマップ（期間比較）\n"
        "- セクター別詳細テーブル（上昇銘柄数・上昇率など）\n"
        "- AIによるローテーション解説とアドバイス"
    )
    st.caption("全銘柄データ取得のため、初回実行には数十秒かかる場合があります。結果は30分キャッシュされます。")


# ─── Tab3: 需給スクリーナー ──────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
st.header("🔥 需給スクリーナー")
st.divider()
st.subheader("🔥 需給スクリーナー（出来高ベース）")

col_v1, col_v2, col_v3 = st.columns([2, 2, 3])
with col_v1:
    surge_ratio = st.slider("出来高急増の閾値（倍）", 1.5, 5.0, 2.0, 0.5)
with col_v2:
    pv_days = st.selectbox("Price x Volume 期間", [10, 20, 60], index=1,
                            format_func=lambda x: f"{x}日")
with col_v3:
    run_volume = True  # 自動実行

st.divider()

if run_volume:
    # ── 需給3データを並列取得 ──────────────────────────────────────
    import concurrent.futures as _cf_vol
    with st.spinner("📊 出来高・VWAP・PriceVolume データを並列取得中..."):
        with _cf_vol.ThreadPoolExecutor(max_workers=3) as _ex_vol:
            _f_surge = _ex_vol.submit(get_volume_surge, ticker_name_map, surge_ratio)
            _f_vwap  = _ex_vol.submit(get_vwap_deviation, ticker_name_map)
            _f_pv    = _ex_vol.submit(get_price_volume_scatter, ticker_name_map, pv_days)
            df_surge = _f_surge.result()
            df_vwap  = _f_vwap.result()
            df_pv    = _f_pv.result()

    st.subheader(f"📊 出来高急増銘柄（過去5日平均が20日平均の{surge_ratio}倍以上）")
    if df_surge.empty:
        st.info(f"現在、出来高が{surge_ratio}倍以上の銘柄は検出されませんでした。")
    else:
        st.success(f"🔺 {len(df_surge)} 銘柄検出")
        def color_surge(val):
            if isinstance(val, float):
                if val >= 3:  return "background-color: #d32f2f; color: white; font-weight:bold"
                elif val >= 2: return "background-color: #f57c00; color: white; font-weight:bold"
            return ""
        styled_surge = df_surge.style.format({
            "出来高倍率": "{:.2f}x",
            "株価変化率(5日%)": "{:+.2f}",
        }).map(color_surge, subset=["出来高倍率"])
        st.dataframe(styled_surge, use_container_width=True)

        top5 = df_surge.head(5)[["企業名", "業種", "出来高倍率", "株価変化率(5日%)"]].to_string(index=False)
        prompt_surge = (
            "以下は直近5日間で出来高が急増した日本株銘柄上位5社です。\n\n"
            f"{top5}\n\n"
            "投資家向けに300文字以内で分析してください:\n"
            "1. 機関投資家・仕手の動きと考えられるか\n"
            "2. 業種・テーマ的な特徴\n"
            "3. 注意点・リスク\n"
        )
        with st.spinner("AI分析中..."):
            try:
                comment, ai_name = generate_ai_comment(prompt_surge)
                st.info(f"🤖 **AI解説（{ai_name}）**\n\n{comment}")
            except Exception as e:
                st.warning(f"AI APIエラー: {e}")

    st.divider()

    st.subheader("📏 VWAP乖離率ランキング（割高・割安スクリーニング）")
    if not df_vwap.empty:
        col_up, col_down = st.columns(2)
        with col_up:
            st.markdown("#### 🔴 割高（VWAP上方乖離 上位10）")
            df_over = df_vwap[df_vwap["VWAP乖離率(%)"] > 0].head(10)
            st.dataframe(df_over.style.format({"VWAP乖離率(%)": "{:+.2f}"}),
                         use_container_width=True)
        with col_down:
            st.markdown("#### 🟢 割安（VWAP下方乖離 下位10）")
            df_under = df_vwap[df_vwap["VWAP乖離率(%)"] < 0].tail(10).sort_values("VWAP乖離率(%)")
            st.dataframe(df_under.style.format({"VWAP乖離率(%)": "{:+.2f}"}),
                         use_container_width=True)

    st.divider()

    st.subheader(f"🗺️ Price x Volume マップ（直近{pv_days}日）")
    if df_pv.empty:
        st.info("散布図データを取得できませんでした。しばらく待って再読み込みしてください。")
    else:
        plot_pv_scatter(df_pv)

        q1 = df_pv[(df_pv["株価騰落率(%)"] > 0) & (df_pv["出来高変化率(%)"] > 0)]
        q1_top = q1.nlargest(5, "株価騰落率(%)")[["企業名", "業種", "株価騰落率(%)", "出来高変化率(%)"]].to_string(index=False)
        prompt_pv = (
            "株価上昇かつ出来高増加の上位銘柄:\n\n"
            f"{q1_top}\n\n"
            "投資家向けに200文字以内でコメントしてください。\n"
        )
        with st.spinner("AI分析中..."):
            try:
                comment, ai_name = generate_ai_comment(prompt_pv)
                st.info(f"🤖 **本命上昇銘柄 AI解説（{ai_name}）**\n\n{comment}")
            except Exception as e:
                st.warning(f"AI APIエラー: {e}")
else:
    st.info(
        "「▶ 需給分析を実行」ボタンを押してください。\n\n"
        "- 📊 出来高急増スクリーナー\n"
        "- 📏 VWAP乖離ランキング\n"
        "- 🗺️ Price x Volume マップ"
    )


# ─── Tab4: 価格パターン ──────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
st.header("📈 価格パターン")
st.divider()
st.subheader("📈 価格パターン分析")

col_p1, col_p2 = st.columns([3, 2])
with col_p1:
    run_price = True  # 自動実行
with col_p2:
    cross_lookback = st.slider("クロスシグナル 直近何日以内を検出？", 3, 20, 10)

st.divider()

if run_price:
    # ── 価格パターン3データを並列取得 ──────────────────────────────
    import concurrent.futures as _cf_pp
    with st.spinner("📈 52週・移動平均・クロスシグナルを並列取得中..."):
        with _cf_pp.ThreadPoolExecutor(max_workers=3) as _ex_pp:
            _f_52  = _ex_pp.submit(get_52week_highlow, ticker_name_map)
            _f_ma  = _ex_pp.submit(get_ma_deviation, ticker_name_map)
            _f_crs = _ex_pp.submit(get_cross_signals, ticker_name_map, cross_lookback)
            df_52    = _f_52.result()
            df_ma    = _f_ma.result()
            df_cross = _f_crs.result()

    st.subheader("🏔️ 52週高値・安値ダッシュボード")
    if not df_52.empty:
        new_highs = df_52[df_52["新高値"] != ""]
        new_lows  = df_52[df_52["新安値"] != ""]

        col_nh, col_nl = st.columns(2)
        with col_nh:
            st.metric("🔺 新高値更新銘柄", f"{len(new_highs)} 銘柄")
            if not new_highs.empty:
                st.dataframe(new_highs[["企業名", "業種", "現在値", "52週高値", "高値からの乖離(%)"]],
                             use_container_width=True)
        with col_nl:
            st.metric("🔻 新安値更新銘柄", f"{len(new_lows)} 銘柄")
            if not new_lows.empty:
                st.dataframe(new_lows[["企業名", "業種", "現在値", "52週安値", "安値からの乖離(%)"]],
                             use_container_width=True)

        hl_index = len(new_highs) / max(len(new_highs) + len(new_lows), 1) * 100
        st.metric("📊 ハイローインデックス", f"{hl_index:.1f}%",
                  help="新高値/(新高値+新安値)x100。50%超=強気市場の目安")
        if hl_index >= 70:
            st.success("📈 強気市場シグナル（新高値銘柄が多数）")
        elif hl_index <= 30:
            st.error("📉 弱気市場シグナル（新安値銘柄が多数）")
        else:
            st.info("⚖️ 中立（方向感なし）")

    st.divider()

    st.subheader("📐 移動平均線乖離率ランキング")
    if not df_ma.empty:
        col_ma1, col_ma2 = st.columns(2)
        with col_ma1:
            st.markdown("#### 🔴 25日MA 上方乖離 上位10（買われすぎ）")
            st.dataframe(
                df_ma.head(10)[["企業名", "業種", "現在値", "25日MA乖離(%)", "75日MA乖離(%)"]].style.format({
                    "25日MA乖離(%)": "{:+.2f}", "75日MA乖離(%)": "{:+.2f}"
                }), use_container_width=True
            )
        with col_ma2:
            st.markdown("#### 🟢 25日MA 下方乖離 下位10（売られすぎ）")
            st.dataframe(
                df_ma.tail(10)[["企業名", "業種", "現在値", "25日MA乖離(%)", "75日MA乖離(%)"]].style.format({
                    "25日MA乖離(%)": "{:+.2f}", "75日MA乖離(%)": "{:+.2f}"
                }), use_container_width=True
            )

    st.divider()

    st.subheader(f"🔔 ゴールデンクロス / デッドクロス（直近{cross_lookback}日以内）")

    if df_cross.empty:
        st.info(f"直近{cross_lookback}日以内にクロスシグナルは検出されませんでした。")
    else:
        gc = df_cross[df_cross["シグナル"].str.contains("ゴールデン")]
        dc = df_cross[df_cross["シグナル"].str.contains("デッド")]
        col_gc, col_dc = st.columns(2)
        with col_gc:
            st.markdown(f"#### 🟡 ゴールデンクロス — {len(gc)} 銘柄")
            if not gc.empty:
                st.dataframe(gc[["企業名", "業種", "発生日", "現在値"]], use_container_width=True)
        with col_dc:
            st.markdown(f"#### 💀 デッドクロス — {len(dc)} 銘柄")
            if not dc.empty:
                st.dataframe(dc[["企業名", "業種", "発生日", "現在値"]], use_container_width=True)

        cross_str = df_cross.head(8)[["企業名", "業種", "シグナル", "発生日"]].to_string(index=False)
        prompt_cross = (
            "直近のゴールデンクロス・デッドクロス発生銘柄:\n\n"
            f"{cross_str}\n\n"
            "投資家向けに200文字以内で注目ポイントをコメントしてください。\n"
        )
        with st.spinner("AI分析中..."):
            try:
                comment, ai_name = generate_ai_comment(prompt_cross)
                st.info(f"🤖 **AI解説（{ai_name}）**\n\n{comment}")
            except Exception as e:
                st.warning(f"AI APIエラー: {e}")
else:
    st.info(
        "「▶ 価格パターン分析を実行」ボタンを押してください。\n\n"
        "- 🏔️ 52週高値・安値ダッシュボード + ハイローインデックス\n"
        "- 📐 25日・75日・200日MA乖離率ランキング\n"
        "- 🔔 ゴールデンクロス/デッドクロス 直近発生銘柄"
    )


# ─── Tab5: モメンタム・相関分析 ──────────────────────────────────

# ─────────────────────────────────────────────────────────────────
st.header("💡 モメンタム・相関分析")
st.divider()
st.subheader("💡 モメンタム・相関分析")

col_u1, col_u2 = st.columns([3, 2])
with col_u1:
    run_unique = True  # 自動実行
with col_u2:
    corr_window = st.slider("相関崩れ 直近ウィンドウ（日）", 10, 30, 20)

st.divider()

if run_unique:
    st.subheader("🚀 週次モメンタムスコアランキング")
    with st.spinner("モメンタムスコア計算中..."):
        df_mom = get_momentum_score(ticker_name_map)

    if not df_mom.empty:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### 📈 上位20銘柄（買いモメンタム）")
            st.dataframe(
                df_mom.head(20).style.format({
                    "モメンタムスコア": "{:.3f}",
                    "株価騰落率(%)": "{:+.2f}",
                    "出来高変化率(%)": "{:+.2f}",
                }), use_container_width=True
            )
        with col_m2:
            st.markdown("#### 📉 下位20銘柄（売りモメンタム）")
            st.dataframe(
                df_mom.tail(20).sort_values("モメンタムスコア").style.format({
                    "モメンタムスコア": "{:.3f}",
                    "株価騰落率(%)": "{:+.2f}",
                    "出来高変化率(%)": "{:+.2f}",
                }), use_container_width=True
            )

        top10 = df_mom.head(10)[["企業名", "業種", "モメンタムスコア", "株価騰落率(%)", "出来高変化率(%)"]].to_string(index=False)
        bot10 = df_mom.tail(10)[["企業名", "業種", "モメンタムスコア", "株価騰落率(%)", "出来高変化率(%)"]].to_string(index=False)
        prompt_mom = (
            "あなたは日本株ストラテジストです。\n"
            "以下は直近1ヶ月のモメンタムスコアランキングです。\n\n"
            f"【高モメンタム上位10銘柄】\n{top10}\n\n"
            f"【低モメンタム下位10銘柄】\n{bot10}\n\n"
            "週次レポートとして400文字以内で分析してください:\n"
            "1. 今週のモメンタム相場の特徴\n"
            "2. 注目銘柄とその理由\n"
            "3. 逆張りの観点からの注意点\n"
        )
        with st.spinner("AI週次レポート生成中..."):
            try:
                comment, ai_name = generate_ai_comment(prompt_mom)
                st.subheader(f"🤖 AI週次モメンタムレポート（{ai_name}）")
                st.info(comment)
            except Exception as e:
                st.warning(f"AI APIエラー: {e}")

    st.divider()

    st.subheader("📅 曜日別平均リターン（市場の癖）")
    with st.spinner("曜日パターン分析中..."):
        df_dow = get_dow_of_week_pattern(ticker_name_map)

    if not df_dow.empty:
        fig_dow = plot_dow_heatmap(df_dow)
        st.pyplot(fig_dow)
        plt.close(fig_dow)

        stack = df_dow.stack().reset_index()
        stack.columns = ["業種", "曜日", "平均リターン(%)"]
        best  = stack.nlargest(3, "平均リターン(%)")
        worst = stack.nsmallest(3, "平均リターン(%)")
        col_b, col_w = st.columns(2)
        with col_b:
            st.markdown("#### 🟢 最もリターンが高い 曜日×セクター")
            st.dataframe(best.style.format({"平均リターン(%)": "{:+.4f}"}), use_container_width=True)
        with col_w:
            st.markdown("#### 🔴 最もリターンが低い 曜日×セクター")
            st.dataframe(worst.style.format({"平均リターン(%)": "{:+.4f}"}), use_container_width=True)

    st.divider()

    st.subheader("🔍 日経平均との相関崩れ検知（個別材料の先行シグナル）")
    with st.spinner("相関分析中..."):
        df_corr = get_correlation_divergence(ticker_name_map, corr_window=corr_window)

    if not df_corr.empty:
        st.caption("相関乖離度が高い = 最近、日経と独自の動きをしている銘柄（個別材料の可能性）")
        col_div1, col_div2 = st.columns(2)
        with col_div1:
            st.markdown("#### 🟡 相関崩れ上位15（独自上昇の可能性）")
            rising_div = df_corr[df_corr["直近5日株価変化(%)"] > 0].head(15)
            st.dataframe(rising_div.style.format({
                "長期相関": "{:.3f}", "直近相関": "{:.3f}",
                "相関乖離度": "{:.3f}", "直近5日株価変化(%)": "{:+.2f}"
            }), use_container_width=True)
        with col_div2:
            st.markdown("#### 🔴 相関崩れ上位15（独自下落・要注意）")
            falling_div = df_corr[df_corr["直近5日株価変化(%)"] < 0].head(15)
            st.dataframe(falling_div.style.format({
                "長期相関": "{:.3f}", "直近相関": "{:.3f}",
                "相関乖離度": "{:.3f}", "直近5日株価変化(%)": "{:+.2f}"
            }), use_container_width=True)

        top_div = df_corr.head(5)[["企業名", "業種", "相関乖離度", "直近5日株価変化(%)"]].to_string(index=False)
        prompt_corr = (
            "以下は日経平均との相関が最近崩れている日本株銘柄上位5社です。\n\n"
            f"{top_div}\n\n"
            "投資家向けに200文字以内でコメントしてください:\n"
            "1. 考えられる個別材料の種類\n"
            "2. 投資機会またはリスク\n"
        )
        with st.spinner("AI分析中..."):
            try:
                comment, ai_name = generate_ai_comment(prompt_corr)
                st.info(f"🤖 **AI解説（{ai_name}）**\n\n{comment}")
            except Exception as e:
                st.warning(f"AI APIエラー: {e}")
else:
    st.info(
        "「▶ モメンタム・相関分析を実行」ボタンを押してください。\n\n"
        "- 🚀 週次モメンタムスコアランキング + AI自動レポート\n"
        "- 📅 曜日別平均リターンヒートマップ（市場の癖）\n"
        "- 🔍 日経平均との相関崩れ検知（個別材料の先行シグナル）"
    )


# ── TDnet / PDF ヘルパー（銘柄別ニュースセクションより前で定義）──────
EDINET_API_BASE = "https://disclosure.edinet-api.go.jp/api/v2"
TDNET_BASE      = "https://www.release.tdnet.info/inbs"

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tdnet_week(code_map: dict, days: int = 7, code4_filter: str = None) -> list:
    import re as _re, time as _time
    from bs4 import BeautifulSoup as _BS
    from datetime import timedelta as _td
    code4_map = {}
    for ticker, (name, sector) in code_map.items():
        c4 = ticker.replace(".T", "").zfill(4)
        code4_map[c4] = (name, sector)
    results = []
    today = datetime.today()
    checked, d = 0, 0
    # days * 2 + 10 でカレンダー日を十分確保（週末・祝日対応）
    while checked < days and d < days * 2 + 10:
        target = today - _td(days=d); d += 1
        if target.weekday() >= 5:
            continue
        checked += 1
        date_str = target.strftime("%Y%m%d")
        for page in range(1, 10):
            url = f"{TDNET_BASE}/I_list_{page:03d}_{date_str}.html"
            try:
                r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code != 200:
                    break
                r.encoding = r.apparent_encoding or "utf-8"
                soup = _BS(r.text, "html.parser")
                rows = soup.find_all("tr")
                found_any = False
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) < 4:
                        continue
                    texts = [c.get_text(strip=True) for c in cells]
                    code = next((t for t in texts if _re.fullmatch(r'\d{4}', t)), None)
                    if not code or code not in code4_map:
                        continue
                    if code4_filter and code != code4_filter:
                        continue
                    found_any = True
                    pdf_url = None
                    title = ""
                    for a in row.find_all("a", href=True):
                        h = a["href"]
                        m = _re.search(r'(\d{14,18})\.pdf', h)
                        if m:
                            pdf_url = f"https://www.release.tdnet.info/inbs/{m.group(1)}.pdf"
                        elif not title and not h.endswith(".pdf"):
                            t = a.get_text(strip=True)
                            if t and len(t) > 2:
                                title = t
                    if not title:
                        ci = texts.index(code)
                        title = texts[ci + 2] if ci + 2 < len(texts) else texts[-1]
                    name, sector = code4_map[code]
                    time_str = next(
                        (t for t in texts if _re.fullmatch(r'\d{1,2}:\d{2}', t)), ""
                    )
                    results.append({
                        "日付": target.strftime("%Y-%m-%d"),
                        "時刻": time_str,
                        "コード": code,
                        "企業名": name,
                        "業種": sector,
                        "タイトル": title,
                        "PDF_URL": pdf_url,
                    })
                if not found_any and page > 1:
                    break
                # 長期取得時のレート制限回避
                if days > 14:
                    _time.sleep(0.15)
            except Exception:
                break
    return results

@st.cache_data(ttl=7200, show_spinner=False)
def fetch_tdnet_pdf_text(pdf_url: str) -> str:
    if not pdf_url:
        return "[PDF URLが見つかりませんでした]"
    try:
        import pdfplumber, io
        r = requests.get(pdf_url, timeout=35, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return f"[HTTP {r.status_code}: PDF取得失敗]"
        if len(r.content) < 500:
            return "[レスポンスが短すぎます]"
        with pdfplumber.open(io.BytesIO(r.content)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages[:5])
        return text if text.strip() else "[テキスト抽出失敗（画像PDFの可能性）]"
    except Exception as e:
        return f"[取得エラー: {e}]"

# ─── Tab6: 銘柄別ニュース ─────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
st.header("📰 銘柄別ニュース")
st.divider()
st.subheader("📰 銘柄別ニュース・適時開示")

ticker_options = {f"{name}（{t}）": t for t, (name, _) in ticker_name_map.items()}
selected_label = st.selectbox(
    "銘柄を選択", list(ticker_options.keys()),
    index=list(ticker_options.keys()).index("トヨタ（7203.T）")
    if "トヨタ（7203.T）" in ticker_options else 0
)
selected_ticker = ticker_options[selected_label]
selected_name   = ticker_name_map[selected_ticker][0]

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    run_news = True  # 自動実行
with col_btn2:
    run_ai   = st.checkbox("🤖 AIによる要約・センチメント分析も行う", value=True)

if run_news:
    with st.spinner(f"{selected_name} のニュースを全ソースから取得中..."):
        all_news = fetch_all_news(selected_ticker, news_max_per_source)

    filtered = [n for n in all_news if n["source"] in show_news_sources] if show_news_sources else all_news

    if not filtered:
        st.warning("ニュースが取得できませんでした（ソース設定を確認してください）")
    else:
        source_colors = {
            "Yahoo!Finance JP":  "🟦",
            "株探(Kabutan)":     "🟩",
            "みんかぶ":          "🟨",
            "TDnet（適時開示）": "🟥",
            "日経新聞":          "⬛",
            "Reuters JP":        "🟫",
        }

        from collections import Counter
        src_counts = Counter(n["source"] for n in filtered)
        cols_stat  = st.columns(len(src_counts))
        for i, (src, cnt) in enumerate(src_counts.items()):
            icon = source_colors.get(src, "⚪")
            cols_stat[i].metric(f"{icon} {src}", f"{cnt}件")

        st.divider()

        for item in filtered:
            icon = source_colors.get(item["source"], "⚪")
            with st.expander(f"{icon} [{item['source']}] {item['title'][:60]}{'...' if len(item['title'])>60 else ''}"):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{item['title']}**")
                    if item.get("summary"):
                        st.caption(item["summary"])
                with c2:
                    if item.get("date"):
                        st.caption(f"📅 {item['date']}")
                    if item.get("link"):
                        st.markdown(f"[🔗 記事を開く]({item['link']})")

        if run_ai:
            st.divider()
            st.subheader("🤖 AI ニュース分析（センチメント）")
            with st.spinner("AI分析中..."):
                ai_result = ai_news_summary(filtered, selected_name, selected_ticker)
            st.info(ai_result)

    # ── TDnet 適時開示（直近3営業日・選択銘柄・自動表示）───────────
    st.divider()
    _sel_code4 = selected_ticker.replace(".T", "").zfill(4)

    _tdcol_h, _tdcol_r = st.columns([3, 1])
    with _tdcol_h:
        st.subheader("📋 TDnet 適時開示")
    with _tdcol_r:
        _tdnet_days_n = st.radio(
            "期間", [3, 7, 14, 30, 60], index=4,
            horizontal=True, key="tdnet_news_days",
            help="30日以上は取得に時間がかかります（約1〜3分）"
        )

    with st.spinner(f"TDnetから {selected_name} の開示を取得中..."):
        _tdnet_news_list = fetch_tdnet_week(
            ticker_name_map, days=_tdnet_days_n, code4_filter=_sel_code4
        )

    if not _tdnet_news_list:
        st.info(
            f"直近{_tdnet_days_n}営業日に **{selected_name}** の適時開示はありません。"
            "（土日・祝日は開示なし。期間を延ばすか平日にご確認ください）"
        )
    else:
        _df_tn = pd.DataFrame(_tdnet_news_list).sort_values("日付", ascending=False)

        # ── AI タイトル要約（自動）
        _titles_str = "\n".join(
            f"・{r['日付']} {r.get('時刻','')}  {r['タイトル']}"
            for _, r in _df_tn.iterrows()
        )
        _prompt_titles = f"""
{selected_name}（証券コード {_sel_code4}）の直近{_tdnet_days_n}営業日の適時開示タイトル一覧:

{_titles_str}

以下の観点で**200字以内**でまとめてください:
1. 📌 主な開示内容のまとめ
2. 📈 業績・財務・経営への影響
3. 🔮 投資家として注目すべき点
"""
        with st.spinner("AIが開示内容を分析中..."):
            try:
                _ai_titles_sum, _ai_titles_nm = generate_ai_comment(_prompt_titles)
                st.info(f"🤖 **AI開示要約（{_ai_titles_nm}）**\n\n{_ai_titles_sum}")
            except Exception:
                pass

        st.caption(f"✅ {selected_name}：{len(_df_tn)}件（直近{_tdnet_days_n}営業日）")

        # ── 開示一覧（カード形式）
        for _i, _row in _df_tn.iterrows():
            _has_pdf = bool(_row.get("PDF_URL"))
            _col_a, _col_b, _col_c = st.columns([1.2, 4, 1.2])
            with _col_a:
                st.markdown(
                    f"<div style='font-size:12px;color:#666;padding-top:6px'>"
                    f"📅 {_row['日付']}<br>"
                    f"{'🕐 ' + _row.get('時刻','') if _row.get('時刻') else ''}</div>",
                    unsafe_allow_html=True,
                )
            with _col_b:
                st.markdown(
                    f"<div style='padding:6px 0;font-size:14px;font-weight:600'>"
                    f"{'📄' if _has_pdf else '📝'} {_row['タイトル']}</div>",
                    unsafe_allow_html=True,
                )
            with _col_c:
                if _has_pdf:
                    if st.button("🤖 PDF要約", key=f"tn_sum_{_i}", use_container_width=True):
                        with st.spinner("PDF取得・AI要約中（〜30秒）..."):
                            _pdf_txt = fetch_tdnet_pdf_text(_row["PDF_URL"])
                        if _pdf_txt.startswith("["):
                            st.error(f"取得失敗: {_pdf_txt}")
                        else:
                            _prompt_tn = f"""
以下は {selected_name}（{_sel_code4}）の適時開示「{_row['タイトル']}」のPDFテキストです。

{_pdf_txt[:4000]}

**300字以内**で以下の観点から要約:
1. 📌 開示の概要（何を発表したか）
2. 📈 業績・財務への影響（数値があれば具体的に）
3. 🔮 投資家へのポイント・今後の注目点
"""
                            _sum, _ai_nm = generate_ai_comment(_prompt_tn)
                            st.success(f"**AI要約（{_ai_nm}）**\n\n{_sum}")
            st.divider()


# ─── Tab7: 市場全体ニュース ──────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
st.header("🌐 市場全体ニュース")
st.divider()
st.subheader("🌐 市場全体ニュース（日経・Reuters）")

if True:  # 自動実行
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

    all_market = nikkei_news + reuters_news
    if all_market and st.checkbox("🤖 市場全体のAI要約を表示", value=True):
        headlines = "\n".join(f"[{n['source']}] {n['title']}" for n in all_market[:12])
        prompt = (
            "以下は本日の日本株マーケット関連ニュースです。\n\n"
            f"{headlines}\n\n"
            "投資家向けに300文字以内でまとめてください:\n"
            "1. 本日の市場全体のセンチメント\n"
            "2. 注目テーマ・セクター\n"
            "3. 今後の注意点\n"
        )
        with st.spinner("AI要約中..."):
            try:
                comment, ai_name = generate_ai_comment(prompt)
                st.subheader(f"🤖 市場全体AI要約（{ai_name}）")
                st.info(comment)
            except Exception as e:
                st.warning(f"AI APIエラー: {e}")


# ================================================================
# 📋 適時開示・EDINET AI分析 ヘルパー関数
# ================================================================
# fetch_tdnet_week / fetch_tdnet_pdf_text は銘柄別ニュースセクション直前で定義済み

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_edinet_docs(days: int = 7, doc_type_filter: str = "") -> list:
    """EDINET APIから過去N営業日の書類一覧を取得"""
    from datetime import timedelta as _td
    results = []
    today = datetime.today()
    for d in range(days * 2 + 10):
        target = today - _td(days=d)
        if target.weekday() >= 5:
            continue
        if sum(1 for r in results if r["日付"] == target.strftime("%Y-%m-%d")) > 0:
            continue
        if len({r["日付"] for r in results}) >= days:
            break
        date_str = target.strftime("%Y-%m-%d")
        try:
            r = requests.get(
                f"{EDINET_API_BASE}/documents.json",
                params={"date": date_str, "type": 2},
                timeout=12,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if r.status_code != 200:
                continue
            for doc in r.json().get("results", []):
                desc = doc.get("docDescription", "")
                if doc_type_filter and doc_type_filter not in desc:
                    continue
                results.append({
                    "日付": date_str,
                    "docID": doc.get("docID", ""),
                    "企業名": doc.get("filerName", ""),
                    "書類種別": desc,
                    "証券コード": doc.get("secCode", "")[:4],
                })
        except Exception:
            continue
    return results


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_edinet_pdf_text(doc_id: str) -> str:
    """EDINETから決算短信PDFを取得しテキスト抽出（先頭4ページ）"""
    try:
        import pdfplumber, io
        r = requests.get(
            f"{EDINET_API_BASE}/documents/{doc_id}",
            params={"type": 4},
            timeout=40,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code != 200:
            return f"[HTTP {r.status_code}]"
        ct = r.headers.get("Content-Type", "")
        if "pdf" not in ct.lower() and len(r.content) < 1000:
            return "[PDFが返されませんでした]"
        with pdfplumber.open(io.BytesIO(r.content)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages[:4])
        return text or "[テキスト抽出失敗]"
    except Exception as e:
        return f"[取得エラー: {e}]"


# ─────────────────────────────────────────────────────────────────
st.header("📋 適時開示・EDINET AI分析")
st.divider()
st.caption(
    "TDnet（東証 適時開示）と EDINET（金融庁）の公式データを活用。"
    "**決算短信のAI要約**・**機関投資家の大量保有動向**・**成長シグナルランキング**を提供します。"
)

edinet_t1, edinet_t2, edinet_t3 = st.tabs([
    "📊 週次 成長シグナルランキング",
    "📄 適時開示 PDF → AI要約",
    "🏦 大量保有報告書 監視",
])

# ── Tab1: 週次 成長シグナルランキング ─────────────────────────────
with edinet_t1:
    st.markdown("#### 📊 週次 適時開示 成長シグナルランキング（Top5）")
    st.caption("過去1週間の適時開示タイトルをAIが分析し、成長シグナルが最も強い企業Top5を表示します。")
    col_td1, col_td2 = st.columns([2, 1])
    with col_td1:
        tdnet_days = st.slider("取得期間（営業日）", 3, 60, 60, key="tdnet_days_sl",
                               help="30日以上は取得に1〜3分かかります")
    with col_td2:
        run_tdnet = st.button("▶ ランキング分析を実行", type="primary", key="run_tdnet_rank")

    if run_tdnet:
        with st.spinner("TDnetから適時開示を取得中（1〜2分かかります）..."):
            tdnet_list = fetch_tdnet_week(ticker_name_map, days=tdnet_days)

        if not tdnet_list:
            st.warning("⚠️ 適時開示データを取得できませんでした。TDnetのHTML構造が変わった可能性があります。")
        else:
            df_td = pd.DataFrame(tdnet_list)
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("取得件数", f"{len(df_td)}件")
            col_m2.metric("開示企業数", f"{df_td['企業名'].nunique()}社")
            col_m3.metric("取得期間", f"{tdnet_days}営業日")

            # 企業別タイトル集約
            grp = df_td.groupby("企業名")["タイトル"].apply(lambda x: " / ".join(x)).reset_index()
            disclosure_text = "\n".join(
                f"【{r['企業名']}】{r['タイトル']}" for _, r in grp.iterrows()
            )

            _prompt_tdnet = f"""
以下は日本株225銘柄の過去{tdnet_days}営業日の適時開示タイトル一覧です。

{disclosure_text[:4500]}

成長シグナルが最も強い企業 上位5社を選んでください。
選定基準：業績上方修正・増配・自社株買い・新規事業・M&A・好決算・黒字転換など

**以下の形式で必ず出力してください：**
🥇 No.1: [企業名] — [選定理由（80字以内）]
🥈 No.2: [企業名] — [選定理由（80字以内）]
🥉 No.3: [企業名] — [選定理由（80字以内）]
4️⃣ No.4: [企業名] — [選定理由（80字以内）]
5️⃣ No.5: [企業名] — [選定理由（80字以内）]

📝 今週の総評（100字以内）：市場全体の傾向
"""
            with st.spinner("AIが成長シグナルを分析中..."):
                try:
                    ai_rank, ai_name_r = generate_ai_comment(_prompt_tdnet)
                    st.markdown(f"### 🤖 AI成長シグナルランキング（{ai_name_r}）")
                    st.success(ai_rank)
                except Exception as _e_td:
                    st.warning(f"AI分析エラー: {_e_td}")

            st.divider()
            with st.expander("📋 取得した適時開示一覧を見る"):
                st.dataframe(
                    df_td[["日付", "企業名", "業種", "タイトル"]].sort_values("日付", ascending=False),
                    use_container_width=True, hide_index=True,
                )

# ── Tab2: TDnet 適時開示 → PDF → AI要約 ───────────────────────────
with edinet_t2:
    st.markdown("#### 📄 適時開示 PDF → Gemini/Groq AI要約（TDnet）")
    st.caption(
        "TDnetから選択銘柄の適時開示を取得し、PDFをそのままGemini/Groqへ渡して自動要約します。"
        "PDFを手動で開く必要はありません。"
    )

    # 銘柄選択
    _company_opts = {
        f"{v[0]}  ({k.replace('.T','')})" : k.replace(".T", "").zfill(4)
        for k, v in ticker_name_map.items()
    }
    col_td2a, col_td2b = st.columns([3, 1])
    with col_td2a:
        sel_company_td = st.selectbox(
            "銘柄を選択", list(_company_opts.keys()), key="tdnet_company_sel"
        )
    with col_td2b:
        sel_days_td2 = st.slider("取得期間（営業日）", 1, 60, 60, key="tdnet_days_company",
                                  help="30日以上は取得に時間がかかります")

    sel_code4_td = _company_opts[sel_company_td]
    sel_cname_td = sel_company_td.split("(")[0].strip()

    run_fetch_td = st.button(
        f"▶ {sel_cname_td} の適時開示を取得", type="primary", key="btn_fetch_tdnet2"
    )

    if run_fetch_td:
        with st.spinner(f"TDnetから {sel_cname_td} の開示を検索中..."):
            _disc_list = fetch_tdnet_week(
                ticker_name_map, days=sel_days_td2, code4_filter=sel_code4_td
            )
        st.session_state["_td2_disclosures"] = _disc_list
        st.session_state["_td2_cname"] = sel_cname_td

    _disc_list = st.session_state.get("_td2_disclosures", [])
    _td2_cname = st.session_state.get("_td2_cname", "")

    if _disc_list:
        df_disc = pd.DataFrame(_disc_list)
        st.success(f"✅ {_td2_cname}: {len(df_disc)} 件の適時開示")

        # セレクトボックスで開示を選択
        disc_opts = [
            f"{r['日付']} {r['時刻']}  {r['タイトル']}"
            for _, r in df_disc.iterrows()
        ]
        sel_disc_idx = st.selectbox(
            "AI要約する開示書類を選択", range(len(disc_opts)),
            format_func=lambda i: disc_opts[i], key="td2_sel_disc"
        )
        sel_disc_row = df_disc.iloc[sel_disc_idx]

        col_pdf1, col_pdf2 = st.columns([3, 1])
        with col_pdf1:
            st.caption(f"📎 PDF: `{sel_disc_row.get('PDF_URL') or '（URLなし）'}`")
        with col_pdf2:
            has_pdf = bool(sel_disc_row.get("PDF_URL"))
            run_summarize = st.button(
                "🤖 PDF取得 → AI要約", type="primary",
                key="run_td2_summary", disabled=not has_pdf
            )
        if not has_pdf:
            st.info("この開示にはPDF URLが取得できませんでした（HTML形式の開示の可能性）")

        if run_summarize and has_pdf:
            pdf_url_td = sel_disc_row["PDF_URL"]
            title_td   = sel_disc_row["タイトル"]
            with st.spinner(f"PDF を取得中…（〜30秒）"):
                pdf_text_td = fetch_tdnet_pdf_text(pdf_url_td)

            if pdf_text_td.startswith("["):
                st.error(f"PDF取得失敗: {pdf_text_td}")
                st.markdown(f"[TDnet で直接開く]({pdf_url_td})")
            else:
                _prompt_td2_pdf = f"""
以下は {_td2_cname}（{sel_disc_row['コード']}）の適時開示「{title_td}」のPDFテキストです。

{pdf_text_td[:4000]}

**400字以内**で以下の観点から要約してください：
1. 📌 開示の概要（何を発表したか）
2. 📈 業績・財務への影響（売上・利益・前年比など具体的数値）
3. 💡 ポジティブな点・成長シグナル
4. ⚠️ リスク・懸念点
5. 🔮 今後の見通し・投資家への示唆
"""
                with st.spinner("Gemini/Groqが要約中..."):
                    try:
                        summary_td, ai_nm_td = generate_ai_comment(_prompt_td2_pdf)
                        st.markdown(
                            f"### 🤖 {_td2_cname}「{title_td}」AI要約（{ai_nm_td}）"
                        )
                        st.success(summary_td)
                        with st.expander("📄 抽出テキスト（先頭2000字）を確認"):
                            st.text(pdf_text_td[:2000])
                    except Exception as _e_td2:
                        st.warning(f"AI要約エラー: {_e_td2}")

        st.divider()
        with st.expander("📋 全開示一覧"):
            st.dataframe(
                df_disc[["日付", "時刻", "タイトル", "PDF_URL"]].reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )
    elif run_fetch_td:
        st.warning(f"⚠️ {sel_cname_td} の適時開示が見つかりませんでした（対象期間: {sel_days_td2}営業日）")

# ── Tab3: 大量保有報告書 監視 ─────────────────────────────────────
with edinet_t3:
    st.markdown("#### 🏦 大量保有報告書 監視（機関投資家・アクティビスト動向）")
    st.caption(
        "株式5%以上の保有変動があると提出義務が生じる「大量保有報告書」を監視。"
        "機関投資家・アクティビストの動向把握に活用できます。"
    )

    col_hd1, col_hd2 = st.columns([2, 1])
    with col_hd1:
        holding_days = st.slider("取得対象期間（営業日）", 1, 60, 30, key="holding_days_sl")
    with col_hd2:
        run_holding = st.button("▶ 大量保有報告書を検索", type="primary", key="run_holding_btn")

    if run_holding:
        with st.spinner("EDINETから大量保有報告書を検索中..."):
            all_docs_h = fetch_edinet_docs(days=holding_days)
            holding_docs = [d for d in all_docs_h if any(
                kw in d["書類種別"] for kw in ["大量保有報告書", "変更報告書"]
            )]

        if not holding_docs:
            st.warning("該当期間に大量保有報告書が見つかりませんでした")
        else:
            df_hold = pd.DataFrame(holding_docs)
            col_hm1, col_hm2 = st.columns(2)
            col_hm1.metric("大量保有報告書 件数", f"{len(df_hold)}件")
            col_hm2.metric("変更報告書を含む", f"{sum('変更' in d['書類種別'] for d in holding_docs)}件")

            tab_h1, tab_h2 = st.tabs(["📋 全件一覧", "🤖 AI動向分析"])

            with tab_h1:
                st.dataframe(
                    df_hold[["日付", "企業名", "書類種別"]].sort_values("日付", ascending=False).reset_index(drop=True),
                    use_container_width=True, hide_index=True,
                )
                csv_hold = df_hold.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "⬇️ CSV ダウンロード", data=csv_hold,
                    file_name=f"large_holdings_{datetime.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv", key="dl_hold",
                )

            with tab_h2:
                hold_summary_text = "\n".join(
                    f"{d['日付']} 【{d['企業名']}】 {d['書類種別']}"
                    for d in holding_docs[:40]
                )
                _prompt_hold = f"""
以下は直近{holding_days}営業日のEDINET大量保有報告書・変更報告書の一覧です。

{hold_summary_text}

機関投資家・アクティビストの動向として注目すべき点を250字以内で解説してください。
特に以下に注目：
- 大量取得（新規5%超え）→ 買い増しシグナル
- 変更報告書で保有率が大幅増加 → アクティビスト候補
- 大量売却（保有率低下） → 機関の撤退シグナル
"""
                with st.spinner("AIが機関投資家動向を分析中..."):
                    try:
                        hold_comment, hold_ai = generate_ai_comment(_prompt_hold)
                        st.info(f"🤖 **AI機関投資家動向分析（{hold_ai}）**\n\n{hold_comment}")
                    except Exception as _e_hold:
                        st.warning(f"AI分析エラー: {_e_hold}")


# ================================================================
# 🔍 LLM as a Judge — 適時開示スコアリング
# ================================================================
st.header("🔍 AI開示評価（LLM as a Judge）")
st.divider()
st.caption(
    "TDnet適時開示をAIが5軸でスコアリング → 重要度ランキング表示。"
    "評価軸: 重要度・市場インパクト・ポジネガ・緊急性・関連テーマ"
)

def _judge_disclosures(disclosures: list[dict]) -> list[dict]:
    """開示リストをAIで一括スコアリング（JSON構造化出力）"""
    import json as _json
    if not disclosures:
        return []

    items_text = "\n".join(
        f"{i+1}. [{r['日付']}] {r['企業名']}（{r['コード']}）: {r['タイトル']}"
        for i, r in enumerate(disclosures)
    )
    prompt = f"""あなたは日本株の機関投資家向けアナリストです。
以下の適時開示リストを評価し、**必ずJSON配列のみ**を返してください（説明文・マークダウン不要）。

## 評価対象
{items_text}

## 評価基準（各項目の定義）
- importance (1–5): 市場全体への重要度。5=決算・業績修正・M&A、1=軽微なお知らせ
- impact ("高"/"中"/"低"): 株価への短期インパクト
- sentiment ("ポジティブ"/"ネガティブ"/"中立"): 投資家心理への影響
- urgency ("高"/"中"/"低"): 即日対応が必要か
- themes (配列, 最大3個): 関連テーマ例: ["半導体","AI","M&A","業績上方修正","株主還元","リストラ"]
- reason (30字以内): スコア根拠

## 出力形式（JSONのみ・他の文字は一切不要）
[
  {{"id":1,"importance":4,"impact":"高","sentiment":"ポジティブ","urgency":"中","themes":["AI"],"reason":"大型受注でEPS押し上げ"}},
  ...
]"""

    raw, model_name = generate_ai_comment(prompt)

    # JSON抽出（コードブロック・余分なテキストを除去）
    import re as _re
    m = _re.search(r'\[[\s\S]*\]', raw)
    if not m:
        return []
    try:
        scores = _json.loads(m.group())
    except Exception:
        return []

    result = []
    score_map = {s["id"]: s for s in scores if "id" in s}
    for i, row in enumerate(disclosures):
        sc = score_map.get(i + 1, {})
        result.append({
            **row,
            "重要度":     sc.get("importance", 0),
            "インパクト": sc.get("impact", "—"),
            "ポジネガ":   sc.get("sentiment", "—"),
            "緊急性":     sc.get("urgency", "—"),
            "テーマ":     "・".join(sc.get("themes", [])),
            "根拠":       sc.get("reason", ""),
            "_model":     model_name,
        })
    return result


# ── コントロール ─────────────────────────────────────────────────
_jdg_c1, _jdg_c2, _jdg_c3 = st.columns([2, 1, 1])
with _jdg_c1:
    _jdg_sector_opts = ["全銘柄（Nikkei225）"] + sorted({
        s for _, (_, s) in ticker_name_map.items()
    })
    _jdg_sector = st.selectbox("絞り込み業種", _jdg_sector_opts, key="jdg_sector")
with _jdg_c2:
    _jdg_days = st.radio("取得期間", [1, 3, 5], index=1,
                         horizontal=True, key="jdg_days", format_func=lambda x: f"{x}営業日")
with _jdg_c3:
    st.markdown(""); st.markdown("")
    _jdg_run = st.button("▶ AI評価を実行", type="primary", key="jdg_run")

# ── フィルタ UI ──────────────────────────────────────────────────
_jdg_f1, _jdg_f2, _jdg_f3 = st.columns(3)
with _jdg_f1:
    _jdg_min_imp = st.slider("重要度フィルタ（以上）", 1, 5, 1, key="jdg_min_imp")
with _jdg_f2:
    _jdg_impact_f = st.multiselect("インパクト", ["高", "中", "低"],
                                    default=["高", "中"], key="jdg_impact_f")
with _jdg_f3:
    _jdg_sent_f = st.multiselect("ポジネガ", ["ポジティブ", "中立", "ネガティブ"],
                                  default=["ポジティブ", "中立", "ネガティブ"], key="jdg_sent_f")

if _jdg_run:
    # 対象 code_map 構築
    if _jdg_sector == "全銘柄（Nikkei225）":
        _jdg_map = ticker_name_map
    else:
        _jdg_map = {t: (n, s) for t, (n, s) in ticker_name_map.items() if s == _jdg_sector}

    with st.spinner(f"TDnetから {_jdg_days} 営業日分の開示を取得中..."):
        _jdg_raw = fetch_tdnet_week(_jdg_map, days=_jdg_days)

    if not _jdg_raw:
        st.warning("取得できた適時開示がありませんでした。期間を延ばすか平日にお試しください。")
    else:
        st.info(f"📋 {len(_jdg_raw)} 件の開示を取得。AIがスコアリング中...")
        # 最大30件（API負荷制限）
        _jdg_batch = _jdg_raw[:30]
        with st.spinner("AIが評価中（Gemini → Groqフォールバック）..."):
            _jdg_scored = _judge_disclosures(_jdg_batch)

        if not _jdg_scored:
            st.error("AI評価の解析に失敗しました。もう一度お試しください。")
        else:
            _model_used = _jdg_scored[0].get("_model", "?")
            st.success(f"✅ {len(_jdg_scored)} 件を評価完了（by {_model_used}）")

            # フィルタ適用
            _jdg_df = pd.DataFrame(_jdg_scored)
            _jdg_df = _jdg_df[
                (_jdg_df["重要度"] >= _jdg_min_imp) &
                (_jdg_df["インパクト"].isin(_jdg_impact_f + ["—"])) &
                (_jdg_df["ポジネガ"].isin(_jdg_sent_f + ["—"]))
            ].sort_values("重要度", ascending=False).reset_index(drop=True)

            # ── サマリーメトリクス
            _jm1, _jm2, _jm3, _jm4 = st.columns(4)
            _jm1.metric("評価件数", f"{len(_jdg_df)} 件")
            _jm2.metric("高インパクト", f"{(_jdg_df['インパクト']=='高').sum()} 件")
            _jm3.metric("ポジティブ", f"{(_jdg_df['ポジネガ']=='ポジティブ').sum()} 件")
            _jm4.metric("平均重要度", f"{_jdg_df['重要度'].mean():.1f} / 5")

            st.divider()

            # ── 重要度スコア分布（横棒グラフ）
            _jdg_dist = _jdg_df["重要度"].value_counts().sort_index(ascending=False)
            _fig_dist, _ax_dist = plt.subplots(figsize=(7, 2.2))
            _colors_dist = {5:"#c62828",4:"#ef6c00",3:"#f9a825",2:"#1565c0",1:"#aaa"}
            _ax_dist.barh(
                [f"★{v}" for v in _jdg_dist.index],
                _jdg_dist.values,
                color=[_colors_dist.get(v,"#aaa") for v in _jdg_dist.index],
                alpha=0.85
            )
            for i, v in enumerate(_jdg_dist.values):
                _ax_dist.text(v + 0.1, i, str(v), va="center", fontsize=9)
            _ax_dist.set_title("重要度スコア分布", fontsize=10)
            _ax_dist.set_xlabel("件数"); _ax_dist.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(_fig_dist, clear_figure=True)

            # ── ランキングテーブル
            st.markdown("#### 📊 開示ランキング（重要度順）")

            _SENT_COLOR = {"ポジティブ":"#1b5e20","ネガティブ":"#b71c1c","中立":"#555"}
            _IMP_COLOR  = {"高":"#c62828","中":"#ef6c00","低":"#1565c0"}

            for _, row in _jdg_df.iterrows():
                _imp_stars = "★" * int(row["重要度"]) + "☆" * (5 - int(row["重要度"]))
                _sc = _SENT_COLOR.get(row["ポジネガ"], "#555")
                _ic = _IMP_COLOR.get(row["インパクト"], "#555")
                with st.container():
                    st.markdown(
                        f"<div style='border-left:4px solid {_ic};padding:8px 12px;margin:4px 0;"
                        f"border-radius:0 6px 6px 0;background:#fafafa'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                        f"<span style='font-size:.85rem;color:gray'>{row['日付']} | {row['企業名']}（{row['コード']}）</span>"
                        f"<span style='font-size:.95rem'>{_imp_stars}</span></div>"
                        f"<div style='font-weight:bold;margin:2px 0'>{row['タイトル']}</div>"
                        f"<div style='display:flex;gap:8px;font-size:.82rem;margin-top:4px'>"
                        f"<span style='color:{_ic}'>インパクト:{row['インパクト']}</span>"
                        f"<span style='color:{_sc}'>{row['ポジネガ']}</span>"
                        f"<span>緊急性:{row['緊急性']}</span>"
                        f"<span style='color:#555'>🏷 {row['テーマ']}</span>"
                        f"</div>"
                        f"<div style='font-size:.8rem;color:#666;margin-top:2px'>💬 {row['根拠']}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            st.divider()

            # ── CSVダウンロード
            _dl_cols = ["日付","企業名","コード","タイトル","重要度","インパクト","ポジネガ","緊急性","テーマ","根拠"]
            _dl_df = _jdg_df[[c for c in _dl_cols if c in _jdg_df.columns]]
            st.download_button(
                "📥 評価結果をCSVダウンロード",
                _dl_df.to_csv(index=False, encoding="utf-8-sig"),
                file_name=f"tdnet_judge_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="jdg_download"
            )


# ================================================================
# J-Quants APIクライアント（V2対応）
# ================================================================
JQUANTS_API_BASE = "https://api.jquants.com/v1"
_JQ_RESPONSE_KEYS = {
    "/equities/bars/daily":      "daily_quotes",
    "/fins/summary":             "statements",
    "/indices/bars/daily/topix": "topix",
    "/equities/investor-types":  "investor_type",
    "/markets/margin-interest":  "margin_interest",
    "/markets/short-ratio":      "short_ratio",
}

def _jq_headers():
    api_key = st.secrets.get("JQUANTS_API_KEY", "")
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

def _jq_get(endpoint, params=None, debug=False):
    headers = _jq_headers()
    if not headers:
        return {"error": "NO_API_KEY"}
    try:
        res = requests.get(f"{JQUANTS_API_BASE}{endpoint}", params=params or {}, headers=headers, timeout=20)
        if debug:
            return {"status": res.status_code, "raw": res.text[:2000], "json": res.json() if res.status_code==200 else {}}
        if res.status_code != 200:
            return {"error": res.status_code, "msg": res.text[:300]}
        d = res.json()
        known_key = _JQ_RESPONSE_KEYS.get(endpoint)
        data_key = known_key if (known_key and known_key in d) else next(
            (k for k in d if k != "pagination_key" and isinstance(d.get(k), list)), None)
        if not data_key:
            return d
        all_data = list(d[data_key])
        while "pagination_key" in d:
            p = dict(params or {}); p["pagination_key"] = d["pagination_key"]
            r2 = requests.get(f"{JQUANTS_API_BASE}{endpoint}", params=p, headers=headers, timeout=20)
            if r2.status_code != 200: break
            d = r2.json(); all_data += list(d.get(data_key, []))
        return {data_key: all_data}
    except Exception as e:
        return {"error": str(e)}

def _jq_to_df(d, endpoint):
    if not d or "error" in d:
        return pd.DataFrame()
    known_key = _JQ_RESPONSE_KEYS.get(endpoint)
    data_key = known_key if (known_key and known_key in d) else next(
        (k for k in d if isinstance(d.get(k), list)), None)
    if not data_key or not d[data_key]:
        return pd.DataFrame()
    df = pd.DataFrame(d[data_key])
    date_col = next((c for c in df.columns if c.lower() in ["date","publisheddate","discloseddate"]), None)
    if date_col and date_col != "Date":
        df = df.rename(columns={date_col: "Date"})
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date")
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def jq_fetch_stock_bars(code, date_from, date_to):
    return _jq_to_df(_jq_get("/equities/bars/daily", {"code": code, "from": date_from, "to": date_to}), "/equities/bars/daily")

@st.cache_data(ttl=3600, show_spinner=False)
def jq_fetch_topix(date_from, date_to):
    return _jq_to_df(_jq_get("/indices/bars/daily/topix", {"date_from": date_from, "date_to": date_to}), "/indices/bars/daily/topix")

@st.cache_data(ttl=3600, show_spinner=False)
def jq_fetch_investor_types(date_from, date_to):
    return _jq_to_df(_jq_get("/equities/investor-types", {"from": date_from, "to": date_to}), "/equities/investor-types")

@st.cache_data(ttl=3600, show_spinner=False)
def jq_fetch_margin(code, date_from, date_to):
    return _jq_to_df(_jq_get("/markets/margin-interest", {"code": code, "from": date_from, "to": date_to}), "/markets/margin-interest")

@st.cache_data(ttl=3600, show_spinner=False)
def jq_fetch_short_ratio(s33, date_from, date_to):
    return _jq_to_df(_jq_get("/markets/short-ratio", {"s33": s33, "from": date_from, "to": date_to}), "/markets/short-ratio")

@st.cache_data(ttl=3600, show_spinner=False)
def jq_fetch_fins(code):
    return _jq_to_df(_jq_get("/fins/summary", {"code": code}), "/fins/summary")

def _plot_candlestick_jq(df, title):
    if df.empty:
        st.warning("データなし"); return
    open_col  = next((c for c in df.columns if c.lower() in ["open","openingprice"]), None)
    high_col  = next((c for c in df.columns if c.lower() in ["high","highprice"]), None)
    low_col   = next((c for c in df.columns if c.lower() in ["low","lowprice"]), None)
    close_col = next((c for c in df.columns if c.lower() in ["close","closeprice"]), None)
    if not all([open_col, high_col, low_col, close_col]):
        st.dataframe(df.tail(20)); return
    fig, ax = plt.subplots(figsize=(12, 4))
    for _, row in df.tail(60).iterrows():
        o, h, l, c = row[open_col], row[high_col], row[low_col], row[close_col]
        color = "#1a7f37" if c >= o else "#d1242f"
        ax.plot([row["Date"], row["Date"]], [l, h], color=color, linewidth=0.8)
        ax.bar(row["Date"], abs(c - o), bottom=min(o, c), color=color, alpha=0.85, width=1.2)
    ax.set_title(title, fontsize=11); ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25); plt.xticks(rotation=45); plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


# ─────────────────────────────────────────────────────────────────
st.header("🏦 J-Quants 需給分析")
st.divider()
st.subheader("🏦 J-Quants 需給分析")
st.caption("J-Quants API V2を使用した日本株の公式データ（JPX提供）")

jq_key = st.secrets.get("JQUANTS_API_KEY", "")
if not jq_key:
    st.warning(
        "⚠️ J-Quants APIキー未設定\n\n"
        "Streamlit Secrets に `JQUANTS_API_KEY = 'your_key'` を追加してください。\n"
        "[J-Quants Webサイト](https://jpx-jquants.com/) のダッシュボードからAPIキーを取得できます。"
    )
else:
    col_jq1, col_jq2, col_jq3 = st.columns(3)
    with col_jq1:
        jq_code_input = st.text_input("銘柄コード（4桁）", value="72030", key="jq_code")
        jq_code = jq_code_input.strip()
    with col_jq2:
        jq_period = st.selectbox("取得期間", ["3ヶ月","6ヶ月","1年"], index=1, key="jq_period")
        period_days_jq = {"3ヶ月":90,"6ヶ月":180,"1年":365}[jq_period]
    with col_jq3:
        st.markdown("")
        st.markdown("")
        run_jq = st.button("▶ J-Quants データ取得", type="primary", key="run_jq")

    if run_jq:
        end_jq   = datetime.today()
        jq_date_to   = end_jq.strftime("%Y%m%d")
        jq_date_from = (end_jq - relativedelta(days=period_days_jq)).strftime("%Y%m%d")

        # 診断モード
        with st.expander("🔧 API診断（動作しない場合はここを確認）", expanded=False):
            raw = _jq_get("/fins/summary", {"code": jq_code}, debug=True)
            st.json(raw)

        jq_t1, jq_t2, jq_t3, jq_t4, jq_t5 = st.tabs([
            "📈 株価・TOPIX","👥 投資部門別","⚖️ 信用取引残高","📉 空売り比率","📋 財務情報"
        ])

        with jq_t1:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                with st.spinner(f"{jq_code} 株価取得中..."):
                    df_bars = jq_fetch_stock_bars(jq_code, jq_date_from, jq_date_to)
                if df_bars.empty:
                    st.warning("株価データ取得失敗（銘柄コードまたはプランを確認）")
                else:
                    st.caption(f"{len(df_bars)}日分取得")
                    _plot_candlestick_jq(df_bars, f"{jq_code} 株価（ローソク足）")
                    with st.expander("生データ"): st.dataframe(df_bars.tail(20), use_container_width=True)
            with col_p2:
                with st.spinner("TOPIX取得中..."):
                    df_topix = jq_fetch_topix(jq_date_from, jq_date_to)
                if df_topix.empty:
                    st.warning("TOPIXデータ取得失敗（Lightプラン以上が必要）")
                else:
                    close_col = next((c for c in df_topix.columns if c.lower() in ["close","closeprice"]), None)
                    if close_col:
                        fig_t, ax_t = plt.subplots(figsize=(6,4))
                        ax_t.plot(df_topix["Date"], df_topix[close_col], color="#1565c0", linewidth=1.5)
                        ax_t.fill_between(df_topix["Date"], df_topix[close_col], df_topix[close_col].min(), alpha=0.1, color="#1565c0")
                        ax_t.set_title("TOPIX", fontsize=11); ax_t.grid(True, alpha=0.25)
                        plt.xticks(rotation=45); plt.tight_layout()
                        st.pyplot(fig_t, clear_figure=True)

        with jq_t2:
            st.caption("Lightプラン以上が必要 | 週次データ（毎週第4営業日更新）")
            with st.spinner("投資部門別データ取得中..."):
                df_inv = jq_fetch_investor_types(jq_date_from, jq_date_to)
            if df_inv.empty:
                st.warning("データ取得失敗（Lightプラン以上が必要）")
            else:
                section_col = next((c for c in df_inv.columns if "section" in c.lower()), None)
                buy_col     = next((c for c in df_inv.columns if "buy" in c.lower()), None)
                sell_col    = next((c for c in df_inv.columns if "sell" in c.lower()), None)
                if section_col and buy_col and sell_col and "Date" in df_inv.columns:
                    fig_inv, ax_inv = plt.subplots(figsize=(12,5))
                    for sec, grp in df_inv.groupby(section_col):
                        net = grp[buy_col].astype(float) - grp[sell_col].astype(float)
                        ax_inv.plot(grp["Date"], net, label=str(sec), linewidth=1.5, marker="o", markersize=3)
                    ax_inv.axhline(0, color="gray", linestyle="--", alpha=0.5)
                    ax_inv.set_title("Investor Type Net Buy/Sell", fontsize=11)
                    ax_inv.legend(fontsize=8); ax_inv.grid(True, alpha=0.25)
                    plt.xticks(rotation=45); plt.tight_layout()
                    st.pyplot(fig_inv, clear_figure=True)
                else:
                    st.dataframe(df_inv, use_container_width=True)
                with st.expander("生データ"): st.dataframe(df_inv, use_container_width=True)

        with jq_t3:
            st.caption("Standardプラン以上が必要 | 週次データ")
            with st.spinner("信用取引残高取得中..."):
                df_mg = jq_fetch_margin(jq_code, jq_date_from, jq_date_to)
            if df_mg.empty:
                st.warning("データ取得失敗")
            else:
                buy_bal  = next((c for c in df_mg.columns if "longmargin" in c.lower()), None)
                sell_bal = next((c for c in df_mg.columns if "shortmargin" in c.lower()), None)
                if buy_bal and sell_bal and "Date" in df_mg.columns:
                    fig_mg, ax_mg = plt.subplots(figsize=(12,4))
                    ax_mg.plot(df_mg["Date"], df_mg[buy_bal].astype(float), label="Long", color="#1a7f37", linewidth=1.8)
                    ax_mg.plot(df_mg["Date"], df_mg[sell_bal].astype(float), label="Short", color="#d1242f", linewidth=1.8)
                    ax_mg.set_title(f"{jq_code} Margin Balance", fontsize=11)
                    ax_mg.legend(); ax_mg.grid(True, alpha=0.25)
                    plt.xticks(rotation=45); plt.tight_layout()
                    st.pyplot(fig_mg, clear_figure=True)
                    ratio = df_mg[buy_bal].astype(float) / (df_mg[sell_bal].astype(float) + 1e-8)
                    c1, c2 = st.columns(2)
                    c1.metric("最新 信用倍率", f"{ratio.iloc[-1]:.2f}倍",
                              delta=f"{ratio.iloc[-1]-ratio.iloc[-2]:+.2f}" if len(ratio)>1 else None)
                    c2.metric("信用買残", f"{int(df_mg[buy_bal].iloc[-1]):,}株")
                else:
                    st.dataframe(df_mg, use_container_width=True)
                with st.expander("生データ"): st.dataframe(df_mg, use_container_width=True)

        with jq_t4:
            st.caption("Standardプラン以上が必要 | 33業種コードで取得")
            S33_OPTIONS = {
                "3650 電気機器":"3650","3700 輸送用機器":"3700","5250 情報・通信":"5250",
                "7050 銀行":"7050","3200 化学":"3200","3600 機械":"3600",
                "6100 小売":"6100","8050 不動産":"8050","9050 サービス":"9050",
                "0050 水産・農林":"0050","2050 建設":"2050","3050 食料品":"3050",
                "3450 鉄鋼":"3450","3500 非鉄":"3500","5050 陸運":"5050",
                "5100 海運":"5100","5150 空運":"5150","7100 証券":"7100",
                "7150 保険":"7150","8050 不動産":"8050",
            }
            selected_s33_label = st.selectbox("業種コード", list(S33_OPTIONS.keys()), key="jq_s33")
            selected_s33 = S33_OPTIONS[selected_s33_label]
            with st.spinner("業種別空売り比率取得中..."):
                df_sr = jq_fetch_short_ratio(selected_s33, jq_date_from, jq_date_to)
            if df_sr.empty:
                st.warning("データ取得失敗")
            else:
                ratio_col = next((c for c in df_sr.columns if "ratio" in c.lower()), None)
                if ratio_col and "Date" in df_sr.columns:
                    fig_sr, ax_sr = plt.subplots(figsize=(12,4))
                    ax_sr.plot(df_sr["Date"], df_sr[ratio_col].astype(float)*100, color="#7b1fa2", linewidth=1.8)
                    ax_sr.fill_between(df_sr["Date"], df_sr[ratio_col].astype(float)*100, alpha=0.15, color="#7b1fa2")
                    ax_sr.set_title(f"Short Ratio - {selected_s33_label} (%)", fontsize=11)
                    ax_sr.grid(True, alpha=0.25); plt.xticks(rotation=45); plt.tight_layout()
                    st.pyplot(fig_sr, clear_figure=True)
                    latest_sr = float(df_sr[ratio_col].iloc[-1])*100
                    avg_sr    = float(df_sr[ratio_col].mean())*100
                    c1, c2 = st.columns(2)
                    c1.metric("最新空売り比率", f"{latest_sr:.1f}%")
                    c2.metric("期間平均", f"{avg_sr:.1f}%", delta=f"{latest_sr-avg_sr:+.1f}%")
                else:
                    st.dataframe(df_sr, use_container_width=True)

        with jq_t5:
            st.caption("Freeプラン以上で利用可能")
            with st.spinner("財務情報取得中..."):
                df_fins = jq_fetch_fins(jq_code)
            if df_fins.empty:
                st.warning("財務データ取得失敗")
            else:
                st.caption(f"取得件数: {len(df_fins)}件")
                key_cols = [c for c in df_fins.columns if any(k in c.lower() for k in
                    ["date","period","sales","profit","income","eps","revenue","operating","net","equity"])]
                st.dataframe(df_fins[key_cols].tail(8) if key_cols else df_fins.tail(8),
                             use_container_width=True)
                if st.checkbox("🤖 AI財務分析", key="jq_ai_fins"):
                    fins_str = df_fins.tail(4).to_string(index=False)
                    prompt_fins = (
                        f"銘柄コード {jq_code} の直近4四半期財務情報:\n\n{fins_str}\n\n"
                        "投資家向けに300文字以内で:\n1. 売上・利益のトレンド\n2. 財務健全性\n3. 注目点"
                    )
                    with st.spinner("AI財務分析中..."):
                        try:
                            comment, ai_name = generate_ai_comment(prompt_fins)
                            st.info(f"🤖 **AI財務分析（{ai_name}）**\n\n{comment}")
                        except Exception as e:
                            st.warning(f"AI APIエラー: {e}")
    else:
        st.info(
            "「▶ J-Quants データ取得」ボタンを押すとデータを取得します。\n\n"
            "- 📈 株価ローソク足チャート・TOPIX（Freeプラン以上）\n"
            "- 👥 投資部門別売買動向（Lightプラン以上）\n"
            "- ⚖️ 信用取引残高・信用倍率（Standardプラン以上）\n"
            "- 📉 業種別空売り比率（Standardプラン以上）\n"
            "- 📋 財務情報・決算短信 + AI分析（Freeプラン以上）"
        )


# =================================================================
# Finnhub + Alpha Vantage セクション
# =================================================================

import time as _time

def _fh_get(endpoint: str, params: dict = {}) -> dict:
    """Finnhub APIリクエスト"""
    try:
        key = st.secrets.get("FINNHUB_API_KEY", "")
        if not key:
            return {}
        r = requests.get(
            f"https://finnhub.io/api/v1{endpoint}",
            params={**params, "token": key},
            timeout=10,
        )
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

def _av_get(func: str, params: dict = {}) -> dict:
    """Alpha Vantage APIリクエスト"""
    try:
        key = st.secrets.get("ALPHA_VANTAGE_KEY", "")
        if not key:
            return {}
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": func, "apikey": key, **params},
            timeout=15,
        )
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


st.markdown("""<div style="background:#bf360c;color:white;padding:12px 22px;border-radius:8px;margin:28px 0 4px 0;font-size:18px;font-weight:bold;">
🗞️ D.&nbsp;外部データ・API &nbsp;<span style="font-size:12px;font-weight:400;opacity:.88">Finnhub / Alpha Vantage</span></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
st.header("📡 Finnhub リアルタイム情報")
st.caption("Finnhub APIによるリアルタイム株価・決算・インサイダー・ニュース")

fh_key = st.secrets.get("FINNHUB_API_KEY", "")
if not fh_key:
    st.warning("⚠️ `FINNHUB_API_KEY` を Streamlit Secrets に追加してください")
else:
    fh_t1, fh_t2, fh_t3, fh_t4 = st.tabs([
        "💹 リアルタイム株価",
        "📊 決算サプライズ",
        "🕵️ インサイダー取引",
        "📰 企業ニュース",
    ])

    # ── Tab1: リアルタイム株価・為替 ────────────────────────────
    with fh_t1:
        st.markdown("#### 💹 リアルタイム株価・為替クォート")

        col_fh1, col_fh2 = st.columns(2)
        with col_fh1:
            st.markdown("**米国株 (例: AAPL, TSLA, NVDA)**")
            us_syms = st.text_input(
                "ティッカー（カンマ区切り）",
                value="AAPL,TSLA,NVDA,MSFT,GOOGL",
                key="fh_us_syms"
            ).upper().split(",")

            quote_rows = []
            for sym in [s.strip() for s in us_syms if s.strip()]:
                q = _fh_get("/quote", {"symbol": sym})
                if q.get("c"):
                    chg = q["c"] - q["pc"]
                    chg_pct = chg / q["pc"] * 100 if q["pc"] else 0
                    quote_rows.append({
                        "銘柄": sym,
                        "現在値": q["c"],
                        "前日比": round(chg, 2),
                        "変化率(%)": round(chg_pct, 2),
                        "高値": q.get("h", "-"),
                        "安値": q.get("l", "-"),
                        "始値": q.get("o", "-"),
                    })

            if quote_rows:
                df_q = pd.DataFrame(quote_rows)
                def _color_chg(val):
                    if isinstance(val, float):
                        if val > 0: return "color:#1a7f37;font-weight:bold"
                        if val < 0: return "color:#d1242f;font-weight:bold"
                    return ""
                st.dataframe(
                    df_q.style.map(_color_chg, subset=["前日比","変化率(%)"]),
                    use_container_width=True, hide_index=True
                )

        with col_fh2:
            st.markdown("**為替クォート (Forex)**")
            fx_pairs = [
                ("USD","JPY"),("EUR","JPY"),("GBP","JPY"),
                ("EUR","USD"),("AUD","JPY"),
            ]
            fx_rows = []
            for base, quote in fx_pairs:
                d = _fh_get("/forex/rates", {"base": base})
                rates = d.get("quote", {})
                if quote in rates:
                    fx_rows.append({
                        "ペア": f"{base}/{quote}",
                        "レート": round(float(rates[quote]), 4),
                    })
            if fx_rows:
                st.dataframe(pd.DataFrame(fx_rows), use_container_width=True, hide_index=True)

    # ── Tab2: 決算サプライズ ─────────────────────────────────────
    with fh_t2:
        st.markdown("#### 📊 決算サプライズ（EPS予想 vs 実績）")
        eps_sym = st.text_input("銘柄コード", value="AAPL", key="fh_eps_sym").upper()
        if eps_sym:
            data = _fh_get("/stock/earnings", {"symbol": eps_sym, "limit": 8})
            if data:
                rows = []
                for item in (data if isinstance(data, list) else []):
                    surprise = item.get("surprise", 0) or 0
                    rows.append({
                        "決算期": item.get("period", ""),
                        "EPS予想": item.get("estimate", "-"),
                        "EPS実績": item.get("actual", "-"),
                        "サプライズ": round(surprise, 4),
                        "サプライズ(%)": round(item.get("surprisePercent", 0) or 0, 2),
                    })
                if rows:
                    df_eps = pd.DataFrame(rows)
                    def _color_sur(val):
                        if isinstance(val, (int, float)):
                            if val > 0: return "color:#1a7f37;font-weight:bold"
                            if val < 0: return "color:#d1242f;font-weight:bold"
                        return ""
                    st.dataframe(
                        df_eps.style.map(_color_sur, subset=["サプライズ","サプライズ(%)"]),
                        use_container_width=True, hide_index=True
                    )
                    # サプライズ推移チャート
                    if "サプライズ(%)" in df_eps.columns and len(df_eps) > 1:
                        fig_eps, ax_eps = plt.subplots(figsize=(8, 3))
                        colors = ["#1a7f37" if v >= 0 else "#d1242f"
                                  for v in df_eps["サプライズ(%)"][::-1]]
                        ax_eps.bar(df_eps["決算期"][::-1],
                                   df_eps["サプライズ(%)"][::-1], color=colors)
                        ax_eps.axhline(0, color="gray", linewidth=0.8)
                        ax_eps.set_title(f"{eps_sym} EPS Surprise (%)", fontsize=11)
                        ax_eps.set_ylabel("Surprise %")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig_eps, clear_figure=True)
                else:
                    st.info("決算データなし")
            else:
                st.warning("データ取得失敗（銘柄コードを確認）")

    # ── Tab3: インサイダー取引 ───────────────────────────────────
    with fh_t3:
        st.markdown("#### 🕵️ インサイダー取引情報")
        ins_sym = st.text_input("銘柄コード", value="AAPL", key="fh_ins_sym").upper()
        if ins_sym:
            data = _fh_get("/stock/insider-transactions", {"symbol": ins_sym})
            txns = data.get("data", [])[:20]
            if txns:
                rows = []
                for t in txns:
                    rows.append({
                        "日付": t.get("transactionDate", ""),
                        "氏名": t.get("name", ""),
                        "役職": t.get("share", ""),
                        "取引種別": t.get("transactionCode", ""),
                        "株数": t.get("share", 0),
                        "単価": t.get("transactionPrice", "-"),
                        "売買区分": "買い" if str(t.get("transactionCode","")) in ["P","A"] else "売り",
                    })
                df_ins = pd.DataFrame(rows)
                def _color_trade(val):
                    if val == "買い": return "color:#1a7f37;font-weight:bold"
                    if val == "売り": return "color:#d1242f;font-weight:bold"
                    return ""
                st.dataframe(
                    df_ins.style.map(_color_trade, subset=["売買区分"]),
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("インサイダー取引データなし")

    # ── Tab4: 企業ニュース ───────────────────────────────────────
    with fh_t4:
        st.markdown("#### 📰 企業ニュースフィード")
        news_sym = st.text_input("銘柄コード", value="AAPL", key="fh_news_sym").upper()
        from datetime import timedelta
        today_str = datetime.today().strftime("%Y-%m-%d")
        week_ago  = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

        if news_sym:
            items = _fh_get("/company-news", {
                "symbol": news_sym, "from": week_ago, "to": today_str
            })
            if isinstance(items, list) and items:
                for item in items[:15]:
                    with st.expander(f"[{item.get('source','')}] {item.get('headline','')[:80]}"):
                        st.markdown(f"**{item.get('headline','')}**")
                        st.caption(f"📅 {item.get('datetime','')}")
                        st.write(item.get("summary","")[:300])
                        if item.get("url"):
                            st.markdown(f"[🔗 記事を開く]({item['url']})")
            else:
                st.info("ニュースデータなし（直近7日）")


# ─────────────────────────────────────────────────────────────────
st.header("📈 Alpha Vantage テクニカル分析・経済指標")
st.caption("Alpha Vantage APIによるテクニカル指標・経済指標・セクター分析")

av_key = st.secrets.get("ALPHA_VANTAGE_KEY", "")
if not av_key:
    st.warning("⚠️ `ALPHA_VANTAGE_KEY` を Streamlit Secrets に追加してください")
else:
    av_t1, av_t2, av_t3 = st.tabs([
        "📉 テクニカル指標",
        "🌐 経済指標",
        "🏭 セクターパフォーマンス",
    ])

    # ── Tab1: テクニカル指標 ─────────────────────────────────────
    with av_t1:
        st.markdown("#### 📉 テクニカル指標（RSI・MACD・ボリンジャーバンド）")

        col_av1, col_av2 = st.columns(2)
        with col_av1:
            av_sym = st.text_input("銘柄コード", value="AAPL", key="av_sym").upper()
        with col_av2:
            av_interval = st.selectbox("時間足", ["daily","weekly","monthly"], key="av_int")

        if av_sym:
            st.markdown("---")
            # RSI
            with st.spinner("RSI取得中..."):
                rsi_data = _av_get("RSI", {
                    "symbol": av_sym, "interval": av_interval,
                    "time_period": 14, "series_type": "close"
                })
            rsi_ts = rsi_data.get("Technical Analysis: RSI", {})

            # MACD
            with st.spinner("MACD取得中..."):
                macd_data = _av_get("MACD", {
                    "symbol": av_sym, "interval": av_interval,
                    "series_type": "close"
                })
            macd_ts = macd_data.get("Technical Analysis: MACD", {})

            # BBANDS
            with st.spinner("ボリンジャーバンド取得中..."):
                bb_data = _av_get("BBANDS", {
                    "symbol": av_sym, "interval": av_interval,
                    "time_period": 20, "series_type": "close"
                })
            bb_ts = bb_data.get("Technical Analysis: BBANDS", {})

            if rsi_ts:
                dates = sorted(rsi_ts.keys(), reverse=True)[:60]
                df_rsi = pd.DataFrame({
                    "Date": pd.to_datetime(dates),
                    "RSI":  [float(rsi_ts[d]["RSI"]) for d in dates],
                })
                fig_rsi, ax_rsi = plt.subplots(figsize=(12, 3))
                ax_rsi.plot(df_rsi["Date"], df_rsi["RSI"], color="#1565c0", linewidth=1.8)
                ax_rsi.axhline(70, color="#d1242f", linestyle="--", alpha=0.7, label="過買い(70)")
                ax_rsi.axhline(30, color="#1a7f37", linestyle="--", alpha=0.7, label="過売り(30)")
                ax_rsi.fill_between(df_rsi["Date"], 70, df_rsi["RSI"].clip(lower=70),
                                    alpha=0.15, color="#d1242f")
                ax_rsi.fill_between(df_rsi["Date"], df_rsi["RSI"].clip(upper=30), 30,
                                    alpha=0.15, color="#1a7f37")
                ax_rsi.set_title(f"{av_sym} RSI(14) - {av_interval}", fontsize=11)
                ax_rsi.set_ylim(0, 100)
                ax_rsi.legend(fontsize=8)
                ax_rsi.grid(True, alpha=0.25)
                plt.xticks(rotation=45); plt.tight_layout()
                st.pyplot(fig_rsi, clear_figure=True)

                latest_rsi = float(rsi_ts[dates[0]]["RSI"])
                if latest_rsi >= 70:
                    st.warning(f"⚠️ RSI {latest_rsi:.1f} — 過買い圏（売り圧力に注意）")
                elif latest_rsi <= 30:
                    st.success(f"✅ RSI {latest_rsi:.1f} — 過売り圏（反発候補）")
                else:
                    st.info(f"📊 RSI {latest_rsi:.1f} — 中立圏")

            if macd_ts:
                dates_m = sorted(macd_ts.keys(), reverse=True)[:60]
                df_macd = pd.DataFrame({
                    "Date":     pd.to_datetime(dates_m),
                    "MACD":     [float(macd_ts[d]["MACD"]) for d in dates_m],
                    "Signal":   [float(macd_ts[d]["MACD_Signal"]) for d in dates_m],
                    "Hist":     [float(macd_ts[d]["MACD_Hist"]) for d in dates_m],
                })
                fig_macd, (ax_m1, ax_m2) = plt.subplots(2, 1, figsize=(12, 5),
                                                          gridspec_kw={"height_ratios": [2,1]})
                ax_m1.plot(df_macd["Date"], df_macd["MACD"],
                           color="#1565c0", linewidth=1.5, label="MACD")
                ax_m1.plot(df_macd["Date"], df_macd["Signal"],
                           color="#e91e63", linewidth=1.5, linestyle="--", label="Signal")
                ax_m1.axhline(0, color="gray", linewidth=0.6)
                ax_m1.legend(fontsize=8); ax_m1.grid(True, alpha=0.25)
                ax_m1.set_title(f"{av_sym} MACD - {av_interval}", fontsize=11)
                colors_hist = ["#1a7f37" if v >= 0 else "#d1242f" for v in df_macd["Hist"]]
                ax_m2.bar(df_macd["Date"], df_macd["Hist"], color=colors_hist, alpha=0.8, width=2)
                ax_m2.axhline(0, color="gray", linewidth=0.6)
                ax_m2.set_ylabel("Histogram"); ax_m2.grid(True, alpha=0.25)
                plt.xticks(rotation=45); plt.tight_layout()
                st.pyplot(fig_macd, clear_figure=True)

            if bb_ts:
                dates_b = sorted(bb_ts.keys(), reverse=True)[:60]
                df_bb = pd.DataFrame({
                    "Date":  pd.to_datetime(dates_b),
                    "Upper": [float(bb_ts[d]["Real Upper Band"]) for d in dates_b],
                    "Mid":   [float(bb_ts[d]["Real Middle Band"]) for d in dates_b],
                    "Lower": [float(bb_ts[d]["Real Lower Band"]) for d in dates_b],
                })
                fig_bb, ax_bb = plt.subplots(figsize=(12, 4))
                ax_bb.plot(df_bb["Date"], df_bb["Upper"],
                           color="#d1242f", linewidth=1.2, linestyle="--", label="Upper")
                ax_bb.plot(df_bb["Date"], df_bb["Mid"],
                           color="#1565c0", linewidth=1.5, label="Middle(SMA20)")
                ax_bb.plot(df_bb["Date"], df_bb["Lower"],
                           color="#1a7f37", linewidth=1.2, linestyle="--", label="Lower")
                ax_bb.fill_between(df_bb["Date"], df_bb["Upper"], df_bb["Lower"],
                                   alpha=0.07, color="#1565c0")
                ax_bb.legend(fontsize=8); ax_bb.grid(True, alpha=0.25)
                ax_bb.set_title(f"{av_sym} Bollinger Bands(20) - {av_interval}", fontsize=11)
                plt.xticks(rotation=45); plt.tight_layout()
                st.pyplot(fig_bb, clear_figure=True)

            if not any([rsi_ts, macd_ts, bb_ts]):
                st.warning("データ取得失敗（無料枠は1分5回制限。少し待ってから再試行してください）")

    # ── Tab2: 経済指標 ───────────────────────────────────────────
    with av_t2:
        st.markdown("#### 🌐 主要経済指標")

        INDICATORS = {
            "実質GDP成長率(米)":        ("REAL_GDP",            "annualReports"),
            "CPI（インフレ率）":         ("CPI",                 "data"),
            "失業率":                    ("UNEMPLOYMENT",        "data"),
            "FF金利（政策金利）":        ("FEDERAL_FUNDS_RATE",  "data"),
            "米国小売売上高":            ("RETAIL_SALES",        "data"),
            "消費者信頼感指数":          ("CONSUMER_CONFIDENCE", "data"),
        }

        ind_choice = st.selectbox("指標を選択", list(INDICATORS.keys()), key="av_ind")
        func, key_path = INDICATORS[ind_choice]

        with st.spinner(f"{ind_choice} 取得中..."):
            ind_data = _av_get(func, {"interval": "monthly" if func not in ["REAL_GDP"] else "annual"})

        series = ind_data.get(key_path, ind_data.get("data", []))
        if series:
            rows = []
            for item in (series[:36] if isinstance(series, list) else []):
                rows.append({
                    "日付": item.get("date", item.get("fiscalDateEnding", "")),
                    "値":   item.get("value", item.get("reportedEPS", "")),
                })
            if rows:
                df_ind = pd.DataFrame(rows)
                df_ind["日付"] = pd.to_datetime(df_ind["日付"], errors="coerce")
                df_ind["値"]   = pd.to_numeric(df_ind["値"], errors="coerce")
                df_ind = df_ind.dropna().sort_values("日付")

                fig_ind, ax_ind = plt.subplots(figsize=(12, 4))
                ax_ind.plot(df_ind["日付"], df_ind["値"],
                            color="#1565c0", linewidth=2, marker="o", markersize=3)
                ax_ind.fill_between(df_ind["日付"], df_ind["値"],
                                    df_ind["値"].min(), alpha=0.1, color="#1565c0")
                ax_ind.set_title(ind_choice, fontsize=12)
                ax_ind.grid(True, alpha=0.25)
                plt.xticks(rotation=45); plt.tight_layout()
                st.pyplot(fig_ind, clear_figure=True)

                latest_val = df_ind["値"].iloc[-1]
                prev_val   = df_ind["値"].iloc[-2] if len(df_ind) >= 2 else None
                col_i1, col_i2 = st.columns(2)
                col_i1.metric(
                    f"最新値（{df_ind['日付'].iloc[-1].strftime('%Y-%m')}）",
                    f"{latest_val:.2f}",
                    delta=f"{latest_val - prev_val:.2f}" if prev_val else None
                )
                col_i2.metric("直近12ヶ月平均", f"{df_ind['値'].tail(12).mean():.2f}")

                with st.expander("データ一覧"):
                    st.dataframe(df_ind.sort_values("日付", ascending=False),
                                 use_container_width=True, hide_index=True)
        else:
            st.warning("データ取得失敗（無料枠は1分5回・1日25回制限）")

    # ── Tab3: セクターパフォーマンス ─────────────────────────────
    with av_t3:
        st.markdown("#### 🏭 米国セクター別パフォーマンス")
        st.caption("S&P500の11セクター別リターン（Alpha Vantage提供）")

        with st.spinner("セクターデータ取得中..."):
            sec_data = _av_get("SECTOR")

        if sec_data:
            periods = {
                "1日": "Rank A: Real-Time Performance",
                "1週": "Rank B: 1 Day Performance",
                "1ヶ月": "Rank C: 5 Day Performance",
                "3ヶ月": "Rank D: 1 Month Performance",
                "1年": "Rank E: 3 Month Performance",
                "3年": "Rank F: Year-to-Date (YTD) Performance",
            }
            period_sel = st.selectbox("期間", list(periods.keys()), index=1, key="av_sec_period")
            pkey = periods[period_sel]

            sec_perf = sec_data.get(pkey, {})
            if sec_perf:
                rows = [{"セクター": k, "リターン(%)": float(v.strip("%"))}
                        for k, v in sec_perf.items() if v and v != "None"]
                if rows:
                    df_sec = pd.DataFrame(rows).sort_values("リターン(%)", ascending=True)
                    colors = ["#d1242f" if v < 0 else "#1a7f37" for v in df_sec["リターン(%)"]]
                    fig_sec, ax_sec = plt.subplots(figsize=(10, 6))
                    bars = ax_sec.barh(df_sec["セクター"], df_sec["リターン(%)"],
                                       color=colors, alpha=0.85)
                    for bar, val in zip(bars, df_sec["リターン(%)"]):
                        xpos = bar.get_width() + (0.05 if val >= 0 else -0.05)
                        ha = "left" if val >= 0 else "right"
                        ax_sec.text(xpos, bar.get_y() + bar.get_height()/2,
                                    f"{val:+.2f}%", va="center", ha=ha, fontsize=9)
                    ax_sec.axvline(0, color="black", linewidth=0.8)
                    ax_sec.set_title(f"US Sector Performance ({period_sel})", fontsize=12)
                    ax_sec.set_xlabel("Return (%)")
                    ax_sec.grid(True, axis="x", alpha=0.25)
                    plt.tight_layout()
                    st.pyplot(fig_sec, clear_figure=True)

                    # ベスト/ワースト
                    best   = df_sec.iloc[-1]
                    worst  = df_sec.iloc[0]
                    c1, c2 = st.columns(2)
                    c1.metric("🥇 最強セクター", best["セクター"],  f"{best['リターン(%)']:+.2f}%")
                    c2.metric("🥈 最弱セクター", worst["セクター"], f"{worst['リターン(%)']:+.2f}%")
            else:
                st.warning("セクターデータが取得できませんでした")
        else:
            st.warning("データ取得失敗（無料枠: 1分5回・1日25回制限）")



st.markdown("""<div style="background:#6a1b9a;color:white;padding:12px 22px;border-radius:8px;margin:28px 0 4px 0;font-size:18px;font-weight:bold;">
💰 C.&nbsp;ファンダメンタルズ分析 &nbsp;<span style="font-size:12px;font-weight:400;opacity:.88">来期業績スクリーニング / サイズ・バリューファクター / 価値創造（ROE・ROIC）</span></div>""", unsafe_allow_html=True)

# =================================================================
# 🔮 来期想定利益からのおすすめ銘柄スクリーニング
# =================================================================

# ─────────────────────────────────────────────────────────────────
st.header("🔮 来期想定利益スクリーニング")
st.caption(
    "yfinance（forward PER・EPS予想）× J-Quants（決算短信）を組み合わせて、"
    "**来期の利益成長が期待できる割安成長株**を抽出します。"
)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_forward_metrics(ticker: str) -> dict:
    """yfinanceからforward指標を取得"""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "ticker":            ticker,
            "trailing_per":      info.get("trailingPE"),
            "forward_per":       info.get("forwardPE"),
            "trailing_eps":      info.get("trailingEps"),
            "forward_eps":       info.get("forwardEps"),
            "peg_ratio":         info.get("pegRatio"),
            "revenue_growth":    info.get("revenueGrowth"),      # 売上成長率
            "earnings_growth":   info.get("earningsGrowth"),     # 利益成長率
            "operating_margins": info.get("operatingMargins"),   # 営業利益率
            "profit_margins":    info.get("profitMargins"),      # 純利益率
            "market_cap":        info.get("marketCap"),
            "price":             info.get("currentPrice") or info.get("regularMarketPrice"),
            "52w_high":          info.get("fiftyTwoWeekHigh"),
            "52w_low":           info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return {"ticker": ticker}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_jq_fins_summary(code: str) -> dict:
    """J-Quantsから最新の決算短信データを取得"""
    df = jq_fetch_fins(code)
    if df.empty:
        return {}
    try:
        latest = df.iloc[-1].to_dict()
        prev   = df.iloc[-2].to_dict() if len(df) >= 2 else {}
        return {"latest": latest, "prev": prev, "count": len(df)}
    except Exception:
        return {}


def _safe_float(val):
    try:
        v = float(val)
        return v if not (v != v) else None  # NaN check
    except Exception:
        return None


def _build_forward_df(df_bulk: pd.DataFrame) -> pd.DataFrame:
    """fetch_all_ticker_info_bulk の結果から来期スクリーニング用DataFrameを構築"""
    rows = []
    for _, row in df_bulk.iterrows():
        trailing_per = _safe_float(row.get("PER"))
        forward_per  = _safe_float(row.get("予想PER_raw"))
        trailing_eps = _safe_float(row.get("実績EPS_raw"))
        forward_eps  = _safe_float(row.get("予想EPS_raw"))
        peg          = _safe_float(row.get("PEGレシオ_raw"))
        rev_growth   = _safe_float(row.get("売上成長率_raw"))
        earn_growth  = _safe_float(row.get("利益成長率_raw"))
        op_margin    = _safe_float(row.get("営業利益率_raw"))
        price        = _safe_float(row.get("現在株価_raw"))

        if trailing_eps and forward_eps and trailing_eps != 0:
            eps_growth = (forward_eps - trailing_eps) / abs(trailing_eps) * 100
        elif earn_growth is not None:
            eps_growth = earn_growth * 100
        else:
            eps_growth = None

        if peg is None and forward_per and eps_growth and eps_growth > 0:
            peg = forward_per / eps_growth

        rows.append({
            "企業名":        row.get("企業名"),
            "業種":          row.get("業種"),
            "ティッカー":    row.get("ティッカー"),
            "現在株価":      round(price, 1) if price else None,
            "実績PER":       round(trailing_per, 1) if trailing_per else None,
            "予想PER":       round(forward_per, 1) if forward_per else None,
            "実績EPS":       round(trailing_eps, 2) if trailing_eps else None,
            "予想EPS":       round(forward_eps, 2) if forward_eps else None,
            "EPS成長率(%)":  round(eps_growth, 1) if eps_growth is not None else None,
            "売上成長率(%)": round(rev_growth * 100, 1) if rev_growth is not None else None,
            "営業利益率(%)": round(op_margin * 100, 1) if op_margin is not None else None,
            "PEGレシオ":     round(peg, 2) if peg is not None else None,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_ticker_info_bulk(ticker_map_items: tuple) -> pd.DataFrame:
    """全銘柄のyfinance情報を並列一括取得（factor・forward metrics統合版）。
    yf.Ticker().info の呼び出しを225回 → 1セットの並列処理に集約する。"""
    import concurrent.futures as _cfe

    def _get_one(item):
        ticker, (name, sector) = item
        try:
            info = yf.Ticker(ticker).info
            return {
                "企業名": name, "業種": sector, "ティッカー": ticker,
                "時価総額":       info.get("marketCap"),
                "PBR":            info.get("priceToBook"),
                "PER":            info.get("trailingPE"),
                "PSR":            info.get("priceToSalesTrailing12Months"),
                "配当利回り(%)":  round((info.get("dividendYield") or 0) * 100, 2),
                "ROE(%)":         round((info.get("returnOnEquity") or 0) * 100, 2),
                "予想PER_raw":    info.get("forwardPE"),
                "実績EPS_raw":    info.get("trailingEps"),
                "予想EPS_raw":    info.get("forwardEps"),
                "PEGレシオ_raw":  info.get("pegRatio"),
                "売上成長率_raw": info.get("revenueGrowth"),
                "利益成長率_raw": info.get("earningsGrowth"),
                "営業利益率_raw": info.get("operatingMargins"),
                "現在株価_raw":   info.get("currentPrice") or info.get("regularMarketPrice"),
                "totalDebt_raw":    info.get("totalDebt"),
                "totalCash_raw":    info.get("totalCash"),
                "totalRevenue_raw": info.get("totalRevenue"),
            }
        except Exception:
            return {"企業名": name, "業種": sector, "ティッカー": ticker}

    with _cfe.ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(_get_one, ticker_map_items))
    return pd.DataFrame(results)


# ── サイドバー設定 ──────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("### 🔮 来期スクリーニング設定")
    fwd_per_max    = st.slider("最大 Forward PER", 5, 60, 30, key="fwd_per_max")
    fwd_eps_growth = st.slider("最小 EPS成長率(%)", -50, 100, 10, key="fwd_eps_growth")
    fwd_peg_max    = st.slider("最大 PEGレシオ", 0.1, 5.0, 2.0, 0.1, key="fwd_peg_max")
    fwd_op_margin  = st.slider("最小 営業利益率(%)", 0, 40, 5, key="fwd_op_margin")
    fwd_top_n      = st.slider("表示銘柄数", 10, 50, 20, key="fwd_top_n")

st.info(
    f"📋 スクリーニング条件: "
    f"Forward PER ≤ {fwd_per_max} | "
    f"EPS成長率 ≥ {fwd_eps_growth}% | "
    f"PEGレシオ ≤ {fwd_peg_max} | "
    f"営業利益率 ≥ {fwd_op_margin}%"
)

# ── データ取得（fetch_all_ticker_info_bulk のキャッシュを流用）──
with st.spinner("来期指標を集計中（並列取得済みデータを使用）..."):
    _df_bulk_fwd = fetch_all_ticker_info_bulk(tuple(ticker_name_map.items()))
    df_fwd = _build_forward_df(_df_bulk_fwd)

if df_fwd.empty:
    st.error("データを取得できませんでした")
else:
    # ── スクリーニング ────────────────────────────────────────────
    df_screen_fwd = df_fwd.copy()
    if "予想PER" in df_screen_fwd.columns:
        df_screen_fwd = df_screen_fwd[
            df_screen_fwd["予想PER"].notna() &
            (df_screen_fwd["予想PER"] > 0) &
            (df_screen_fwd["予想PER"] <= fwd_per_max)
        ]
    if "EPS成長率(%)" in df_screen_fwd.columns:
        df_screen_fwd = df_screen_fwd[
            df_screen_fwd["EPS成長率(%)"].notna() &
            (df_screen_fwd["EPS成長率(%)"] >= fwd_eps_growth)
        ]
    if "PEGレシオ" in df_screen_fwd.columns:
        df_screen_fwd = df_screen_fwd[
            df_screen_fwd["PEGレシオ"].notna() &
            (df_screen_fwd["PEGレシオ"] <= fwd_peg_max) &
            (df_screen_fwd["PEGレシオ"] > 0)
        ]
    if "営業利益率(%)" in df_screen_fwd.columns:
        df_screen_fwd = df_screen_fwd[
            df_screen_fwd["営業利益率(%)"].notna() &
            (df_screen_fwd["営業利益率(%)"] >= fwd_op_margin)
        ]

    # PEGレシオでソート（低いほど割安成長）
    df_screen_fwd = df_screen_fwd.sort_values("PEGレシオ", ascending=True).reset_index(drop=True)

    # ── サマリーメトリクス ────────────────────────────────────────
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("スクリーニング通過", f"{len(df_screen_fwd)}銘柄",
               f"全{len(df_fwd[df_fwd['予想PER'].notna()])}銘柄中")
    if not df_screen_fwd.empty:
        sm2.metric("平均 予想PER",
                   f"{df_screen_fwd['予想PER'].mean():.1f}倍")
        sm3.metric("平均 EPS成長率",
                   f"{df_screen_fwd['EPS成長率(%)'].mean():.1f}%")
        sm4.metric("最小 PEGレシオ（割安）",
                   f"{df_screen_fwd['PEGレシオ'].min():.2f}",
                   df_screen_fwd.iloc[0]["企業名"])

    st.divider()

    fwd_t1, fwd_t2, fwd_t3, fwd_t4 = st.tabs([
        "💎 割安成長株ランキング（PEG）",
        "📊 PER比較・EPS成長",
        "📈 チャート分析",
        "🤖 AI銘柄コメント",
    ])

    # ── Tab1: PEGランキング ──────────────────────────────────────
    with fwd_t1:
        st.markdown("#### 💎 PEGレシオ順 割安成長株ランキング")
        st.caption(
            "PEGレシオ = 予想PER ÷ EPS成長率。**1以下が割安成長株の目安**。"
            "低いほど「成長に対して株価が安い」銘柄。"
        )

        if df_screen_fwd.empty:
            st.warning("条件を満たす銘柄がありません。サイドバーの条件を緩めてください。")
        else:
            disp_cols = ["企業名", "業種", "PEGレシオ", "予想PER", "実績PER",
                         "EPS成長率(%)", "売上成長率(%)", "営業利益率(%)", "現在株価"]

            def _color_peg(val):
                if isinstance(val, float):
                    if val < 1.0:   return "color:#1a7f37;font-weight:bold;font-size:14px"
                    elif val < 1.5: return "color:#1a7f37;font-weight:bold"
                    elif val < 2.0: return "color:#f57c00"
                    else:           return "color:#d1242f"
                return ""

            def _color_growth(val):
                if isinstance(val, float):
                    if val >= 30:   return "color:#1a7f37;font-weight:bold"
                    elif val >= 10: return "color:#388e3c"
                    elif val < 0:   return "color:#d1242f"
                return ""

            st.dataframe(
                df_screen_fwd[disp_cols].head(fwd_top_n)
                .style.format({
                    "PEGレシオ":     "{:.2f}",
                    "予想PER":       "{:.1f}倍",
                    "実績PER":       "{:.1f}倍",
                    "EPS成長率(%)":  "{:+.1f}%",
                    "売上成長率(%)": "{:+.1f}%",
                    "営業利益率(%)": "{:.1f}%",
                    "現在株価":      "{:,.0f}円",
                }, na_rep="N/A")
                .map(_color_peg,    subset=["PEGレシオ"])
                .map(_color_growth, subset=["EPS成長率(%)"]),
                use_container_width=True, hide_index=True
            )

            # CSV DL
            csv_fwd = df_screen_fwd[disp_cols].to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "⬇️ スクリーニング結果CSV",
                data=csv_fwd,
                file_name=f"fwd_screen_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv", key="fwd_dl"
            )

    # ── Tab2: PER比較・EPS成長 ───────────────────────────────────
    with fwd_t2:
        st.markdown("#### 📊 実績PER vs 予想PER 比較 + EPS成長率")

        if df_screen_fwd.empty:
            st.warning("条件を満たす銘柄がありません")
        else:
            top20 = df_screen_fwd.head(20)

            fig_per, axes_per = plt.subplots(1, 2, figsize=(16, 6))

            # PER比較バー
            ax_per = axes_per[0]
            x_pos = np.arange(len(top20))
            w = 0.35
            ax_per.bar(x_pos - w/2, top20["実績PER"].fillna(0),
                       width=w, label="実績PER", color="#90caf9", alpha=0.85)
            ax_per.bar(x_pos + w/2, top20["予想PER"].fillna(0),
                       width=w, label="予想PER", color="#1565c0", alpha=0.85)
            ax_per.set_xticks(x_pos)
            ax_per.set_xticklabels(top20["企業名"], rotation=45,
                                   ha="right", fontsize=9)
            ax_per.set_ylabel("PER（倍）")
            ax_per.set_title("実績PER vs 予想PER（上位20銘柄）",
                              fontsize=12, fontweight="bold")
            ax_per.legend(fontsize=10)
            ax_per.grid(True, axis="y", alpha=0.3)
            ax_per.axhline(fwd_per_max, color="red", linestyle="--",
                           alpha=0.5, label=f"PER上限({fwd_per_max})")

            # EPS成長率バー
            ax_eps = axes_per[1]
            colors_eps = ["#1a7f37" if v >= 0 else "#d1242f"
                          for v in top20["EPS成長率(%)"].fillna(0)]
            ax_eps.bar(top20["企業名"], top20["EPS成長率(%)"].fillna(0),
                       color=colors_eps, alpha=0.85)
            ax_eps.axhline(0, color="black", linewidth=0.8)
            ax_eps.axhline(fwd_eps_growth, color="orange", linestyle="--",
                           alpha=0.6, label=f"最低成長率({fwd_eps_growth}%)")
            ax_eps.set_xticklabels(top20["企業名"], rotation=45,
                                   ha="right", fontsize=9)
            ax_eps.set_ylabel("EPS成長率 (%)")
            ax_eps.set_title("EPS成長率（予想 vs 実績）",
                              fontsize=12, fontweight="bold")
            ax_eps.legend(fontsize=9)
            ax_eps.grid(True, axis="y", alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig_per, clear_figure=True)

            # 営業利益率ランキング
            st.markdown("#### 🏭 営業利益率ランキング（スクリーニング通過銘柄）")
            df_margin = df_screen_fwd.dropna(subset=["営業利益率(%)"])\
                                     .sort_values("営業利益率(%)", ascending=False)\
                                     .head(20)
            fig_mg, ax_mg = plt.subplots(figsize=(12, 5))
            colors_mg = ["#1565c0" if v >= 15 else "#42a5f5"
                         for v in df_margin["営業利益率(%)"]]
            ax_mg.barh(df_margin["企業名"][::-1],
                       df_margin["営業利益率(%)"][::-1],
                       color=colors_mg[::-1], alpha=0.85)
            ax_mg.axvline(fwd_op_margin, color="red", linestyle="--",
                          alpha=0.5, label=f"最低利益率({fwd_op_margin}%)")
            ax_mg.axvline(15, color="green", linestyle=":",
                          alpha=0.5, label="優良水準(15%)")
            ax_mg.set_xlabel("営業利益率 (%)")
            ax_mg.set_title("営業利益率ランキング", fontsize=12, fontweight="bold")
            ax_mg.legend(fontsize=9)
            ax_mg.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_mg, clear_figure=True)

    # ── Tab3: チャート分析 ───────────────────────────────────────
    with fwd_t3:
        st.markdown("#### 📈 PEGレシオ vs EPS成長率 散布図")
        st.caption("右上ほど高成長・左下ほど割安。バブルサイズ = 営業利益率")

        if df_screen_fwd.empty:
            st.warning("条件を満たす銘柄がありません")
        else:
            df_plot = df_screen_fwd.dropna(
                subset=["PEGレシオ", "EPS成長率(%)", "予想PER"]
            ).head(40)

            if not df_plot.empty:
                fig_sc2, ax_sc2 = plt.subplots(figsize=(14, 8))
                sectors_p = df_plot["業種"].unique()
                cmap_p = plt.colormaps["tab20"].resampled(len(sectors_p))
                sec_c_p = {s: cmap_p(i) for i, s in enumerate(sectors_p)}

                for sec in sectors_p:
                    sub = df_plot[df_plot["業種"] == sec]
                    size = (sub["営業利益率(%)"].fillna(5).clip(lower=1) * 15)
                    ax_sc2.scatter(
                        sub["EPS成長率(%)"],
                        sub["予想PER"],
                        s=size,
                        color=sec_c_p[sec],
                        label=sec, alpha=0.8,
                        edgecolors="gray", linewidth=0.5, zorder=3
                    )
                    for _, row in sub.iterrows():
                        ax_sc2.annotate(
                            row["企業名"],
                            (row["EPS成長率(%)"], row["予想PER"]),
                            fontsize=7, alpha=0.9,
                            xytext=(4, 4), textcoords="offset points"
                        )

                # PEG=1の等高線
                x_range = np.linspace(
                    df_plot["EPS成長率(%)"].min(),
                    df_plot["EPS成長率(%)"].max(), 100
                )
                ax_sc2.plot(x_range, x_range * 1.0, "g--",
                            alpha=0.5, linewidth=1.5, label="PEG=1（割安ライン）")
                ax_sc2.plot(x_range, x_range * 2.0, "r--",
                            alpha=0.5, linewidth=1.5, label="PEG=2（割高ライン）")

                ax_sc2.set_xlabel("EPS成長率 (%)", fontsize=12)
                ax_sc2.set_ylabel("予想PER（倍）", fontsize=12)
                ax_sc2.set_title(
                    "EPS成長率 vs 予想PER マップ\n"
                    "（バブルサイズ=営業利益率 / 緑点線=PEG1 / 赤点線=PEG2）",
                    fontsize=12, fontweight="bold"
                )
                ax_sc2.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
                              fontsize=8, framealpha=0.9)
                ax_sc2.grid(True, alpha=0.2)
                ax_sc2.spines["top"].set_visible(False)
                ax_sc2.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_sc2, clear_figure=True)

    # ── Tab4: AI銘柄コメント ─────────────────────────────────────
    with fwd_t4:
        st.markdown("#### 🤖 AI来期おすすめ銘柄コメント")
        st.caption("スクリーニング通過上位銘柄をAIが総合評価します")

        if df_screen_fwd.empty:
            st.warning("条件を満たす銘柄がありません")
        else:
            top_ai = df_screen_fwd.head(10)
            summary_str = top_ai[[
                "企業名", "業種", "PEGレシオ", "予想PER",
                "EPS成長率(%)", "売上成長率(%)", "営業利益率(%)"
            ]].to_string(index=False)

            prompt_fwd = f"""
以下は日本株の来期想定利益スクリーニング結果（上位10銘柄）です。

{summary_str}

スクリーニング条件:
- 予想PER ≤ {fwd_per_max}倍
- EPS成長率 ≥ {fwd_eps_growth}%
- PEGレシオ ≤ {fwd_peg_max}
- 営業利益率 ≥ {fwd_op_margin}%

投資家向けに以下の観点で分析してください（600文字以内）:
1. 🏆 特に注目すべき銘柄とその理由（PEG・成長率・利益率の観点）
2. 📊 業種・セクターの傾向（どの業種に割安成長株が集まっているか）
3. ⚠️ 注意点・リスク（PERや成長率の信頼性など）
4. 💡 総合的な投資戦略（時期・分散など）

※ 投資判断は自己責任である旨を最後に一言付記してください。
"""
            with st.spinner("AI分析中..."):
                try:
                    ai_comment, ai_name = generate_ai_comment(prompt_fwd)
                    st.info(f"🤖 **AI来期銘柄分析（{ai_name}）**\n\n{ai_comment}")
                except Exception as e:
                    st.warning(f"AI APIエラー: {e}")

            # 個別銘柄の簡易コメント
            st.markdown("---")
            st.markdown("#### 📋 上位銘柄サマリー")
            for _, row in top_ai.head(5).iterrows():
                peg_val  = row.get("PEGレシオ", None)
                per_val  = row.get("予想PER", None)
                eps_val  = row.get("EPS成長率(%)", None)
                op_val   = row.get("営業利益率(%)", None)

                peg_str  = f"{peg_val:.2f}" if peg_val else "N/A"
                per_str  = f"{per_val:.1f}倍" if per_val else "N/A"
                eps_str  = f"{eps_val:+.1f}%" if eps_val else "N/A"
                op_str   = f"{op_val:.1f}%" if op_val else "N/A"

                # PEG評価
                if peg_val and peg_val < 1.0:
                    eval_icon = "💎 極めて割安"
                    eval_color = "#1a7f37"
                elif peg_val and peg_val < 1.5:
                    eval_icon = "✅ 割安成長"
                    eval_color = "#388e3c"
                else:
                    eval_icon = "📊 適正水準"
                    eval_color = "#1565c0"

                st.markdown(
                    f'<div style="background:#f8f9fa;border-left:4px solid {eval_color};'
                    f'border-radius:6px;padding:12px 16px;margin:8px 0;">'
                    f'<b style="font-size:15px;color:{eval_color};">'
                    f'{eval_icon} {row["企業名"]}（{row["業種"]}）</b><br>'
                    f'<span style="font-size:13px;color:#555;">'
                    f'PEGレシオ: <b>{peg_str}</b> | '
                    f'予想PER: <b>{per_str}</b> | '
                    f'EPS成長率: <b>{eps_str}</b> | '
                    f'営業利益率: <b>{op_str}</b>'
                    f'</span></div>',
                    unsafe_allow_html=True
                )


# =================================================================
# 📏 サイズファクター分析  /  💰 バリューファクター分析
# =================================================================

from datetime import timedelta as _fac_td


def _size_label(mc):
    if mc is None or (isinstance(mc, float) and mc != mc):
        return "不明"
    if mc >= 1_000_000_000_000:
        return "大型株(>1兆)"
    elif mc >= 100_000_000_000:
        return "中型株(1000億-1兆)"
    return "小型株(<1000億)"


def _value_label(pbr):
    if pbr is None or (isinstance(pbr, float) and pbr != pbr):
        return "不明"
    if pbr < 1.0:   return "割安(PBR<1)"
    if pbr < 2.0:   return "適正(PBR 1-2)"
    if pbr < 3.0:   return "やや割高(PBR 2-3)"
    return "割高(PBR>3)"


def _port_returns(tickers, start, end, max_n=20):
    """等重みポートフォリオの日次リターンを返す"""
    rets = []
    for t in tickers[:max_n]:
        df_t = _yfdownload(t, start=start, end=end, progress=False)
        if df_t.empty or len(df_t) < 10:
            continue
        r = _to_series(df_t["Close"]).pct_change().dropna()
        rets.append(r)
    if not rets:
        return pd.Series(dtype=float)
    return pd.concat(rets, axis=1).mean(axis=1)


# ─── ファクターデータ（fetch_all_ticker_info_bulk のキャッシュを流用）────
df_factor = fetch_all_ticker_info_bulk(tuple(ticker_name_map.items()))

df_factor["サイズ分類"]    = df_factor["時価総額"].apply(_size_label)
df_factor["バリュー分類"]  = df_factor["PBR"].apply(_value_label)
df_factor["時価総額(億円)"] = (df_factor["時価総額"].fillna(0) / 1e8).round(1)

try:
    _has_results = not df_results.empty
except Exception:
    _has_results = False

if _has_results:
    _perf_cols = ["企業名", "年間平均リターン(%)", "年間リスク(%)",
                  "シャープレシオ", "アルファ(%)", "ベータ"]
    df_factor_merged = df_factor.merge(df_results[_perf_cols], on="企業名", how="inner")
else:
    df_factor_merged = df_factor.copy()

_SIZE_ORDER  = ["大型株(>1兆)", "中型株(1000億-1兆)", "小型株(<1000億)"]
_VALUE_ORDER = ["割安(PBR<1)", "適正(PBR 1-2)", "やや割高(PBR 2-3)", "割高(PBR>3)"]


# =================================================================
# 📏 サイズファクター分析
# =================================================================

st.header("📏 サイズファクター分析")
st.divider()
st.caption(
    "時価総額で銘柄を **大型・中型・小型** に分類し、リターン・リスク・シャープレシオを比較します。"
    "**SMBファクター**（Small Minus Big）で小型株プレミアムを検証します。"
)

sz_t1, sz_t2, sz_t3, sz_t4 = st.tabs([
    "📊 時価総額分布",
    "📈 サイズ別パフォーマンス",
    "📉 SMBファクター",
    "🗺️ 時価総額×アルファ",
])

with sz_t1:
    df_mc_v = df_factor.dropna(subset=["時価総額"])
    sc = df_mc_v["サイズ分類"].value_counts()
    cols_sc = st.columns(3)
    for i, (lbl, col) in enumerate(zip(_SIZE_ORDER, cols_sc)):
        col.metric(lbl, f"{int(sc.get(lbl, 0))}銘柄")

    col_pie, col_hist = st.columns(2)
    with col_pie:
        labels_p = [l for l in _SIZE_ORDER if l in sc.index]
        fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
        ax_pie.pie([sc[l] for l in labels_p], labels=labels_p,
                   colors=["#1565c0", "#42a5f5", "#90caf9"],
                   autopct="%1.1f%%", startangle=90,
                   wedgeprops=dict(edgecolor="white", linewidth=1.5))
        ax_pie.set_title("サイズ分類の割合", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_pie, clear_figure=True)

    with col_hist:
        mc_log = np.log10(df_mc_v[df_mc_v["時価総額(億円)"] > 0]["時価総額(億円)"] + 1)
        fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
        ax_hist.hist(mc_log, bins=30, color="#1565c0", alpha=0.75, edgecolor="white")
        ax_hist.axvline(np.log10(1000), color="#f57c00", linestyle="--",
                        alpha=0.8, label="1000億円")
        ax_hist.axvline(np.log10(10000), color="#d1242f", linestyle="--",
                        alpha=0.8, label="1兆円")
        ax_hist.set_xlabel("log10(時価総額・億円)", fontsize=10)
        ax_hist.set_ylabel("銘柄数", fontsize=10)
        ax_hist.set_title("時価総額分布（対数）", fontsize=12, fontweight="bold")
        ax_hist.legend(fontsize=8); ax_hist.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig_hist, clear_figure=True)

    col_top20, col_bot20 = st.columns(2)
    with col_top20:
        st.markdown("**時価総額 上位20銘柄**")
        st.dataframe(
            df_mc_v.nlargest(20, "時価総額")[
                ["企業名", "業種", "サイズ分類", "時価総額(億円)"]
            ].reset_index(drop=True)
            .style.format({"時価総額(億円)": "{:,.0f}"}),
            use_container_width=True, hide_index=True,
        )
    with col_bot20:
        st.markdown("**時価総額 下位20銘柄**")
        st.dataframe(
            df_mc_v.nsmallest(20, "時価総額")[
                ["企業名", "業種", "サイズ分類", "時価総額(億円)"]
            ].reset_index(drop=True)
            .style.format({"時価総額(億円)": "{:,.0f}"}),
            use_container_width=True, hide_index=True,
        )

with sz_t2:
    if not _has_results:
        st.info("上部の「パフォーマンス分析」が自動実行後に表示されます")
    else:
        df_sz_p = df_factor_merged.dropna(subset=["シャープレシオ", "サイズ分類"])
        sz_agg = df_sz_p.groupby("サイズ分類").agg(
            銘柄数=("企業名", "count"),
            平均リターン=("年間平均リターン(%)", "mean"),
            平均リスク=("年間リスク(%)", "mean"),
            平均シャープレシオ=("シャープレシオ", "mean"),
            平均アルファ=("アルファ(%)", "mean"),
        ).round(3).reset_index()
        sz_agg["_o"] = sz_agg["サイズ分類"].map({s: i for i, s in enumerate(_SIZE_ORDER)})
        sz_agg = sz_agg.sort_values("_o").drop("_o", axis=1)

        cols_m = st.columns(len(sz_agg))
        for i, (_, row) in enumerate(sz_agg.iterrows()):
            cols_m[i].metric(row["サイズ分類"],
                             f"SR: {row['平均シャープレシオ']:.2f}",
                             f"α: {row['平均アルファ']:+.2f}%")

        st.dataframe(sz_agg.style.format({
            "平均リターン": "{:+.2f}%", "平均リスク": "{:.2f}%",
            "平均シャープレシオ": "{:.3f}", "平均アルファ": "{:+.2f}%",
        }), use_container_width=True, hide_index=True)

        fig_sz, axes_sz = plt.subplots(1, 3, figsize=(15, 5))
        for ax, metric, title in zip(
            axes_sz,
            ["平均シャープレシオ", "平均リターン", "平均アルファ"],
            ["平均シャープレシオ", "平均リターン(%)", "平均アルファ(%)"],
        ):
            vld = sz_agg.dropna(subset=[metric])
            ax.bar(vld["サイズ分類"], vld[metric],
                   color=["#1a7f37" if v >= 0 else "#d1242f" for v in vld[metric]],
                   alpha=0.85)
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.tick_params(axis="x", rotation=15, labelsize=9)
            ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_sz, clear_figure=True)

        box_sz = [df_sz_p[df_sz_p["サイズ分類"] == s]["シャープレシオ"].dropna().values
                  for s in _SIZE_ORDER if s in df_sz_p["サイズ分類"].values]
        box_sz_lbl = [s for s in _SIZE_ORDER if s in df_sz_p["サイズ分類"].values]
        if any(len(d) > 0 for d in box_sz):
            fig_bx, ax_bx = plt.subplots(figsize=(9, 5))
            bp = ax_bx.boxplot([d for d in box_sz if len(d) > 0],
                               tick_labels=[l for l, d in zip(box_sz_lbl, box_sz) if len(d) > 0],
                               patch_artist=True,
                               medianprops=dict(color="black", linewidth=2))
            for patch, color in zip(bp["boxes"], ["#1565c0", "#42a5f5", "#90caf9"]):
                patch.set_facecolor(color); patch.set_alpha(0.7)
            ax_bx.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax_bx.set_ylabel("シャープレシオ", fontsize=12)
            ax_bx.set_title("サイズ別シャープレシオ分布", fontsize=12, fontweight="bold")
            ax_bx.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_bx, clear_figure=True)

with sz_t3:
    smb_days  = st.slider("分析期間（日）", 30, 250, 120, key="smb_days")
    end_smb   = datetime.today()
    start_smb = end_smb - _fac_td(days=smb_days + 10)

    df_mc_s = df_factor.dropna(subset=["時価総額"]).sort_values("時価総額")
    n_s = max(int(len(df_mc_s) * 0.3), 5)

    with st.spinner("SMBファクター計算中..."):
        small_ret = _port_returns(df_mc_s.head(n_s)["ティッカー"].tolist(), start_smb, end_smb)
        large_ret = _port_returns(df_mc_s.tail(n_s)["ティッカー"].tolist(), start_smb, end_smb)

    if small_ret.empty or large_ret.empty:
        st.warning("SMBデータ取得に失敗しました")
    else:
        idx_s   = small_ret.index.intersection(large_ret.index)
        smb_d   = small_ret.loc[idx_s] - large_ret.loc[idx_s]
        smb_cum = (1 + smb_d).cumprod() - 1
        s_cum   = (1 + small_ret.loc[idx_s]).cumprod() - 1
        l_cum   = (1 + large_ret.loc[idx_s]).cumprod() - 1

        c1, c2, c3 = st.columns(3)
        c1.metric("小型株 累積リターン",  f"{s_cum.iloc[-1]*100:+.2f}%")
        c2.metric("大型株 累積リターン",  f"{l_cum.iloc[-1]*100:+.2f}%")
        c3.metric("SMBスプレッド", f"{smb_cum.iloc[-1]*100:+.2f}%",
                  "小型株プレミアム" if smb_cum.iloc[-1] > 0 else "大型株優位")

        fig_smb, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                            gridspec_kw={"height_ratios": [2, 1]})
        ax1.plot(s_cum.index, s_cum * 100, label="小型株", color="#1a7f37", linewidth=2)
        ax1.plot(l_cum.index, l_cum * 100, label="大型株", color="#1565c0", linewidth=2)
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax1.set_ylabel("累積リターン (%)"); ax1.legend(fontsize=10)
        ax1.set_title("小型株 vs 大型株 累積リターン", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.2); ax1.tick_params(axis="x", rotation=30)
        ax2.bar(smb_cum.index, smb_cum * 100,
                color=["#1a7f37" if v >= 0 else "#d1242f" for v in smb_cum],
                alpha=0.7, width=1.5)
        ax2.axhline(0, color="gray", linewidth=0.7)
        ax2.set_ylabel("SMB (%)"); ax2.set_title("SMBファクター", fontsize=11)
        ax2.grid(True, alpha=0.2); ax2.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig_smb, clear_figure=True)

        prompt_smb = (
            f"日本株の過去{smb_days}日間SMBファクター: "
            f"小型株{s_cum.iloc[-1]*100:+.2f}%、大型株{l_cum.iloc[-1]*100:+.2f}%、"
            f"スプレッド{smb_cum.iloc[-1]*100:+.2f}%。"
            "投資家向け200文字以内: 1)サイズプレミアムの現状 2)投資戦略への示唆"
        )
        with st.spinner("AI解説中..."):
            try:
                comment, ai_name = generate_ai_comment(prompt_smb)
                st.info(f"🤖 **AI解説（{ai_name}）**\n\n{comment}")
            except Exception as e:
                st.warning(f"AI APIエラー: {e}")

with sz_t4:
    if not _has_results:
        st.info("上部の「パフォーマンス分析」が自動実行後に表示されます")
    else:
        df_sc = df_factor_merged.dropna(subset=["時価総額", "アルファ(%)"]).copy()
        df_sc = df_sc[df_sc["時価総額(億円)"] > 0]
        df_sc["log時価総額"] = np.log10(df_sc["時価総額(億円)"])
        try:
            import plotly.express as px
            fig_szsc = px.scatter(
                df_sc, x="log時価総額", y="アルファ(%)",
                color="サイズ分類",
                size=df_sc["シャープレシオ"].clip(lower=0.1),
                hover_name="企業名",
                hover_data={"業種": True, "時価総額(億円)": ":.0f",
                            "アルファ(%)": ":.2f", "シャープレシオ": ":.2f"},
                title="時価総額 vs アルファ（バブルサイズ=シャープレシオ）",
                height=600,
                color_discrete_map={
                    "大型株(>1兆)": "#1565c0",
                    "中型株(1000億-1兆)": "#42a5f5",
                    "小型株(<1000億)": "#90caf9",
                },
            )
            fig_szsc.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_szsc.update_layout(xaxis_title="log10(時価総額・億円)",
                                   yaxis_title="アルファ (%)", plot_bgcolor="white")
            st.plotly_chart(fig_szsc, use_container_width=True)
        except ImportError:
            fig_szsc2, ax_sz = plt.subplots(figsize=(12, 7))
            for size_cat, color in zip(_SIZE_ORDER, ["#1565c0", "#42a5f5", "#90caf9"]):
                sub = df_sc[df_sc["サイズ分類"] == size_cat]
                ax_sz.scatter(sub["log時価総額"], sub["アルファ(%)"],
                              label=size_cat, color=color, s=60, alpha=0.75)
            ax_sz.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax_sz.set_xlabel("log10(時価総額・億円)", fontsize=12)
            ax_sz.set_ylabel("アルファ (%)", fontsize=12)
            ax_sz.set_title("時価総額 vs アルファ", fontsize=12, fontweight="bold")
            ax_sz.legend(fontsize=10); ax_sz.grid(True, alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig_szsc2, clear_figure=True)


# =================================================================
# 💰 バリューファクター分析
# =================================================================

st.header("💰 バリューファクター分析")
st.divider()
st.caption(
    "PBR（株価純資産倍率）・PER・PSR・ROEで **割安株（バリュー）と割高株（グロース）** を分類・比較します。"
    "**HMLファクター**（High Minus Low PBR）でバリュープレミアムを検証します。"
)

vl_t1, vl_t2, vl_t3, vl_t4 = st.tabs([
    "💎 低PBR スクリーニング",
    "📊 バリュー別パフォーマンス",
    "📉 HMLファクター",
    "🗺️ PBR×アルファ散布図",
])

with vl_t1:
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        pbr_max_vl = st.slider("最大PBR", 0.3, 5.0, 1.5, 0.1, key="vl_pbr_max")
    with col_v2:
        per_max_vl = st.slider("最大PER", 5, 80, 30, key="vl_per_max")

    df_vl_s = df_factor.dropna(subset=["PBR", "PER"])
    df_vl_s = df_vl_s[
        (df_vl_s["PBR"] > 0) & (df_vl_s["PBR"] <= pbr_max_vl) &
        (df_vl_s["PER"] > 0) & (df_vl_s["PER"] <= per_max_vl)
    ].sort_values("PBR").reset_index(drop=True)

    st.markdown(f"**{len(df_vl_s)}銘柄** が条件を満たしています"
                f"（PBR≤{pbr_max_vl} & PER≤{per_max_vl}）")

    if not df_vl_s.empty:
        disp_vl = [c for c in
                   ["企業名", "業種", "サイズ分類", "PBR", "PER", "PSR",
                    "配当利回り(%)", "ROE(%)"]
                   if c in df_vl_s.columns]

        def _color_pbr(val):
            if isinstance(val, float):
                if val < 0.5: return "color:#1a7f37;font-weight:bold;font-size:14px"
                if val < 1.0: return "color:#1a7f37;font-weight:bold"
                if val < 1.5: return "color:#388e3c"
            return ""

        st.dataframe(
            df_vl_s[disp_vl].head(40)
            .style.format({
                "PBR": "{:.2f}", "PER": "{:.1f}", "PSR": "{:.2f}",
                "配当利回り(%)": "{:.2f}%", "ROE(%)": "{:.1f}%",
            }, na_rep="N/A")
            .map(_color_pbr, subset=["PBR"]),
            use_container_width=True, hide_index=True,
        )
        csv_vl = df_vl_s[disp_vl].to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "⬇️ CSVダウンロード", data=csv_vl,
            file_name=f"value_screen_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv", key="vl_dl",
        )

    df_pbr_all = df_factor.dropna(subset=["PBR"])
    df_pbr_all = df_pbr_all[df_pbr_all["PBR"].between(0, 10)]
    fig_pbr_d, ax_pd = plt.subplots(figsize=(10, 4))
    ax_pd.hist(df_pbr_all["PBR"], bins=40, color="#1565c0", alpha=0.75, edgecolor="white")
    ax_pd.axvline(1.0, color="#1a7f37", linestyle="--", linewidth=1.5, label="PBR=1.0")
    ax_pd.axvline(pbr_max_vl, color="#f57c00", linestyle="--",
                  linewidth=1.5, label=f"上限({pbr_max_vl})")
    ax_pd.set_xlabel("PBR", fontsize=11); ax_pd.set_ylabel("銘柄数", fontsize=11)
    ax_pd.set_title("PBR分布（全銘柄）", fontsize=12, fontweight="bold")
    ax_pd.legend(fontsize=9); ax_pd.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig_pbr_d, clear_figure=True)

with vl_t2:
    if not _has_results:
        st.info("上部の「パフォーマンス分析」が自動実行後に表示されます")
    else:
        df_vp = df_factor_merged.dropna(subset=["PBR", "シャープレシオ"]).copy()
        df_vp["バリュー分類"] = df_vp["PBR"].apply(_value_label)

        vl_agg = df_vp.groupby("バリュー分類").agg(
            銘柄数=("企業名", "count"),
            平均PBR=("PBR", "mean"),
            平均リターン=("年間平均リターン(%)", "mean"),
            平均リスク=("年間リスク(%)", "mean"),
            平均シャープレシオ=("シャープレシオ", "mean"),
            平均アルファ=("アルファ(%)", "mean"),
        ).round(3).reset_index()
        vl_agg["_o"] = vl_agg["バリュー分類"].map(
            {v: i for i, v in enumerate(_VALUE_ORDER)}
        )
        vl_agg = vl_agg.sort_values("_o").drop("_o", axis=1)

        st.dataframe(vl_agg.style.format({
            "平均PBR": "{:.2f}", "平均リターン": "{:+.2f}%",
            "平均リスク": "{:.2f}%", "平均シャープレシオ": "{:.3f}",
            "平均アルファ": "{:+.2f}%",
        }), use_container_width=True, hide_index=True)

        fig_vp, axes_vp = plt.subplots(1, 3, figsize=(15, 5))
        for ax_vp, metric, title in zip(
            axes_vp,
            ["平均シャープレシオ", "平均リターン", "平均アルファ"],
            ["平均シャープレシオ", "平均リターン(%)", "平均アルファ(%)"],
        ):
            vld_vp = vl_agg.dropna(subset=[metric])
            ax_vp.bar(vld_vp["バリュー分類"], vld_vp[metric],
                      color=["#1a7f37" if v >= 0 else "#d1242f" for v in vld_vp[metric]],
                      alpha=0.85)
            ax_vp.axhline(0, color="gray", linewidth=0.8)
            ax_vp.set_title(title, fontsize=11, fontweight="bold")
            ax_vp.tick_params(axis="x", rotation=15, labelsize=8)
            ax_vp.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_vp, clear_figure=True)

        box_vl = [df_vp[df_vp["バリュー分類"] == v]["シャープレシオ"].dropna().values
                  for v in _VALUE_ORDER if v in df_vp["バリュー分類"].values]
        box_vl_lbl = [v for v in _VALUE_ORDER if v in df_vp["バリュー分類"].values]
        if any(len(d) > 0 for d in box_vl):
            fig_vbx, ax_vbx = plt.subplots(figsize=(9, 5))
            bpv = ax_vbx.boxplot(
                [d for d in box_vl if len(d) > 0],
                tick_labels=[l for l, d in zip(box_vl_lbl, box_vl) if len(d) > 0],
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
            )
            for patch, color in zip(bpv["boxes"],
                                    ["#1a7f37", "#42a5f5", "#f57c00", "#d1242f"]):
                patch.set_facecolor(color); patch.set_alpha(0.65)
            ax_vbx.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax_vbx.set_ylabel("シャープレシオ", fontsize=12)
            ax_vbx.set_title("バリュー分類別シャープレシオ分布",
                              fontsize=12, fontweight="bold")
            ax_vbx.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_vbx, clear_figure=True)

with vl_t3:
    hml_days  = st.slider("分析期間（日）", 30, 250, 120, key="hml_days")
    end_hml   = datetime.today()
    start_hml = end_hml - _fac_td(days=hml_days + 10)

    df_pbr_s = df_factor.dropna(subset=["PBR"])
    df_pbr_s = df_pbr_s[df_pbr_s["PBR"] > 0].sort_values("PBR")
    n_h = max(int(len(df_pbr_s) * 0.3), 5)

    with st.spinner("HMLファクター計算中..."):
        low_ret  = _port_returns(df_pbr_s.head(n_h)["ティッカー"].tolist(), start_hml, end_hml)
        high_ret = _port_returns(df_pbr_s.tail(n_h)["ティッカー"].tolist(), start_hml, end_hml)

    if low_ret.empty or high_ret.empty:
        st.warning("HMLデータ取得に失敗しました")
    else:
        idx_h   = low_ret.index.intersection(high_ret.index)
        hml_d   = low_ret.loc[idx_h] - high_ret.loc[idx_h]
        hml_cum = (1 + hml_d).cumprod() - 1
        l_cum_h = (1 + low_ret.loc[idx_h]).cumprod() - 1
        h_cum_h = (1 + high_ret.loc[idx_h]).cumprod() - 1

        c1, c2, c3 = st.columns(3)
        c1.metric("低PBR（割安）累積リターン", f"{l_cum_h.iloc[-1]*100:+.2f}%")
        c2.metric("高PBR（割高）累積リターン", f"{h_cum_h.iloc[-1]*100:+.2f}%")
        c3.metric("HMLスプレッド", f"{hml_cum.iloc[-1]*100:+.2f}%",
                  "バリュープレミアム" if hml_cum.iloc[-1] > 0 else "グロース優位")

        fig_hml, (ax_h1, ax_h2) = plt.subplots(2, 1, figsize=(12, 8),
                                                gridspec_kw={"height_ratios": [2, 1]})
        ax_h1.plot(l_cum_h.index, l_cum_h * 100,
                   label="低PBR（バリュー）", color="#1a7f37", linewidth=2)
        ax_h1.plot(h_cum_h.index, h_cum_h * 100,
                   label="高PBR（グロース）", color="#d1242f", linewidth=2)
        ax_h1.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax_h1.set_ylabel("累積リターン (%)"); ax_h1.legend(fontsize=10)
        ax_h1.set_title("低PBR（バリュー）vs 高PBR（グロース）累積リターン",
                        fontsize=12, fontweight="bold")
        ax_h1.grid(True, alpha=0.2); ax_h1.tick_params(axis="x", rotation=30)
        ax_h2.bar(hml_cum.index, hml_cum * 100,
                  color=["#1a7f37" if v >= 0 else "#d1242f" for v in hml_cum],
                  alpha=0.7, width=1.5)
        ax_h2.axhline(0, color="gray", linewidth=0.7)
        ax_h2.set_ylabel("HML (%)"); ax_h2.set_title("HMLファクター（バリュープレミアム）",
                                                      fontsize=11)
        ax_h2.grid(True, alpha=0.2); ax_h2.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig_hml, clear_figure=True)

        prompt_hml = (
            f"日本株の過去{hml_days}日間HMLファクター: "
            f"低PBR（割安）{l_cum_h.iloc[-1]*100:+.2f}%、"
            f"高PBR（グロース）{h_cum_h.iloc[-1]*100:+.2f}%、"
            f"スプレッド{hml_cum.iloc[-1]*100:+.2f}%。"
            "投資家向け200文字以内: 1)バリュープレミアムの現状 2)バリュー投資戦略への示唆"
        )
        with st.spinner("AI解説中..."):
            try:
                comment, ai_name = generate_ai_comment(prompt_hml)
                st.info(f"🤖 **AI解説（{ai_name}）**\n\n{comment}")
            except Exception as e:
                st.warning(f"AI APIエラー: {e}")

with vl_t4:
    if not _has_results:
        st.info("上部の「パフォーマンス分析」が自動実行後に表示されます")
    else:
        df_vsc = df_factor_merged.dropna(subset=["PBR", "アルファ(%)"]).copy()
        df_vsc = df_vsc[df_vsc["PBR"].between(0.1, 15)]
        df_vsc["バリュー分類"] = df_vsc["PBR"].apply(_value_label)
        try:
            import plotly.express as px
            fig_vsc = px.scatter(
                df_vsc, x="PBR", y="アルファ(%)",
                color="バリュー分類",
                size=df_vsc["シャープレシオ"].clip(lower=0.1),
                hover_name="企業名",
                hover_data={"業種": True, "PBR": ":.2f",
                            "アルファ(%)": ":.2f", "シャープレシオ": ":.2f"},
                title="PBR vs アルファ（バブルサイズ=シャープレシオ）",
                height=600,
                color_discrete_map={
                    "割安(PBR<1)":       "#1a7f37",
                    "適正(PBR 1-2)":     "#42a5f5",
                    "やや割高(PBR 2-3)": "#f57c00",
                    "割高(PBR>3)":       "#d1242f",
                },
            )
            fig_vsc.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_vsc.add_vline(x=1.0, line_dash="dash", line_color="#1a7f37",
                              opacity=0.5, annotation_text="PBR=1")
            fig_vsc.update_layout(xaxis_title="PBR（株価純資産倍率）",
                                  yaxis_title="アルファ (%)", plot_bgcolor="white")
            st.plotly_chart(fig_vsc, use_container_width=True)
        except ImportError:
            fig_vsc2, ax_vs = plt.subplots(figsize=(12, 7))
            for cat, color in zip(_VALUE_ORDER,
                                  ["#1a7f37", "#42a5f5", "#f57c00", "#d1242f"]):
                sub = df_vsc[df_vsc["バリュー分類"] == cat]
                if not sub.empty:
                    ax_vs.scatter(sub["PBR"], sub["アルファ(%)"],
                                  label=cat, color=color, s=60, alpha=0.8)
                    for _, row in sub.iterrows():
                        if abs(row["アルファ(%)"]) > df_vsc["アルファ(%)"].std() * 1.5:
                            ax_vs.annotate(row["企業名"],
                                          (row["PBR"], row["アルファ(%)"]),
                                          fontsize=7, alpha=0.9,
                                          xytext=(4, 4), textcoords="offset points")
            ax_vs.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax_vs.axvline(1.0, color="#1a7f37", linewidth=1.0,
                          linestyle="--", alpha=0.6, label="PBR=1.0")
            ax_vs.set_xlabel("PBR（株価純資産倍率）", fontsize=12)
            ax_vs.set_ylabel("アルファ (%)", fontsize=12)
            ax_vs.set_title("PBR vs アルファ分析マップ", fontsize=12, fontweight="bold")
            ax_vs.legend(fontsize=9); ax_vs.grid(True, alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig_vsc2, clear_figure=True)



# =================================================================
# 🌟 バリューファクターによる価値創造分析
# =================================================================

_ERP_JAPAN = 0.05  # 日本株式リスクプレミアム想定（5%）

st.header("🌟 バリューファクターによる価値創造分析")
st.divider()
st.caption(
    "**価値創造（Value Creation）= ROE > 資本コスト（CoE）**。"
    "CAPMで推定した資本コストとROEを比較し、企業が株主価値を創造しているかを判定します。"
    "伊藤レポートが提唱するROE 8%基準・PBR-ROEマトリクスで「割安の価値創造企業」を発掘します。"
)

try:
    _vc_ok = not df_results.empty and not df_factor.empty
except Exception:
    _vc_ok = False

if not _vc_ok:
    st.info("上部の「パフォーマンス分析」が自動実行後に表示されます")
else:
    df_vc = df_factor_merged.dropna(subset=["ROE(%)", "ベータ", "PBR"]).copy()
    df_vc = df_vc[df_vc["ROE(%)"].abs() < 200]  # 異常値除去

    _rf_pct = risk_free_rate * 100
    df_vc["資本コスト(%)"]        = (_rf_pct + df_vc["ベータ"] * _ERP_JAPAN * 100).round(2)
    df_vc["価値創造スプレッド(%)"] = (df_vc["ROE(%)"] - df_vc["資本コスト(%)"]).round(2)
    df_vc["価値創造判定"]          = df_vc["価値創造スプレッド(%)"].apply(
        lambda x: "✅ 価値創造" if x > 0 else "❌ 価値破壊"
    )
    _avg_coe = df_vc["資本コスト(%)"].mean()
    df_vc["理論PBR"] = (
        df_vc["ROE(%)"] / df_vc["資本コスト(%)"].replace(0, np.nan)
    ).round(2)

    _creators  = (df_vc["価値創造スプレッド(%)"] > 0).sum()
    _destroyers = (df_vc["価値創造スプレッド(%)"] <= 0).sum()
    _ito_pass  = (df_vc["ROE(%)"] >= 8).sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("✅ 価値創造企業", f"{_creators}社", f"全{_creators+_destroyers}社中")
    m2.metric("❌ 価値破壊企業", f"{_destroyers}社")
    m3.metric("平均スプレッド（ROE−CoE）", f"{df_vc['価値創造スプレッド(%)'].mean():+.2f}%")
    m4.metric("伊藤レポート ROE≥8%", f"{_ito_pass}社")

    st.divider()

    vc_t1, vc_t2, vc_t3, vc_t4, vc_t5 = st.tabs([
        "🏆 価値創造ランキング",
        "🗺️ PBR-ROEマトリクス",
        "📊 価値創造スコアカード",
        "🤖 AI価値創造分析",
        "🔬 ROIC-WACC分析",
    ])

    with vc_t1:
        st.markdown("#### 🏆 価値創造スプレッドランキング（ROE − 資本コスト）")
        st.caption(
            f"資本コスト = 無リスク金利（{_rf_pct:.1f}%）"
            f" + β × 株式リスクプレミアム（{_ERP_JAPAN*100:.0f}%）"
        )

        col_cr, col_ds = st.columns(2)
        with col_cr:
            st.markdown("**✅ 価値創造企業 上位30（ROE > CoE）**")
            top_vc = df_vc[df_vc["価値創造スプレッド(%)"] > 0].nlargest(
                30, "価値創造スプレッド(%)")

            def _color_spread(val):
                if isinstance(val, float):
                    if val >= 20: return "color:#1a7f37;font-weight:bold;font-size:14px"
                    if val >= 10: return "color:#1a7f37;font-weight:bold"
                    if val > 0:   return "color:#388e3c"
                return ""

            st.dataframe(
                top_vc[["企業名", "業種", "ROE(%)", "資本コスト(%)",
                         "価値創造スプレッド(%)", "PBR", "アルファ(%)"]].reset_index(drop=True)
                .style.format({
                    "ROE(%)": "{:.1f}%", "資本コスト(%)": "{:.1f}%",
                    "価値創造スプレッド(%)": "{:+.2f}%",
                    "PBR": "{:.2f}", "アルファ(%)": "{:+.2f}%",
                })
                .map(_color_spread, subset=["価値創造スプレッド(%)"]),
                use_container_width=True, hide_index=True,
            )

        with col_ds:
            st.markdown("**❌ 価値破壊企業 下位20（ROE < CoE）**")
            bot_vc = df_vc[df_vc["価値創造スプレッド(%)"] <= 0].nsmallest(
                20, "価値創造スプレッド(%)")

            def _color_neg(val):
                if isinstance(val, float):
                    if val <= -20: return "color:#d1242f;font-weight:bold;font-size:14px"
                    if val <= -10: return "color:#d1242f;font-weight:bold"
                    if val < 0:    return "color:#e57373"
                return ""

            st.dataframe(
                bot_vc[["企業名", "業種", "ROE(%)", "資本コスト(%)",
                         "価値創造スプレッド(%)", "PBR"]].reset_index(drop=True)
                .style.format({
                    "ROE(%)": "{:.1f}%", "資本コスト(%)": "{:.1f}%",
                    "価値創造スプレッド(%)": "{:+.2f}%", "PBR": "{:.2f}",
                })
                .map(_color_neg, subset=["価値創造スプレッド(%)"]),
                use_container_width=True, hide_index=True,
            )

        top20_sp = df_vc.nlargest(20, "価値創造スプレッド(%)")
        fig_sp, ax_sp = plt.subplots(figsize=(14, 6))
        ax_sp.bar(top20_sp["企業名"], top20_sp["価値創造スプレッド(%)"],
                  color=["#1a7f37" if v > 0 else "#d1242f"
                         for v in top20_sp["価値創造スプレッド(%)"]],
                  alpha=0.85)
        ax_sp.axhline(0, color="black", linewidth=0.8)
        ax_sp.axhline(8 - _rf_pct, color="orange", linestyle="--",
                      alpha=0.7, label="伊藤レポート基準（ROE8%相当）")
        ax_sp.set_title("価値創造スプレッド 上位20銘柄（ROE − 資本コスト）",
                        fontsize=12, fontweight="bold")
        ax_sp.set_ylabel("スプレッド (%)")
        ax_sp.tick_params(axis="x", rotation=45, labelsize=9)
        ax_sp.legend(fontsize=9); ax_sp.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_sp, clear_figure=True)

        csv_vc = df_vc.sort_values("価値創造スプレッド(%)", ascending=False)[[
            "企業名", "業種", "ROE(%)", "資本コスト(%)",
            "価値創造スプレッド(%)", "価値創造判定", "PBR", "理論PBR", "アルファ(%)"
        ]].to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "⬇️ 価値創造分析CSV", data=csv_vc,
            file_name=f"value_creation_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv", key="vc_dl",
        )

    with vc_t2:
        st.markdown("#### 🗺️ PBR-ROEマトリクス（価値創造マップ）")
        st.caption(
            "**理論値**: PBR ≈ ROE ÷ 資本コスト（成長ゼロ仮定）。"
            " **💎左上**（高ROE・低PBR）= 割安の価値創造企業（投資好機候補）"
        )

        df_pbr_roe = df_vc.dropna(subset=["PBR", "ROE(%)"])
        df_pbr_roe = df_pbr_roe[df_pbr_roe["PBR"].between(0.1, 15)].copy()

        try:
            import plotly.express as px
            df_pbr_roe["_dist2"] = np.sqrt(
                (df_pbr_roe["ROE(%)"] - df_pbr_roe["ROE(%)"].mean())**2 +
                (df_pbr_roe["PBR"] - df_pbr_roe["PBR"].mean())**2
            )
            top_idx2 = df_pbr_roe.nlargest(20, "_dist2").index
            df_pbr_roe["_label2"] = ""
            df_pbr_roe.loc[top_idx2, "_label2"] = df_pbr_roe.loc[top_idx2, "企業名"]

            fig_pr = px.scatter(
                df_pbr_roe,
                x="ROE(%)", y="PBR",
                color="価値創造判定",
                text="_label2",
                size=df_pbr_roe["時価総額(億円)"].clip(lower=100),
                hover_name="企業名",
                hover_data={
                    "業種": True, "ROE(%)": ":.1f", "PBR": ":.2f",
                    "価値創造スプレッド(%)": ":.2f",
                    "資本コスト(%)": ":.1f", "理論PBR": ":.2f",
                    "時価総額(億円)": ":.0f",
                    "_label2": False, "_dist2": False,
                },
                title="PBR-ROEマトリクス（バブルサイズ=時価総額、ラベル=注目銘柄）",
                height=680,
                color_discrete_map={
                    "✅ 価値創造": "#1a7f37",
                    "❌ 価値破壊": "#d1242f",
                },
            )
            roe_line = np.linspace(
                max(df_pbr_roe["ROE(%)"].min(), -5),
                min(df_pbr_roe["ROE(%)"].max(), 60), 100
            )
            fig_pr.add_scatter(
                x=roe_line, y=roe_line / _avg_coe,
                mode="lines",
                name=f"理論PBRライン（CoE≈{_avg_coe:.1f}%）",
                line=dict(color="#f57c00", dash="dash", width=2),
            )
            fig_pr.add_hline(y=1.0, line_dash="dot", line_color="gray",
                             opacity=0.5, annotation_text="PBR=1")
            fig_pr.add_vline(x=8.0, line_dash="dot", line_color="orange",
                             opacity=0.5, annotation_text="ROE=8%（伊藤）")
            fig_pr.add_vline(x=_rf_pct, line_dash="dot", line_color="#1565c0",
                             opacity=0.3, annotation_text=f"rf={_rf_pct:.1f}%")
            fig_pr.update_traces(
                textposition="top center",
                textfont=dict(size=8, color="rgba(0,0,0,0.72)"),
            )
            fig_pr.update_layout(
                xaxis_title="ROE (%)", yaxis_title="PBR（株価純資産倍率）",
                plot_bgcolor="white",
                hoverlabel=dict(bgcolor="white", font_size=12, namelength=-1),
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        except ImportError:
            fig_pr2, ax_pr = plt.subplots(figsize=(12, 8))
            for verdict, color in [("✅ 価値創造", "#1a7f37"), ("❌ 価値破壊", "#d1242f")]:
                sub = df_pbr_roe[df_pbr_roe["価値創造判定"] == verdict]
                ax_pr.scatter(sub["ROE(%)"], sub["PBR"],
                              label=verdict, color=color, s=60, alpha=0.75)
                for _, row in sub.iterrows():
                    if abs(row["価値創造スプレッド(%)"]) > df_vc["価値創造スプレッド(%)"].std() * 1.5:
                        ax_pr.annotate(row["企業名"], (row["ROE(%)"], row["PBR"]),
                                      fontsize=7, xytext=(4, 4),
                                      textcoords="offset points")
            roe_r = np.linspace(df_pbr_roe["ROE(%)"].min(),
                                df_pbr_roe["ROE(%)"].max(), 100)
            ax_pr.plot(roe_r, roe_r / _avg_coe, "r--", alpha=0.5,
                      label=f"理論PBR（CoE≈{_avg_coe:.1f}%）")
            ax_pr.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
            ax_pr.axvline(8.0, color="orange", linestyle=":", alpha=0.5,
                         label="ROE=8%")
            ax_pr.set_xlabel("ROE (%)"); ax_pr.set_ylabel("PBR")
            ax_pr.set_title("PBR-ROEマトリクス", fontsize=12, fontweight="bold")
            ax_pr.legend(fontsize=9); ax_pr.grid(True, alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig_pr2, clear_figure=True)

        st.markdown("#### 📋 4象限サマリー")
        _med_roe = df_pbr_roe["ROE(%)"].median()
        _med_pbr = df_pbr_roe["PBR"].median()
        q1 = df_pbr_roe[(df_pbr_roe["ROE(%)"] >= _med_roe) & (df_pbr_roe["PBR"] < _med_pbr)]
        q2 = df_pbr_roe[(df_pbr_roe["ROE(%)"] >= _med_roe) & (df_pbr_roe["PBR"] >= _med_pbr)]
        q3 = df_pbr_roe[(df_pbr_roe["ROE(%)"] < _med_roe)  & (df_pbr_roe["PBR"] < _med_pbr)]
        q4 = df_pbr_roe[(df_pbr_roe["ROE(%)"] < _med_roe)  & (df_pbr_roe["PBR"] >= _med_pbr)]

        cq1, cq2, cq3, cq4 = st.columns(4)
        cq1.metric("💎 高ROE・低PBR（割安価値創造）", f"{len(q1)}社")
        cq2.metric("🚀 高ROE・高PBR（成長期待）",     f"{len(q2)}社")
        cq3.metric("⚠️ 低ROE・低PBR（バリュートラップ）", f"{len(q3)}社")
        cq4.metric("🔴 低ROE・高PBR（割高・価値破壊）", f"{len(q4)}社")

        if not q1.empty:
            st.markdown("**💎 割安の価値創造企業（高ROE・低PBR）上位10銘柄**")
            st.dataframe(
                q1.nlargest(10, "価値創造スプレッド(%)")[
                    ["企業名", "業種", "ROE(%)", "PBR", "価値創造スプレッド(%)", "アルファ(%)"]
                ].reset_index(drop=True)
                .style.format({
                    "ROE(%)": "{:.1f}%", "PBR": "{:.2f}",
                    "価値創造スプレッド(%)": "{:+.2f}%", "アルファ(%)": "{:+.2f}%",
                }),
                use_container_width=True, hide_index=True,
            )

    with vc_t3:
        st.markdown("#### 📊 価値創造スコアカード（マルチファクター総合評価）")
        st.caption(
            "価値創造スプレッド・PBR割安度・α・シャープレシオの4軸を"
            "0〜25点に正規化して合算した総合スコア（最高100点）でランキングします。"
        )

        df_score = df_vc.copy()

        def _norm(series, reverse=False, pts=25):
            mn, mx = series.min(), series.max()
            if mx == mn:
                return pd.Series(pts / 2, index=series.index)
            n = (series - mn) / (mx - mn) * pts
            return (pts - n) if reverse else n

        df_score["S_spread"] = _norm(df_score["価値創造スプレッド(%)"])
        df_score["S_pbr"]    = _norm(df_score["PBR"], reverse=True)
        df_score["S_alpha"]  = _norm(df_score["アルファ(%)"])
        df_score["S_sr"]     = _norm(df_score["シャープレシオ"])
        df_score["総合スコア"] = (
            df_score["S_spread"] + df_score["S_pbr"] +
            df_score["S_alpha"]  + df_score["S_sr"]
        ).round(1)
        df_score = df_score.sort_values("総合スコア", ascending=False).reset_index(drop=True)

        def _color_sc(val):
            if isinstance(val, float):
                if val >= 80: return "color:#1a7f37;font-weight:bold;font-size:14px"
                if val >= 60: return "color:#1a7f37;font-weight:bold"
                if val >= 40: return "color:#f57c00"
            return ""

        st.dataframe(
            df_score.head(30)[[
                "企業名", "業種", "総合スコア", "価値創造スプレッド(%)",
                "PBR", "アルファ(%)", "シャープレシオ", "ROE(%)", "価値創造判定"
            ]].style.format({
                "総合スコア": "{:.1f}", "価値創造スプレッド(%)": "{:+.2f}%",
                "PBR": "{:.2f}", "アルファ(%)": "{:+.2f}%",
                "シャープレシオ": "{:.2f}", "ROE(%)": "{:.1f}%",
            }).map(_color_sc, subset=["総合スコア"]),
            use_container_width=True, hide_index=True,
        )

        st.markdown("#### 🕸️ 上位5銘柄 レーダーチャート")
        top5_r = df_score.head(5)
        _rcols  = ["S_spread", "S_pbr", "S_alpha", "S_sr"]
        _rlabels = ["価値創造\nスプレッド", "割安度\n(低PBR)", "アルファ", "シャープ\nレシオ"]
        angles  = np.linspace(0, 2 * np.pi, len(_rcols), endpoint=False).tolist()
        angles += angles[:1]
        colors_r = ["#1565c0", "#1a7f37", "#f57c00", "#7b1fa2", "#d32f2f"]

        fig_r, ax_r = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))
        for i, (_, row) in enumerate(top5_r.iterrows()):
            vals = [row[c] for c in _rcols] + [row[_rcols[0]]]
            ax_r.plot(angles, vals, "o-", linewidth=2,
                     color=colors_r[i], label=row["企業名"])
            ax_r.fill(angles, vals, alpha=0.08, color=colors_r[i])
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(_rlabels, fontsize=10)
        ax_r.set_ylim(0, 25)
        ax_r.set_title("価値創造 上位5銘柄 マルチファクター評価",
                       fontsize=12, fontweight="bold", pad=20)
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
        ax_r.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_r, clear_figure=True)

    with vc_t4:
        st.markdown("#### 🤖 AI価値創造企業 投資テーゼ分析")

        top10_ai = df_score.head(10)
        _vc_sum = top10_ai[[
            "企業名", "業種", "総合スコア", "価値創造スプレッド(%)",
            "PBR", "ROE(%)", "アルファ(%)"
        ]].to_string(index=False)

        _top1 = df_vc.nlargest(1, "価値創造スプレッド(%)").iloc[0]
        _q1_top = q1.nlargest(3, "価値創造スプレッド(%)")[
            ["企業名", "ROE(%)", "PBR"]
        ].to_string(index=False) if not q1.empty else "該当なし"

        _prompt_vc = f"""
あなたは日本株の機関投資家向けバリュー分析ストラテジストです。

【価値創造分析 前提条件】
- 無リスク金利: {_rf_pct:.1f}%  株式リスクプレミアム: {_ERP_JAPAN*100:.0f}%
- 平均資本コスト（CoE）: {_avg_coe:.1f}%
- 価値創造企業: {_creators}社 / 全{_creators+_destroyers}社
- 伊藤レポート ROE8%達成: {_ito_pass}社

【総合スコア上位10銘柄】
{_vc_sum}

【最高スプレッド企業】
{_top1["企業名"]}（ROE {_top1["ROE(%)"]:.1f}%、CoE {_top1["資本コスト(%)"]:.1f}%、スプレッド {_top1["価値創造スプレッド(%)"]:.1f}%）

【割安の価値創造候補（高ROE・低PBR）上位3社】
{_q1_top}

以下の観点で600文字以内で分析してください:
1. 🏆 特に注目すべき価値創造企業とその投資テーゼ
2. 📊 日本市場全体の価値創造状況（ROE vs CoEの現状評価）
3. 💎 割安の価値創造企業（高ROE・低PBR）への戦略的アプローチ
4. ⚠️ バリュートラップを避けるための注意点
5. 💡 伊藤レポート・東証改革の観点からの総合評価

※ 投資判断は自己責任である旨を最後に一言付記してください。
"""
        with st.spinner("AI価値創造分析中..."):
            try:
                comment, ai_name = generate_ai_comment(_prompt_vc)
                st.info(f"🤖 **AI価値創造分析（{ai_name}）**\n\n{comment}")
            except Exception as e:
                st.warning(f"AI APIエラー: {e}")

    with vc_t5:
        st.markdown("#### 🔬 ROIC-WACC 分析（投下資本 vs 加重平均資本コスト）")
        st.caption(
            "**ROIC（Return on Invested Capital）> WACC（Weighted Average Cost of Capital）** = 真の価値創造。"
            "ROE-CoEが株主目線のみなのに対し、**ROIC-WACCは負債コストも含む資本全体**に対するリターンを評価します。"
        )
        st.info(
            f"⚙️ **前提**: 法人税率 30% ／ 負債コスト = 無リスク金利({risk_free_rate*100:.1f}%) + 2.5%（日本IG格クレジットスプレッド）"
            f" ／ CoE = CAPM（β × ERP {_ERP_JAPAN*100:.0f}%）"
        )

        _TAX_JP = 0.30
        _COD    = risk_free_rate + 0.025

        df_rw = df_factor_merged.copy()

        # Revenue: totalRevenue 優先、なければ 時価総額/PSR
        df_rw["Revenue_est"] = np.where(
            df_rw["totalRevenue_raw"].notna() & (df_rw["totalRevenue_raw"] > 0),
            df_rw["totalRevenue_raw"],
            np.where(
                df_rw["PSR"].notna() & (df_rw["PSR"] > 0),
                df_rw["時価総額"] / df_rw["PSR"],
                np.nan,
            ),
        )

        # NOPAT = Revenue × 営業利益率 × (1 - t)
        df_rw["NOPAT"] = (
            df_rw["Revenue_est"] *
            df_rw["営業利益率_raw"].fillna(0) *
            (1 - _TAX_JP)
        )

        # 投下資本 IC = 簿価株主資本 + 有利子負債 - 現金
        df_rw["BookEq"]    = df_rw["時価総額"] / df_rw["PBR"].replace(0, np.nan)
        df_rw["TotalDebt"] = df_rw["totalDebt_raw"].fillna(0)
        df_rw["TotalCash"] = df_rw["totalCash_raw"].fillna(0)
        df_rw["IC"]        = df_rw["BookEq"] + df_rw["TotalDebt"] - df_rw["TotalCash"]

        # ROIC
        df_rw["ROIC(%)"] = (df_rw["NOPAT"] / df_rw["IC"].replace(0, np.nan) * 100).round(2)

        # WACC
        df_rw["CoE_w(%)"] = (_rf_pct + df_rw["ベータ"] * _ERP_JAPAN * 100).round(2)
        df_rw["TC"]   = (df_rw["時価総額"] + df_rw["TotalDebt"]).replace(0, np.nan)
        df_rw["W_E"]  = (df_rw["時価総額"] / df_rw["TC"]).clip(0, 1)
        df_rw["W_D"]  = (df_rw["TotalDebt"] / df_rw["TC"]).clip(0, 1)
        df_rw["WACC(%)"] = (
            df_rw["W_E"] * df_rw["CoE_w(%)"] +
            df_rw["W_D"] * (_COD * 100) * (1 - _TAX_JP)
        ).round(2)

        # ROIC-WACC スプレッド
        df_rw["ROIC-WACC(%)"] = (df_rw["ROIC(%)"] - df_rw["WACC(%)"]).round(2)

        # 有効データのみ残す
        df_rw = df_rw.dropna(subset=["ROIC(%)", "WACC(%)", "ROIC-WACC(%)"])
        df_rw = df_rw[df_rw["ROIC(%)"].between(-100, 200) & df_rw["WACC(%)"].between(0, 50)]
        df_rw["価値創造_RW"] = df_rw["ROIC-WACC(%)"].apply(
            lambda x: "✅ 価値創造" if x > 0 else "❌ 価値破壊"
        )

        _rw_n      = len(df_rw)
        _rw_pos    = (df_rw["ROIC-WACC(%)"] > 0).sum()
        _avg_roic  = df_rw["ROIC(%)"].mean()
        _avg_wacc  = df_rw["WACC(%)"].mean()

        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("✅ ROIC>WACC企業", f"{_rw_pos}社", f"全{_rw_n}社中")
        mm2.metric("平均ROIC", f"{_avg_roic:.1f}%")
        mm3.metric("平均WACC", f"{_avg_wacc:.1f}%")
        mm4.metric("平均スプレッド", f"{df_rw['ROIC-WACC(%)'].mean():+.2f}%")

        st.divider()

        rw_t1, rw_t2, rw_t3 = st.tabs(["🏆 ランキング", "🗺️ ROIC-WACCマップ", "🏭 業種別分析"])

        with rw_t1:
            col_rw1, col_rw2 = st.columns(2)
            with col_rw1:
                st.markdown("**✅ 価値創造企業 上位30（ROIC > WACC）**")
                top_rw = df_rw[df_rw["ROIC-WACC(%)"] > 0].nlargest(30, "ROIC-WACC(%)")

                def _col_rw(v):
                    if not isinstance(v, float): return ""
                    if v >= 15: return "color:#1a7f37;font-weight:bold;font-size:14px"
                    if v >= 5:  return "color:#1a7f37;font-weight:bold"
                    if v > 0:   return "color:#388e3c"
                    return ""

                st.dataframe(
                    top_rw[["企業名", "業種", "ROIC(%)", "WACC(%)", "ROIC-WACC(%)", "PBR"]]
                    .reset_index(drop=True)
                    .style.format({
                        "ROIC(%)": "{:.1f}%", "WACC(%)": "{:.1f}%",
                        "ROIC-WACC(%)": "{:+.2f}%", "PBR": "{:.2f}",
                    }).map(_col_rw, subset=["ROIC-WACC(%)"]),
                    use_container_width=True, hide_index=True,
                )

            with col_rw2:
                st.markdown("**❌ 価値破壊企業 下位20（ROIC < WACC）**")
                bot_rw = df_rw[df_rw["ROIC-WACC(%)"] <= 0].nsmallest(20, "ROIC-WACC(%)")

                def _col_rw_neg(v):
                    if not isinstance(v, float): return ""
                    if v <= -15: return "color:#d1242f;font-weight:bold;font-size:14px"
                    if v <= -5:  return "color:#d1242f;font-weight:bold"
                    if v < 0:    return "color:#e57373"
                    return ""

                st.dataframe(
                    bot_rw[["企業名", "業種", "ROIC(%)", "WACC(%)", "ROIC-WACC(%)", "PBR"]]
                    .reset_index(drop=True)
                    .style.format({
                        "ROIC(%)": "{:.1f}%", "WACC(%)": "{:.1f}%",
                        "ROIC-WACC(%)": "{:+.2f}%", "PBR": "{:.2f}",
                    }).map(_col_rw_neg, subset=["ROIC-WACC(%)"]),
                    use_container_width=True, hide_index=True,
                )

            csv_rw = df_rw.sort_values("ROIC-WACC(%)", ascending=False)[[
                "企業名", "業種", "ROIC(%)", "WACC(%)", "ROIC-WACC(%)", "CoE_w(%)", "PBR", "価値創造_RW"
            ]].to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "⬇️ ROIC-WACC分析CSV", data=csv_rw,
                file_name=f"roic_wacc_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv", key="rw_dl",
            )

        with rw_t2:
            st.markdown("#### 🗺️ ROIC vs WACC マップ")
            st.caption("対角線（ROIC = WACC）の**上方** = 価値創造、**下方** = 価値破壊。バブルサイズ = 時価総額")

            try:
                import plotly.express as px
                df_rw_p = df_rw.copy()
                df_rw_p["_abs_sp"] = df_rw_p["ROIC-WACC(%)"].abs()
                _top_idx = df_rw_p.nlargest(20, "_abs_sp").index
                df_rw_p["_lbl"] = ""
                df_rw_p.loc[_top_idx, "_lbl"] = df_rw_p.loc[_top_idx, "企業名"]

                fig_rw = px.scatter(
                    df_rw_p, x="WACC(%)", y="ROIC(%)",
                    color="価値創造_RW", text="_lbl",
                    size=df_rw_p["時価総額(億円)"].clip(lower=100),
                    hover_name="企業名",
                    hover_data={
                        "業種": True, "ROIC(%)": ":.1f", "WACC(%)": ":.1f",
                        "ROIC-WACC(%)": ":.2f", "PBR": ":.2f",
                        "_lbl": False, "_abs_sp": False,
                    },
                    title="ROIC vs WACC マップ（対角線 = 損益分岐点）",
                    height=680,
                    color_discrete_map={"✅ 価値創造": "#1a7f37", "❌ 価値破壊": "#d1242f"},
                )
                _w_rng = np.linspace(
                    max(df_rw["WACC(%)"].min() - 1, 0),
                    min(df_rw["WACC(%)"].max() + 1, 30), 50,
                )
                fig_rw.add_scatter(
                    x=_w_rng, y=_w_rng, mode="lines",
                    name="ROIC = WACC（損益分岐）",
                    line=dict(color="gray", dash="dash", width=2),
                )
                fig_rw.add_hline(y=0, line_dash="dot", line_color="lightgray", opacity=0.4)
                fig_rw.update_traces(
                    textposition="top center",
                    textfont=dict(size=8, color="rgba(0,0,0,0.72)"),
                )
                fig_rw.update_layout(
                    xaxis_title="WACC (%)", yaxis_title="ROIC (%)",
                    plot_bgcolor="white",
                    hoverlabel=dict(bgcolor="white", font_size=12, namelength=-1),
                )
                st.plotly_chart(fig_rw, use_container_width=True)

            except Exception as _e_rw:
                st.error(f"チャート描画エラー: {_e_rw}")

        with rw_t3:
            st.markdown("#### 🏭 業種別 ROIC-WACC 分析")

            df_sec_rw = df_rw.groupby("業種").agg(
                ROIC平均=("ROIC(%)", "mean"),
                WACC平均=("WACC(%)", "mean"),
                スプレッド平均=("ROIC-WACC(%)", "mean"),
                価値創造企業数=("ROIC-WACC(%)", lambda x: (x > 0).sum()),
                銘柄数=("ROIC-WACC(%)", "count"),
            ).reset_index()
            df_sec_rw["価値創造率(%)"] = (
                df_sec_rw["価値創造企業数"] / df_sec_rw["銘柄数"] * 100
            ).round(1)
            df_sec_rw = df_sec_rw.sort_values("スプレッド平均", ascending=False)

            def _col_sec(v):
                if not isinstance(v, float): return ""
                if v > 5:  return "color:#1a7f37;font-weight:bold"
                if v > 0:  return "color:#388e3c"
                if v < 0:  return "color:#d1242f"
                return ""

            st.dataframe(
                df_sec_rw.style.format({
                    "ROIC平均": "{:.1f}%", "WACC平均": "{:.1f}%",
                    "スプレッド平均": "{:+.2f}%", "価値創造率(%)": "{:.0f}%",
                }).map(_col_sec, subset=["スプレッド平均"]),
                use_container_width=True, hide_index=True,
            )

            fig_sec, ax_sec = plt.subplots(figsize=(12, 6))
            _sec_colors = ["#1a7f37" if v > 0 else "#d1242f"
                           for v in df_sec_rw["スプレッド平均"]]
            ax_sec.bar(df_sec_rw["業種"], df_sec_rw["スプレッド平均"],
                       color=_sec_colors, alpha=0.85)
            ax_sec.axhline(0, color="black", linewidth=0.8)
            ax_sec.set_title("業種別 平均 ROIC-WACC スプレッド", fontsize=12, fontweight="bold")
            ax_sec.set_ylabel("スプレッド (%)")
            ax_sec.tick_params(axis="x", rotation=45, labelsize=9)
            ax_sec.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_sec, clear_figure=True)


# ================================================================
# 🎯 テーマ市場サマリー
# ================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def calc_theme_performance() -> pd.DataFrame:
    """テーマ別週次・月次リターンと出来高比率を計算"""
    import warnings
    rows = []
    all_tickers = list({t for tlist in THEME_GROUPS.values() for t in tlist})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_all = _yfdownload(all_tickers, period="2mo")
        if df_all.empty:
            return pd.DataFrame()
        close_df = df_all["Close"] if "Close" in df_all.columns else pd.DataFrame()
        vol_df   = df_all["Volume"] if "Volume" in df_all.columns else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    for theme, tickers in THEME_GROUPS.items():
        try:
            valid = [t for t in tickers if t in close_df.columns]
            if not valid:
                continue
            c = close_df[valid].dropna(how="all")
            v = vol_df[valid].dropna(how="all") if not vol_df.empty else pd.DataFrame()
            if len(c) < 5:
                continue
            w_ret = float((c.iloc[-1] / c.iloc[-6].replace(0, float("nan")) - 1).mean() * 100) if len(c) >= 6 else 0.0
            m_ret = float((c.iloc[-1] / c.iloc[-22].replace(0, float("nan")) - 1).mean() * 100) if len(c) >= 22 else w_ret
            if not v.empty and len(v) >= 10:
                vol_ratio = float(v.tail(5).mean().mean() / (v.iloc[-25:-5].mean().mean() + 1e-8))
            else:
                vol_ratio = 1.0
            rows.append({"テーマ": theme, "週次(%)": round(w_ret, 1),
                         "月次(%)": round(m_ret, 1), "出来高比": round(vol_ratio, 1)})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["週次ランク"] = df["週次(%)"].rank(ascending=False).astype(int)
    df["月次ランク"] = df["月次(%)"].rank(ascending=False).astype(int)
    df["ランク変化"] = df["月次ランク"] - df["週次ランク"]
    return df.sort_values("週次(%)", ascending=False).reset_index(drop=True)


def _draw_speedometer(value: int) -> "plt.Figure":
    """半円スピードメーター（テーマ温度計）"""
    import numpy as np
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.3, 1.2)
    colors = ["#c62828","#ef6c00","#f9a825","#1565c0","#2e7d32"]
    for i, col in enumerate(colors):
        t1 = np.radians(180 - i * 36)
        t2 = np.radians(180 - (i + 1) * 36)
        th = np.linspace(t2, t1, 60)
        ro, ri = 1.0, 0.6
        xo, yo = ro * np.cos(th), ro * np.sin(th)
        xi, yi = ri * np.cos(th[::-1]), ri * np.sin(th[::-1])
        ax.fill(np.concatenate([xo, xi]), np.concatenate([yo, yi]), color=col, alpha=0.85)
    ang = np.radians(180 - value * 1.8)
    ax.plot([0, 0.72 * np.cos(ang)], [0, 0.72 * np.sin(ang)],
            color="black", linewidth=2.5, zorder=5, solid_capstyle="round")
    ax.add_patch(plt.Circle((0, 0), 0.07, color="black", zorder=6))
    label = "強気" if value >= 70 else ("弱気" if value < 40 else "中立")
    ax.text(0, 0.28, str(value), ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(0, 0.08, "/100", ha="center", va="center", fontsize=9, color="gray")
    ax.text(0, -0.12, label, ha="center", va="center", fontsize=10, color="gray")
    ax.axis("off")
    plt.tight_layout(pad=0.1)
    return fig


st.header("🎯 テーマ市場サマリー")
st.divider()

with st.spinner("テーマパフォーマンスを計算中..."):
    _df_theme = calc_theme_performance()

if _df_theme.empty:
    st.warning("テーマデータを取得できませんでした（yfinanceのレート制限の可能性があります）")
else:
    # ── ① サマリー指標 3カラム ─────────────────────────────────────
    _n_up   = int((_df_theme["週次(%)"] > 0).sum())
    _n_all  = len(_df_theme)
    _up_pct = round(_n_up / _n_all * 100, 1) if _n_all else 0
    _temp   = min(100, max(0, int(_up_pct * 1.5 - 25 + _df_theme["週次(%)"].mean() * 3)))
    _avg_vol= round(_df_theme["出来高比"].mean(), 2)
    _vol_chg= round((_avg_vol - 1.0) * 100, 1)

    _sm1, _sm2, _sm3 = st.columns(3)
    with _sm1:
        st.markdown("**テーマ温度**")
        _fig_gauge = _draw_speedometer(_temp)
        st.pyplot(_fig_gauge, clear_figure=True)
    with _sm2:
        st.markdown("**値上がりテーマ比率**")
        import plotly.graph_objects as _go_th
        _fig_donut = _go_th.Figure(_go_th.Pie(
            values=[_up_pct, 100 - _up_pct],
            labels=["上昇", "下落"],
            hole=0.6,
            marker_colors=["#2e7d32", "#ffcdd2"],
            textinfo="none",
        ))
        _fig_donut.update_layout(
            showlegend=True, height=220, margin=dict(t=10, b=10, l=10, r=10),
            annotations=[dict(text=f"{_up_pct}%<br>上昇", x=0.5, y=0.5,
                              font_size=16, showarrow=False)]
        )
        st.plotly_chart(_fig_donut, use_container_width=True)
    with _sm3:
        st.markdown("**売買代金前日比（平均）**")
        _vol_color = "#2e7d32" if _vol_chg >= 0 else "#c62828"
        st.markdown(
            f"<div style='font-size:2.2rem;font-weight:bold;color:{_vol_color};margin-top:28px'>"
            f"{_vol_chg:+.1f}%</div>",
            unsafe_allow_html=True
        )
        _bar_pct = min(100, max(0, 50 + _vol_chg * 2))
        st.progress(int(_bar_pct))
        _vol_label = "買いは活発" if _vol_chg > 5 else ("売りが優勢" if _vol_chg < -5 else "概ね平常")
        st.caption(_vol_label)

    st.divider()

    # ── ② 出来高急増テーマ ──────────────────────────────────────────
    st.markdown("##### ② 出来高急増テーマ")
    _df_vol_surge = _df_theme.nlargest(4, "出来高比")[["テーマ","出来高比","週次(%)"]]
    _surge_cols = st.columns(4)
    for i, (_, row) in enumerate(_df_vol_surge.iterrows()):
        with _surge_cols[i]:
            ret_color = "#2e7d32" if row["週次(%)"] >= 0 else "#c62828"
            st.markdown(
                f"**{row['テーマ']}**  \n"
                f"<span style='font-size:1.3rem;font-weight:bold'>{row['出来高比']:.1f}x</span>  \n"
                f"<span style='color:{ret_color}'>当日リターン {row['週次(%)']:+.1f}%</span>",
                unsafe_allow_html=True
            )

    st.divider()

    # ── ③ 主な変動（急浮上 / 常勝 / 失速）─────────────────────────
    st.markdown("##### ③ 主な変動")
    _col_rise, _col_steady, _col_fall = st.columns(3)

    # 急浮上: 週次ランクが月次より大幅改善
    _df_rise = _df_theme[_df_theme["ランク変化"] > 0].nsmallest(9, "週次ランク")[
        ["テーマ","月次ランク","週次ランク","週次(%)"]].reset_index(drop=True)
    with _col_rise:
        st.markdown("**急浮上** <small>週近で動き出したテーマ</small>", unsafe_allow_html=True)
        for _, r in _df_rise.iterrows():
            chg = int(r["月次ランク"] - r["週次ランク"])
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:2px 0'>"
                f"<span>{r['テーマ']}</span>"
                f"<span style='color:#2e7d32;font-size:.85rem'>▲{chg} 位</span>"
                f"<span style='font-size:.85rem'>{int(r['週次ランク'])}位</span></div>",
                unsafe_allow_html=True
            )

    # 常勝: 週次・月次ともに上位半分
    _median_rank = _n_all / 2
    _df_steady = _df_theme[
        (_df_theme["週次ランク"] <= _median_rank) & (_df_theme["月次ランク"] <= _median_rank)
    ].nsmallest(6, "週次ランク")[["テーマ","月次ランク","週次ランク"]].reset_index(drop=True)
    with _col_steady:
        st.markdown("**常勝** <small>上位を維持しているテーマ</small>", unsafe_allow_html=True)
        for _, r in _df_steady.iterrows():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:2px 0'>"
                f"<span>{r['テーマ']}</span>"
                f"<span style='color:#1565c0;font-size:.85rem'>● {int(r['月次ランク'])}位</span>"
                f"<span style='font-size:.85rem'>{int(r['週次ランク'])}位</span></div>",
                unsafe_allow_html=True
            )

    # 失速: 月次より週次ランクが大幅悪化
    _df_fall = _df_theme[_df_theme["ランク変化"] < 0].nlargest(12, "週次ランク")[
        ["テーマ","月次ランク","週次ランク","週次(%)"]].reset_index(drop=True)
    with _col_fall:
        st.markdown("**失速** <small>順位を落としたテーマ</small>", unsafe_allow_html=True)
        for _, r in _df_fall.iterrows():
            drop = int(r["週次ランク"] - r["月次ランク"])
            ret_color = "#c62828" if r["週次(%)"] < 0 else "#555"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:2px 0'>"
                f"<span style='color:{ret_color}'>{r['テーマ']}</span>"
                f"<span style='color:#c62828;font-size:.85rem'>▼{drop} 位</span>"
                f"<span style='font-size:.85rem'>{int(r['週次ランク'])}位</span></div>",
                unsafe_allow_html=True
            )


# ================================================================
# 📊 需給分析
# ================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def calc_demand_supply(ticker: str) -> dict:
    """yfinance + J-Quants（あれば）で需給スコアを計算"""
    try:
        df = _yfdownload(ticker, period="1y")
        if df.empty:
            return {"error": "データなし"}
        close  = df["Close"].dropna()
        volume = df["Volume"].dropna()
        if len(close) < 10:
            return {"error": "データ不足"}

        current = float(close.iloc[-1])
        high52  = float(close.tail(252).max())
        vol5    = float(volume.tail(5).mean())
        vol20   = float(volume.tail(20).mean())
        vol_ratio = vol5 / (vol20 + 1e-8)

        # 基礎スコア (0–100)
        pos_score  = max(0, min(30, (current / high52) * 30))
        vol_score  = max(0, min(20, (vol_ratio - 0.5) * 20))
        mom5       = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0
        mom_score  = max(0, min(20, (mom5 + 5) * 2))
        ma25_score = 15 if current > float(close.tail(25).mean()) else 0
        ma75       = float(close.tail(75).mean()) if len(close) >= 75 else float(close.tail(25).mean())
        ma75_score = 15 if current > ma75 else 0
        base_score = int(pos_score + vol_score + mom_score + ma25_score + ma75_score)

        # J-Quants 信用データ（Standard プラン）
        code4 = ticker.replace(".T", "").zfill(4)
        credit_ratio, credit_score, balance_score = None, 0, 0
        df_mg = pd.DataFrame()
        jq_key = st.secrets.get("JQUANTS_API_KEY", "")
        if jq_key:
            end_d  = datetime.today().strftime("%Y%m%d")
            start_d = (datetime.today() - __import__("dateutil.relativedelta", fromlist=["relativedelta"]).relativedelta(months=3)).strftime("%Y%m%d")
            df_mg = jq_fetch_margin(code4, start_d, end_d)
            if not df_mg.empty:
                buy_col  = next((c for c in df_mg.columns if "longmargin"  in c.lower()), None)
                sell_col = next((c for c in df_mg.columns if "shortmargin" in c.lower()), None)
                if buy_col and sell_col:
                    buy_ser  = df_mg[buy_col].astype(float)
                    sell_ser = df_mg[sell_col].astype(float)
                    credit_ratio = float(buy_ser.iloc[-1] / (sell_ser.iloc[-1] + 1e-8))
                    # 信用倍率スコア
                    if credit_ratio < 1.0:    credit_score = 5
                    elif credit_ratio < 2.0:  credit_score = 0
                    elif credit_ratio < 3.0:  credit_score = -2
                    elif credit_ratio < 5.0:  credit_score = -8
                    elif credit_ratio < 10.0: credit_score = -15
                    else:                     credit_score = -25
                    # 残高トレンドスコア
                    if len(buy_ser) >= 3:
                        trend = (buy_ser.iloc[-1] - buy_ser.iloc[-3]) / (buy_ser.iloc[-3] + 1e-8)
                        if trend < -0.05:   balance_score = 10
                        elif trend < 0.05:  balance_score = 0
                        elif trend < 0.15:  balance_score = -8
                        else:               balance_score = -15

        total = max(-30, min(100, base_score + credit_score + balance_score))
        grade = "A" if total >= 80 else ("B" if total >= 60 else ("C" if total >= 40 else ("D" if total >= 20 else "E")))
        if total >= 70:   status, detail = "需給は強い圏内です", "買い需要が売り圧力を上回っています"
        elif total >= 50: status, detail = "需給は中立圏です",   "売買内訳と信用残の変化を継続確認してください"
        else:             status, detail = "需給は弱い圏内です", "売り圧力が高まっています。注意が必要です"

        return {
            "base": base_score, "credit_sc": credit_score,
            "balance_sc": balance_score, "total": total, "grade": grade,
            "status": status, "detail": detail,
            "credit_ratio": credit_ratio, "vol_ratio": vol_ratio, "mom5": mom5,
            "current": current, "high52": high52,
            "vol_today": int(volume.iloc[-1]), "vol20avg": int(vol20),
            "df": df, "df_mg": df_mg,
        }
    except Exception as e:
        return {"error": str(e)}


st.header("📊 需給分析")
st.divider()
st.caption("信用取引残高・売買動向・テクニカルから需給バランスをスコアリング（信用データはJ-Quants Standardプラン以上）")

_sq_all_tickers = {
    f"{n}（{t}）": t
    for t, (n, _) in {**ticker_name_map, **MEMORY_TICKERS}.items()
}
_sq_c1, _sq_c2, _sq_c3 = st.columns([3, 1, 1])
with _sq_c1:
    _sq_sel = st.selectbox("銘柄を選択", list(_sq_all_tickers.keys()), key="sq_sel_ticker")
    _sq_ticker = _sq_all_tickers[_sq_sel]
with _sq_c2:
    st.markdown(""); st.markdown("")
    _sq_run = st.button("▶ 需給分析を実行", type="primary", key="sq_run")

if _sq_run:
    with st.spinner(f"{_sq_sel} の需給データを取得・計算中..."):
        _sq = calc_demand_supply(_sq_ticker)

    if "error" in _sq:
        st.error(f"取得失敗: {_sq['error']}")
    else:
        # ── スコアカード ────────────────────────────────────────────
        _grade_colors = {"A":"#1b5e20","B":"#1565c0","C":"#f57f17","D":"#bf360c","E":"#6a1b9a"}
        _gc = _grade_colors.get(_sq["grade"], "#555")
        _sc1, _sc2 = st.columns([1, 3])
        with _sc1:
            st.markdown(
                f"<div style='border:2px solid #ddd;border-radius:10px;padding:20px;text-align:center'>"
                f"<div style='font-size:.8rem;color:gray'>需給ランク</div>"
                f"<div style='font-size:4rem;font-weight:bold;color:{_gc}'>{_sq['grade']}</div>"
                f"<div style='font-size:1.4rem;font-weight:bold'>{_sq['total']}<span style='font-size:.9rem;color:gray'>/100</span></div>"
                f"</div>", unsafe_allow_html=True
            )
        with _sc2:
            st.markdown(f"**需給ステータス**: {_sq['status']}")
            st.caption(_sq["detail"])
            # カラーグラデーションバー
            _bar_pos = max(0, min(100, _sq["total"]))
            st.markdown(
                f"<div style='height:14px;border-radius:7px;background:linear-gradient(to right,#c62828,#ef6c00,#f9a825,#1565c0,#2e7d32);margin:8px 0'>"
                f"<div style='height:14px;width:{_bar_pos}%;border-right:3px solid black;border-radius:7px 0 0 7px'></div></div>",
                unsafe_allow_html=True
            )
            # スコア内訳
            _bd = _sq
            st.markdown(
                f"スコア内訳: 基礎 **{_bd['base']}** "
                f"{'%+d' % _bd['credit_sc']} (信用倍率) "
                f"{'%+d' % _bd['balance_sc']} (残高トレンド) "
                f"= **{_bd['total']}**"
            )
            _sc_cols = st.columns(4)
            _sc_cols[0].metric("基礎スコア", f"{_bd['base']}")
            _sc_cols[1].metric("信用倍率スコア", f"{_bd['credit_sc']:+d}")
            _sc_cols[2].metric("残高トレンドスコア", f"{_bd['balance_sc']:+d}")
            cr_val = f"{_sq['credit_ratio']:.2f}倍" if _sq["credit_ratio"] else "—"
            _sc_cols[3].metric("信用倍率", cr_val)

        st.divider()

        # ── 主要メトリクス ─────────────────────────────────────────
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("現在株価", f"¥{_sq['current']:,.0f}")
        _m2.metric("5日騰落率", f"{_sq['mom5']:+.1f}%")
        _m3.metric("出来高比（5日/20日）", f"{_sq['vol_ratio']:.2f}x")
        _m4.metric("52週高値比", f"{_sq['current']/_sq['high52']*100:.1f}%")

        st.divider()

        _ch1, _ch2 = st.columns(2)

        # ── ① 当日の売買動向（出来高バー）──────────────────────────
        with _ch1:
            st.markdown("**① 当日の売買動向**")
            _df_ch = _sq["df"]
            _close = _df_ch["Close"].dropna()
            _vol   = _df_ch["Volume"].dropna()
            _recent = _df_ch.tail(10)
            _dates  = [d.strftime("%m/%d") for d in _recent.index]
            _vols   = _recent["Volume"].tolist()
            _avg_v  = float(_vol.tail(20).mean())
            _colors_v = ["#2e7d32" if float(_recent["Close"].iloc[i]) >= float(_recent["Close"].iloc[i-1] if i>0 else _recent["Close"].iloc[i])
                         else "#c62828" for i in range(len(_recent))]
            _fig_v, _ax_v = plt.subplots(figsize=(6, 3))
            _bars_v = _ax_v.bar(_dates, [v/1e4 for v in _vols], color=_colors_v, alpha=0.8)
            _ax_v.axhline(_avg_v/1e4, color="orange", linestyle="--", linewidth=1.2, label="20日平均")
            _ax_v.set_ylabel("万株"); _ax_v.set_title("出来高（直近10営業日）", fontsize=10)
            _ax_v.legend(fontsize=8); _ax_v.grid(True, axis="y", alpha=0.3)
            plt.xticks(rotation=45, fontsize=8); plt.tight_layout()
            st.pyplot(_fig_v, clear_figure=True)
            _today_vol = _sq["vol_today"]; _avg20 = _sq["vol20avg"]
            _diff = _today_vol - _avg20
            _sign = "買い超" if _diff >= 0 else "売り超"
            st.caption(f"差し引き {_sign} {abs(_diff)/1e4:+.1f}万株  |  5日平均 {vol5/1e4:+.1f}万株")

        # ── ② 株価推移（25日・75日MA）────────────────────────────
        with _ch2:
            st.markdown("**② 株価推移（MA付き）**")
            _close_plot = _close.tail(90)
            _fig_c, _ax_c = plt.subplots(figsize=(6, 3))
            _ax_c.plot(_close_plot.index, _close_plot.values, color="#1565c0", linewidth=1.5, label="株価")
            if len(_close) >= 25:
                _ma25_s = _close.rolling(25).mean().tail(90)
                _ax_c.plot(_close_plot.index, _ma25_s.values[-len(_close_plot):],
                           color="orange", linewidth=1, linestyle="--", label="MA25")
            if len(_close) >= 75:
                _ma75_s = _close.rolling(75).mean().tail(90)
                _ax_c.plot(_close_plot.index, _ma75_s.values[-len(_close_plot):],
                           color="#c62828", linewidth=1, linestyle="--", label="MA75")
            _ax_c.set_title("株価チャート（90日）", fontsize=10)
            _ax_c.legend(fontsize=8); _ax_c.grid(True, alpha=0.25)
            plt.xticks(rotation=45, fontsize=7); plt.tight_layout()
            st.pyplot(_fig_c, clear_figure=True)

        # ── ③ 信用残推移（J-Quants あれば）──────────────────────
        st.markdown("**③ 信用残推移**")
        _df_mg = _sq.get("df_mg", pd.DataFrame())
        if not _df_mg.empty:
            _buy_col  = next((c for c in _df_mg.columns if "longmargin"  in c.lower()), None)
            _sell_col = next((c for c in _df_mg.columns if "shortmargin" in c.lower()), None)
            if _buy_col and _sell_col and "Date" in _df_mg.columns:
                _fig_mg, _ax_mg = plt.subplots(figsize=(10, 3.5))
                _ax2_mg = _ax_mg.twinx()
                _x_mg = range(len(_df_mg))
                _w = 0.35
                _buy_v  = _df_mg[_buy_col].astype(float)
                _sell_v = _df_mg[_sell_col].astype(float)
                _ax_mg.bar([x - _w/2 for x in _x_mg], _buy_v/1000, width=_w,
                           color="#2e7d32", alpha=0.7, label="信用買残(千株)")
                _ax_mg.bar([x + _w/2 for x in _x_mg], _sell_v/1000, width=_w,
                           color="#ffcdd2", alpha=0.7, label="信用売残(千株)")
                _ratio_mg = _buy_v / (_sell_v + 1e-8)
                _ax2_mg.plot(_x_mg, _ratio_mg, color="black", linewidth=1.8,
                             marker="o", markersize=3, label="信用倍率")
                _xlabels = [d.strftime("%m/%d") if hasattr(d, "strftime") else str(d)
                            for d in _df_mg["Date"]]
                _ax_mg.set_xticks(list(_x_mg)); _ax_mg.set_xticklabels(_xlabels, rotation=45, fontsize=7)
                _ax_mg.set_ylabel("千株"); _ax2_mg.set_ylabel("信用倍率")
                _ax_mg.set_title("信用残推移", fontsize=10)
                lines1, labs1 = _ax_mg.get_legend_handles_labels()
                lines2, labs2 = _ax2_mg.get_legend_handles_labels()
                _ax_mg.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")
                _ax_mg.grid(True, axis="y", alpha=0.2); plt.tight_layout()
                st.pyplot(_fig_mg, clear_figure=True)
            else:
                st.dataframe(_df_mg.tail(10), use_container_width=True)
        else:
            st.info("信用残データはJ-Quants Standardプラン以上で取得できます。基礎スコアのみ表示しています。")


# ================================================================
# 💾 メモリ業界分析
# ================================================================
st.header("💾 メモリ業界分析")
st.divider()
st.caption(
    "半導体メモリ関連銘柄の適時開示・国内外ニュースをAIが時間軸別に分析。"
    "対象: キオクシア, 東京エレクトロン, アドバンテスト, レーザーテック, 信越化学, SUMCO 他"
)

_mem_t1, _mem_t2, _mem_t3 = st.tabs(["⏱️ 24時間の影響", "📅 1週間の影響", "📊 3ヶ月の影響"])

def _render_memory_tab(horizon_label: str, tdnet_days: int):
    """メモリ業界分析タブの共通描画"""
    col_tdnet, col_news = st.columns(2)

    # ── 適時開示（メモリ関連銘柄）
    with col_tdnet:
        st.markdown("##### 📋 関連銘柄の適時開示")
        mem_code_map = {t: (n, s) for t, (n, s) in MEMORY_TICKERS.items()}
        with st.spinner(f"TDnetを取得中（{tdnet_days}営業日）..."):
            _mem_tdnet = fetch_tdnet_week(mem_code_map, days=tdnet_days)
        if _mem_tdnet:
            _df_mem_td = (
                pd.DataFrame(_mem_tdnet)
                .sort_values("日付", ascending=False)
                [["日付", "企業名", "タイトル"]]
            )
            st.dataframe(_df_mem_td, use_container_width=True, hide_index=True,
                         height=min(300, 36 * len(_df_mem_td) + 40))
        else:
            st.info(f"直近{tdnet_days}営業日に対象銘柄の適時開示はありません")

    # ── 国内ニュース
    with col_news:
        st.markdown("##### 🗞️ 国内ニュース（半導体・メモリ関連）")
        _dom_news = fetch_memory_news_domestic(15)
        if _dom_news:
            for n in _dom_news[:8]:
                st.markdown(
                    f"**[{n['title']}]({n['link']})**  \n"
                    f"<small>{n['source']} | {n.get('date','')[:16]}</small>",
                    unsafe_allow_html=True
                )
                st.divider()
        else:
            st.info("国内ニュースは取得できませんでした")

    # ── 海外ニュース
    st.markdown("##### 🌐 海外ニュース（英語）")
    _ovs_news = fetch_memory_news_overseas(15)
    if _ovs_news:
        _ovs_cols = st.columns(2)
        for i, n in enumerate(_ovs_news[:6]):
            with _ovs_cols[i % 2]:
                st.markdown(
                    f"**[{n['title']}]({n['link']})**  \n"
                    f"<small>{n['source']} | {n.get('date','')[:16]}</small>",
                    unsafe_allow_html=True
                )
                st.divider()
    else:
        st.info("海外ニュースは取得できませんでした")

    # ── AI総合分析
    st.markdown(f"##### 🤖 AI分析：{horizon_label}の影響まとめ")
    if st.button(f"▶ AI分析を実行（{horizon_label}）", key=f"mem_ai_{tdnet_days}"):
        _tdnet_titles = "\n".join(
            f"- [{r['日付']}] {r['企業名']}: {r['タイトル']}"
            for r in (_mem_tdnet or [])[:30]
        ) or "（適時開示なし）"
        _dom_titles = "\n".join(
            f"- {n['title']}" for n in (_dom_news or [])[:10]
        ) or "（国内ニュースなし）"
        _ovs_titles = "\n".join(
            f"- {n['title']}" for n in (_ovs_news or [])[:10]
        ) or "（海外ニュースなし）"

        _mem_prompt = f"""
あなたは半導体・メモリ業界の専門アナリストです。
以下の情報をもとに、**{horizon_label}の日本のメモリ・半導体関連株への影響**を分析してください。

## 対象銘柄の適時開示（直近{tdnet_days}営業日）
{_tdnet_titles}

## 国内ニュース（半導体・メモリ関連）
{_dom_titles}

## 海外ニュース（英語）
{_ovs_titles}

## 分析指示
1. **{horizon_label}の主要材料**を箇条書きで3〜5点
2. **ポジティブ材料** と **ネガティブ材料** をそれぞれ整理
3. **注目銘柄と注目理由**（東エレク, アドテスト, レーザーテク, 信越化, SUMCO など）
4. **{horizon_label}の総合見通し**（強気/中立/弱気）と根拠

簡潔・具体的に。投資推奨は含まないこと。
"""
        with st.spinner("AI分析中..."):
            _mem_ai_text, _mem_ai_model = generate_ai_comment(_mem_prompt)
        st.markdown(_mem_ai_text)
        st.caption(f"by {_mem_ai_model}")


with _mem_t1:
    st.markdown("**直近1営業日**の適時開示・ニュースから当日〜翌日の短期影響を分析")
    _render_memory_tab("24時間", tdnet_days=1)

with _mem_t2:
    st.markdown("**直近5営業日（1週間）**の適時開示・ニュースから今週の影響を分析")
    _render_memory_tab("1週間", tdnet_days=5)

with _mem_t3:
    st.markdown("**直近60営業日（約3ヶ月）**の適時開示から中期トレンドを分析（初回取得に2〜3分かかります）")
    _render_memory_tab("3ヶ月", tdnet_days=60)


st.divider()
st.caption("データソース: Yahoo Finance / J-Quants / Finnhub / Alpha Vantage / TDnet / 株探 / みんかぶ / 日経 | 投資判断は自己責任で")