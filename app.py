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

# ── アクセス解析モジュール ────────────────────────────────────────
try:
    from analytics import track_pageview
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    def track_pageview(*a, **kw):
        pass

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
GROQ_MODEL = "llama-3.3-8b-instant"

# -----------------------------
# ページ設定
# -----------------------------
st.set_page_config(layout="wide", page_title="📈 日本株 分析ダッシュボード", page_icon="📈", initial_sidebar_state="expanded")
st.title("📈 日本株 シャープレシオ分析 + ニュース統合")

# ── アクセス計測（1セッション1回）────────────────────────────────
track_pageview("jstock")

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
    """
    yfinance v0.2以降のMultiIndex列を自動フラット化して返す。
    単一銘柄でも ('Close','7203.T') → 'Close' に変換。
    """
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
        import logging as _logging
        _logging.getLogger(__name__).warning(f"_yfdownload({ticker}): {e}")
        return pd.DataFrame()


def _to_series(col) -> pd.Series:
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

@st.cache_data(ttl=1800)
def get_sector_performance(ticker_name_map: dict, period_days: int = 20) -> pd.DataFrame:
    from datetime import timedelta
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=period_days + 10)
    sector_returns = {}
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            df = _yfdownload(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 2:
                continue
            close = _to_series(df["Close"]).dropna()
            ret = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append(float(ret))
        except Exception:
            continue
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
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days + 5)
    sector_price_data = {}
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            df = _yfdownload(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 5:
                continue
            close = _to_series(df["Close"]).dropna()
            norm  = close / close.iloc[0] * 100
            if sector not in sector_price_data:
                sector_price_data[sector] = []
            sector_price_data[sector].append(norm)
        except Exception:
            continue
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
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax = axes[0]
    cmap = plt.cm.get_cmap("Greens", len(top_sectors) + 2)
    for i, sec in enumerate(top_sectors):
        if sec in df_ts.columns:
            series = df_ts[sec].dropna()
            ax.plot(series.index, series - 100, label=sec, color=cmap(i + 2), linewidth=1.8)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_title("買われているセクター（累積リターン）", fontsize=11, fontweight="bold")
    ax.set_ylabel("累積リターン (%)")
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(axis="x", rotation=30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax = axes[1]
    cmap2 = plt.cm.get_cmap("Reds", len(bottom_sectors) + 2)
    for i, sec in enumerate(bottom_sectors):
        if sec in df_ts.columns:
            series = df_ts[sec].dropna()
            ax.plot(series.index, series - 100, label=sec, color=cmap2(i + 2), linewidth=1.8)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_title("売られているセクター（累積リターン）", fontsize=11, fontweight="bold")
    ax.set_ylabel("累積リターン (%)")
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(axis="x", rotation=30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_sector_heatmap(df_multi: pd.DataFrame) -> plt.Figure:
    df_heat = df_multi.set_index("業種")[["1週間", "1ヶ月", "3ヶ月"]]
    df_heat = df_heat.sort_values("1ヶ月", ascending=False)
    vmax = max(abs(df_heat.values.max()), abs(df_heat.values.min()), 3)
    fig, ax = plt.subplots(figsize=(9, max(6, len(df_heat) * 0.42)))
    im = ax.imshow(df_heat.values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(df_heat.columns)))
    ax.set_xticklabels(df_heat.columns, fontsize=10)
    ax.set_yticks(range(len(df_heat.index)))
    ax.set_yticklabels(df_heat.index, fontsize=9)
    for i in range(len(df_heat.index)):
        for j in range(len(df_heat.columns)):
            val = df_heat.values[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
    plt.colorbar(im, ax=ax, label="リターン (%)", shrink=0.8)
    ax.set_title("セクター別リターン ヒートマップ（期間比較）", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


# ================================================================
# 🔥 需給系モジュール
# ================================================================

@st.cache_data(ttl=1800)
def get_volume_surge(ticker_name_map: dict, surge_ratio: float = 2.0,
                     short_days: int = 5, base_days: int = 20) -> pd.DataFrame:
    from datetime import timedelta
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=base_days + 10)
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            df = _yfdownload(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < base_days:
                continue
            vol = _to_series(df["Volume"]).dropna()
            recent_avg = vol.iloc[-short_days:].mean()
            base_avg   = vol.iloc[-base_days:-short_days].mean()
            if base_avg == 0:
                continue
            ratio = recent_avg / base_avg
            price_chg = (_to_series(df["Close"]).iloc[-1] - _to_series(df["Close"]).iloc[-short_days]) / _to_series(df["Close"]).iloc[-short_days] * 100
            if ratio >= surge_ratio:
                results.append({
                    "企業名": name, "業種": sector, "ティッカー": ticker,
                    "出来高倍率": round(ratio, 2),
                    "直近5日平均出来高": int(recent_avg),
                    "基準平均出来高": int(base_avg),
                    "株価変化率(5日%)": round(float(price_chg), 2),
                    "最新株価": round(float(_to_series(df["Close"]).iloc[-1]), 1),
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
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days + 5)
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            df = _yfdownload(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 5:
                continue
            df = df.dropna(subset=["Close", "Volume"])
            vwap = (_to_series(df["Close"]) * _to_series(df["Volume"])).sum() / _to_series(df["Volume"]).sum()
            current_price = float(_to_series(df["Close"]).iloc[-1])
            deviation = (current_price - float(vwap)) / float(vwap) * 100
            results.append({
                "企業名": name, "業種": sector, "ティッカー": ticker,
                "現在値": round(current_price, 1),
                "VWAP": round(float(vwap), 1),
                "VWAP乖離率(%)": round(deviation, 2),
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
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days + 10)
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            df = _yfdownload(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 5:
                continue
            price_chg = (_to_series(df["Close"]).iloc[-1] - _to_series(df["Close"]).iloc[0]) / _to_series(df["Close"]).iloc[0] * 100
            vol_chg   = (_to_series(df["Volume"]).iloc[-5:].mean() - _to_series(df["Volume"]).iloc[:5].mean()) / (_to_series(df["Volume"]).iloc[:5].mean() + 1) * 100
            results.append({
                "企業名": name, "業種": sector,
                "株価騰落率(%)": round(float(price_chg), 2),
                "出来高変化率(%)": round(float(vol_chg), 2),
            })
        except Exception:
            continue
    return pd.DataFrame(results)


def plot_pv_scatter(df: pd.DataFrame) -> None:
    """Price x Volume 散布図（Plotly・ホバーで銘柄名表示）"""
    import plotly.express as px

    x_max = df["出来高変化率(%)"].max()
    x_min = df["出来高変化率(%)"].min()
    y_max = df["株価騰落率(%)"].max()
    y_min = df["株価騰落率(%)"].min()

    fig = px.scatter(
        df,
        x="出来高変化率(%)",
        y="株価騰落率(%)",
        color="業種",
        hover_name="企業名",
        hover_data={
            "業種": True,
            "株価騰落率(%)": ":.2f",
            "出来高変化率(%)": ":.2f",
        },
        title="Price x Volume マップ（セクター別）― ホバーで銘柄名表示",
        height=600,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

    fig.update_layout(
        annotations=[
            dict(x=x_max * 0.7, y=y_max * 0.85, text="株高+出来高増<br>（本命上昇）",
                 showarrow=False, font=dict(color="#388e3c", size=11)),
            dict(x=x_min * 0.7, y=y_max * 0.85, text="株高+出来高減<br>（戻り弱い）",
                 showarrow=False, font=dict(color="#f57c00", size=11)),
            dict(x=x_max * 0.7, y=y_min * 0.85, text="株安+出来高増<br>（売り圧力）",
                 showarrow=False, font=dict(color="#d32f2f", size=11)),
            dict(x=x_min * 0.7, y=y_min * 0.85, text="株安+出来高減<br>（静かな下落）",
                 showarrow=False, font=dict(color="#9e9e9e", size=11)),
        ],
        xaxis_title="出来高変化率 (%)",
        yaxis_title="株価騰落率 (%)",
        legend=dict(orientation="v", x=1.02, y=1, font=dict(size=10)),
        margin=dict(r=150),
    )

    st.plotly_chart(fig, use_container_width=True)


# ================================================================
# 📊 価格パターン系モジュール
# ================================================================

@st.cache_data(ttl=1800)
def get_52week_highlow(ticker_name_map: dict) -> pd.DataFrame:
    from datetime import timedelta
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=365)
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            df = _yfdownload(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 50:
                continue
            high_52w = float(_to_series(df["High"]).max())
            low_52w  = float(_to_series(df["Low"]).min())
            current  = float(_to_series(df["Close"]).iloc[-1])
            from_high = (current - high_52w) / high_52w * 100
            from_low  = (current - low_52w)  / low_52w  * 100
            is_new_high = float(_to_series(df["High"]).iloc[-1]) >= high_52w * 0.995
            is_new_low  = float(_to_series(df["Low"]).iloc[-1])  <= low_52w  * 1.005
            results.append({
                "企業名": name, "業種": sector,
                "現在値": round(current, 1),
                "52週高値": round(high_52w, 1),
                "52週安値": round(low_52w, 1),
                "高値からの乖離(%)": round(from_high, 2),
                "安値からの乖離(%)": round(from_low, 2),
                "新高値": "新高値" if is_new_high else "",
                "新安値": "新安値" if is_new_low else "",
            })
        except Exception:
            continue
    return pd.DataFrame(results)


@st.cache_data(ttl=1800)
def get_ma_deviation(ticker_name_map: dict) -> pd.DataFrame:
    from datetime import timedelta
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=250)
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            df = _yfdownload(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 200:
                continue
            close   = _to_series(df["Close"]).dropna()
            current = float(close.iloc[-1])
            ma25    = float(close.rolling(25).mean().iloc[-1])
            ma75    = float(close.rolling(75).mean().iloc[-1])
            ma200   = float(close.rolling(200).mean().iloc[-1])
            results.append({
                "企業名": name, "業種": sector,
                "現在値": round(current, 1),
                "25日MA乖離(%)": round((current - ma25) / ma25 * 100, 2),
                "75日MA乖離(%)": round((current - ma75) / ma75 * 100, 2),
                "200日MA乖離(%)": round((current - ma200) / ma200 * 100, 2),
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
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=120)
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            df = _yfdownload(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 75:
                continue
            close = _to_series(df["Close"]).dropna()
            ma25  = close.rolling(25).mean()
            ma75  = close.rolling(75).mean()
            diff  = ma25 - ma75
            for i in range(max(1, len(diff) - lookback_days), len(diff)):
                if pd.isna(diff.iloc[i]) or pd.isna(diff.iloc[i-1]):
                    continue
                if diff.iloc[i-1] < 0 and diff.iloc[i] >= 0:
                    results.append({
                        "企業名": name, "業種": sector,
                        "シグナル": "ゴールデンクロス",
                        "発生日": str(diff.index[i])[:10],
                        "現在値": round(float(close.iloc[-1]), 1),
                    })
                    break
                elif diff.iloc[i-1] > 0 and diff.iloc[i] <= 0:
                    results.append({
                        "企業名": name, "業種": sector,
                        "シグナル": "デッドクロス",
                        "発生日": str(diff.index[i])[:10],
                        "現在値": round(float(close.iloc[-1]), 1),
                    })
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
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days)
    dow_map    = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金"}
    sector_dow: dict = {}
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            df = _yfdownload(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 20:
                continue
            ret = _to_series(df["Close"]).pct_change().dropna() * 100
            ret.index = pd.to_datetime(ret.index)
            for dow_num, dow_label in dow_map.items():
                avg = float(ret[ret.index.dayofweek == dow_num].mean())
                key = (sector, dow_label)
                if key not in sector_dow:
                    sector_dow[key] = []
                sector_dow[key].append(avg)
        except Exception:
            continue
    rows = []
    for (sector, dow), vals in sector_dow.items():
        rows.append({"業種": sector, "曜日": dow, "平均リターン(%)": round(np.mean(vals), 4)})
    df_long = pd.DataFrame(rows)
    if df_long.empty:
        return df_long
    df_pivot = df_long.pivot(index="業種", columns="曜日", values="平均リターン(%)")
    dow_order = ["月", "火", "水", "木", "金"]
    df_pivot  = df_pivot.reindex(columns=[d for d in dow_order if d in df_pivot.columns])
    return df_pivot


@st.cache_data(ttl=1800)
def get_correlation_divergence(ticker_name_map: dict, days: int = 60,
                                corr_window: int = 20) -> pd.DataFrame:
    from datetime import timedelta
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days + 10)
    benchmark = _yfdownload("^N225", start=start_date, end=end_date, progress=False)
    if isinstance(benchmark.columns, pd.MultiIndex):
        benchmark.columns = benchmark.columns.droplevel(1)
    market_ret = benchmark["Close"].pct_change().dropna()
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            df = _yfdownload(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < corr_window + 5:
                continue
            ret = _to_series(df["Close"]).pct_change().dropna()
            common = ret.index.intersection(market_ret.index)
            if len(common) < corr_window + 5:
                continue
            r = ret.loc[common]
            m = market_ret.loc[common]
            corr_long   = float(r.corr(m))
            corr_recent = float(r.iloc[-corr_window:].corr(m.iloc[-corr_window:]))
            divergence  = corr_long - corr_recent
            price_chg   = (_to_series(df["Close"]).iloc[-1] - _to_series(df["Close"]).iloc[-5]) / _to_series(df["Close"]).iloc[-5] * 100
            results.append({
                "企業名": name, "業種": sector,
                "長期相関": round(corr_long, 3),
                "直近相関": round(corr_recent, 3),
                "相関乖離度": round(divergence, 3),
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
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=30)
    results = []
    for ticker, (name, sector) in ticker_name_map.items():
        try:
            df = _yfdownload(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 10:
                continue
            price_chg = (_to_series(df["Close"]).iloc[-1] - _to_series(df["Close"]).iloc[0]) / _to_series(df["Close"]).iloc[0] * 100
            vol_chg   = (_to_series(df["Volume"]).iloc[-5:].mean() - _to_series(df["Volume"]).mean()) / (_to_series(df["Volume"]).mean() + 1) * 100
            score = float(price_chg) * np.log1p(max(float(vol_chg), 0) / 100 + 1)
            results.append({
                "企業名": name, "業種": sector,
                "モメンタムスコア": round(score, 3),
                "株価騰落率(%)": round(float(price_chg), 2),
                "出来高変化率(%)": round(float(vol_chg), 2),
                "現在値": round(float(_to_series(df["Close"]).iloc[-1]), 1),
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
    st.header("🔧 需給スクリーナー設定")
    surge_ratio = st.slider("出来高急増の閾値（倍）", 1.5, 5.0, 2.0, 0.5)
    pv_days     = st.selectbox("Price x Volume 期間", [10, 20, 60], index=1,
                                format_func=lambda x: f"{x}日")
    st.divider()
    st.header("🔄 セクター設定")
    rotation_period = st.selectbox(
        "セクター分析期間",
        options=[5, 10, 20, 60, 90], index=2,
        format_func=lambda x: {5:"1週間",10:"2週間",20:"1ヶ月",60:"3ヶ月",90:"約半年"}[x],
    )
    top_bottom_n = st.slider("上位・下位 表示セクター数", 3, 8, 5)
    st.divider()
    st.header("📈 価格パターン設定")
    corr_window = st.slider("相関分析ウィンドウ（日）", 20, 120, 60)
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
@st.cache_data(ttl=3600, show_spinner=False)
def get_price(ticker, start, end):
    return _yfdownload(ticker, start=start, end=end)

@st.cache_data(ttl=3600)
def get_benchmark(start, end):
    return _yfdownload("^N225", start=start, end=end)


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

def _plot_candlestick(df, title):
    if df.empty:
        st.warning("データなし"); return
    open_col  = next((c for c in df.columns if c.lower() in ["open","openingprice"]), None)
    high_col  = next((c for c in df.columns if c.lower() in ["high","highprice"]), None)
    low_col   = next((c for c in df.columns if c.lower() in ["low","lowprice"]), None)
    close_col = next((c for c in df.columns if c.lower() in ["close","closeprice"]), None)
    if not all([open_col, high_col, low_col, close_col]):
        st.dataframe(df.tail(20)); return
    fig, ax = plt.subplots(figsize=(12,4))
    for _, row in df.tail(60).iterrows():
        o,h,l,c = row[open_col],row[high_col],row[low_col],row[close_col]
        color = "#1a7f37" if c>=o else "#d1242f"
        ax.plot([row["Date"],row["Date"]], [l,h], color=color, linewidth=0.8)
        ax.bar(row["Date"], abs(c-o), bottom=min(o,c), color=color, alpha=0.85, width=1.2)
    ax.set_title(title, fontsize=11); ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25); plt.xticks(rotation=45); plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


# ================================================================
# メインコンテンツ（全セクション自動実行・1ページ表示）
# ================================================================

end_date   = datetime.today()
start_date = end_date - relativedelta(years=int(years))
top_n_int  = int(top_n)

# ─────────────────────────────────────────────────────────────────
# § 1. パフォーマンス分析
# ─────────────────────────────────────────────────────────────────
st.header("📊 パフォーマンス分析")

with st.spinner("市場データ（日経225）を取得中..."):
    benchmark = get_benchmark(start_date, end_date)

if benchmark.empty:
    st.error("市場データ（日経225）取得失敗。しばらく待って再読み込みしてください。")
    st.caption("yfinance のレート制限または一時的な接続エラーの可能性があります。")
else:
    # Close列を確実に1次元Seriesに変換
    _bench_close = benchmark["Close"]
    if isinstance(_bench_close, pd.DataFrame):
        _bench_close = _bench_close.iloc[:, 0]
    market_returns = _bench_close.pct_change().dropna()
    results = []
    progress    = st.progress(0)
    status_text = st.empty()
    for i, (ticker, (name, sector)) in enumerate(ticker_name_map.items()):
        status_text.text(f"取得中: {name} ({ticker})")
        df = get_price(ticker, start_date, end_date)
        progress.progress((i + 1) / len(ticker_name_map))
        if df.empty:
            continue
        # Close列を確実に1次元Seriesに変換
        close = _to_series(df["Close"])
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        returns = close.pct_change().dropna()
        common  = returns.index.intersection(market_returns.index)
        if len(common) < 30:
            continue
        x = np.array(returns.loc[common], dtype=float).flatten()
        y = np.array(market_returns.loc[common], dtype=float).flatten()
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
            continue
        annual_return = x.mean() * 252
        annual_vol    = x.std() * np.sqrt(252)
        if annual_vol == 0:
            continue
        try:
            beta = np.cov(x, y)[0][1] / np.var(y)
        except Exception:
            beta = 0.0
        sharpe = (annual_return - risk_free_rate) / annual_vol
        results.append({
            "企業名": name, "業種": sector,
            "年間平均リターン(%)": annual_return * 100,
            "年間リスク(%)": annual_vol * 100,
            "シャープレシオ": sharpe, "ベータ": beta,
        })
    progress.empty()
    status_text.empty()

    if results:
        df_results = pd.DataFrame(results).sort_values("シャープレシオ", ascending=False)
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("分析銘柄数", f"{len(df_results)}社")
        col_m2.metric("平均シャープレシオ", f"{df_results['シャープレシオ'].mean():.2f}")
        col_m3.metric("平均年間リターン", f"{df_results['年間平均リターン(%)'].mean():.1f}%")
        st.subheader(f"📋 上位{top_n_int}銘柄")
        st.dataframe(
            df_results.head(top_n_int).style.format({
                "年間平均リターン(%)": "{:.2f}",
                "年間リスク(%)": "{:.2f}",
                "シャープレシオ": "{:.2f}",
                "ベータ": "{:.2f}",
            }),
            use_container_width=True,
        )

st.divider()

# ─────────────────────────────────────────────────────────────────
# § 2. セクターローテーション
# ─────────────────────────────────────────────────────────────────
st.header("🔄 セクターローテーション")
st.caption("各業種に属する銘柄の平均リターンを集計し、資金が流入・流出しているセクターを可視化します。")

with st.spinner(f"セクターデータ取得中（{len(ticker_name_map)}銘柄）..."):
    df_sector = get_sector_performance(ticker_name_map, period_days=rotation_period)

if not df_sector.empty:
    top_sec    = df_sector.iloc[0]
    bottom_sec = df_sector.iloc[-1]
    rising  = (df_sector["平均リターン(%)"] > 0).sum()
    falling = (df_sector["平均リターン(%)"] < 0).sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📈 最強セクター",  top_sec["業種"],    f"{top_sec['騰落率(%)']:+.2f}%")
    k2.metric("📉 最弱セクター",  bottom_sec["業種"], f"{bottom_sec['騰落率(%)']:+.2f}%")
    k3.metric("🟢 上昇セクター数", f"{rising} 業種")
    k4.metric("🔴 下落セクター数", f"{falling} 業種")
    period_label = {5:"1週間",10:"2週間",20:"1ヶ月",60:"3ヶ月",90:"約半年"}[rotation_period]
    fig_bar = plot_sector_bar(df_sector, title=f"セクター別平均リターン（{period_label}）")
    st.pyplot(fig_bar); plt.close(fig_bar)

st.divider()

# ─────────────────────────────────────────────────────────────────
# § 3. 需給スクリーナー
# ─────────────────────────────────────────────────────────────────
st.header("🔥 需給スクリーナー")

with st.spinner("出来高データ取得中..."):
    df_surge = get_volume_surge(ticker_name_map, surge_ratio=surge_ratio)

if df_surge.empty:
    st.info(f"出来高が{surge_ratio}倍以上の銘柄は現在なし")
else:
    st.success(f"🔺 {len(df_surge)} 銘柄検出")
    def color_surge(val):
        if isinstance(val, float):
            if val >= 3:  return "background-color: #d32f2f; color: white; font-weight:bold"
            elif val >= 2: return "background-color: #f57c00; color: white; font-weight:bold"
        return ""
    st.dataframe(
        df_surge.style.format({"出来高倍率":"{:.2f}x","株価変化率(5日%)":"{:+.2f}"})
                      .applymap(color_surge, subset=["出来高倍率"]),
        use_container_width=True,
    )

with st.spinner("VWAPデータ計算中..."):
    df_vwap = get_vwap_deviation(ticker_name_map)

if not df_vwap.empty:
    col_up, col_down = st.columns(2)
    with col_up:
        st.markdown("#### 🔴 割高（VWAP上方乖離 上位10）")
        st.dataframe(df_vwap[df_vwap["VWAP乖離率(%)"] > 0].head(10)
                     .style.format({"VWAP乖離率(%)":"{:+.2f}"}), use_container_width=True)
    with col_down:
        st.markdown("#### 🟢 割安（VWAP下方乖離 下位10）")
        st.dataframe(df_vwap[df_vwap["VWAP乖離率(%)"] < 0].tail(10)
                     .sort_values("VWAP乖離率(%)")
                     .style.format({"VWAP乖離率(%)":"{:+.2f}"}), use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────
# § 4. 価格パターン
# ─────────────────────────────────────────────────────────────────
st.header("📈 価格パターン")

with st.spinner("52週高安値データ取得中..."):
    df_52w = get_52week_highlow(ticker_name_map)

if not df_52w.empty:
    col_h, col_l = st.columns(2)
    with col_h:
        st.markdown("#### 🏔️ 52週高値圏（高値比-5%以内）上位20")
        df_near_high = df_52w[df_52w["高値乖離(%)"] >= -5].sort_values("高値乖離(%)", ascending=False).head(20)
        st.dataframe(df_near_high.style.format({"高値乖離(%)":"{:+.2f}","安値乖離(%)":"{:+.2f}","直近株価":"{:.0f}"}), use_container_width=True)
    with col_l:
        st.markdown("#### 🕳️ 52週安値圏（安値比+10%以内）上位20")
        df_near_low = df_52w[df_52w["安値乖離(%)"] <= 10].sort_values("安値乖離(%)").head(20)
        st.dataframe(df_near_low.style.format({"高値乖離(%)":"{:+.2f}","安値乖離(%)":"{:+.2f}","直近株価":"{:.0f}"}), use_container_width=True)

with st.spinner("移動平均データ計算中..."):
    try:
        _ma_results = []
        for ticker, (name, sector) in list(ticker_name_map.items())[:50]:  # 先頭50銘柄
            df_tmp = get_price(ticker, start_date, end_date)
            if df_tmp.empty or len(df_tmp) < 75:
                continue
            close = df_tmp["Close"]
            ma25  = close.rolling(25).mean().iloc[-1]
            ma75  = close.rolling(75).mean().iloc[-1]
            last  = close.iloc[-1]
            if pd.isna(ma25) or pd.isna(ma75) or ma25 == 0:
                continue
            _ma_results.append({
                "企業名": name, "業種": sector,
                "直近株価": last,
                "MA25乖離(%)": (last - ma25) / ma25 * 100,
                "MA75乖離(%)": (last - ma75) / ma75 * 100,
            })
        df_ma = pd.DataFrame(_ma_results)
    except Exception:
        df_ma = pd.DataFrame()

if not df_ma.empty:
    st.markdown("#### 📊 移動平均乖離ランキング（25日MA）")
    col_ma1, col_ma2 = st.columns(2)
    with col_ma1:
        st.markdown("**🔴 上方乖離上位15（割高・過熱）**")
        st.dataframe(df_ma.nlargest(15,"MA25乖離(%)").style.format({"MA25乖離(%)":"{:+.2f}","MA75乖離(%)":"{:+.2f}"}), use_container_width=True)
    with col_ma2:
        st.markdown("**🟢 下方乖離上位15（割安・反発期待）**")
        st.dataframe(df_ma.nsmallest(15,"MA25乖離(%)").style.format({"MA25乖離(%)":"{:+.2f}","MA75乖離(%)":"{:+.2f}"}), use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────
# § 5. モメンタム・相関分析
# ─────────────────────────────────────────────────────────────────
st.header("💡 モメンタム・相関分析")

with st.spinner("モメンタムスコア計算中..."):
    df_mom = get_momentum_score(ticker_name_map)

if not df_mom.empty:
    top10  = df_mom.head(10)[["企業名","業種","モメンタムスコア"]].to_string(index=False)
    bot10  = df_mom.tail(10)[["企業名","業種","モメンタムスコア"]].to_string(index=False)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### 🚀 高モメンタム上位10")
        st.dataframe(df_mom.head(10).style.format({"モメンタムスコア":"{:.3f}"}), use_container_width=True)
    with col_m2:
        st.markdown("#### 🐢 低モメンタム下位10")
        st.dataframe(df_mom.tail(10).style.format({"モメンタムスコア":"{:.3f}"}), use_container_width=True)

with st.spinner("日経平均との相関分析中..."):
    df_corr = get_correlation_divergence(ticker_name_map, corr_window=corr_window)

if not df_corr.empty:
    st.markdown("#### 🔍 日経平均との相関崩れ検知")
    st.caption("相関乖離度が高い = 最近、日経と独自の動きをしている銘柄（個別材料の可能性）")
    col_div1, col_div2 = st.columns(2)
    with col_div1:
        st.markdown("**🟡 相関崩れ上位15（独自上昇）**")
        rising_div = df_corr[df_corr["直近5日株価変化(%)"] > 0].head(15)
        st.dataframe(rising_div.style.format({"長期相関":"{:.3f}","直近相関":"{:.3f}","相関乖離度":"{:.3f}","直近5日株価変化(%)":"{:+.2f}"}), use_container_width=True)
    with col_div2:
        st.markdown("**🔴 相関崩れ上位15（独自下落）**")
        falling_div = df_corr[df_corr["直近5日株価変化(%)"] < 0].head(15)
        st.dataframe(falling_div.style.format({"長期相関":"{:.3f}","直近相関":"{:.3f}","相関乖離度":"{:.3f}","直近5日株価変化(%)":"{:+.2f}"}), use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────
# § 6. J-Quants 需給分析
# ─────────────────────────────────────────────────────────────────
st.header("🏦 J-Quants 需給分析")

jq_key = st.secrets.get("JQUANTS_API_KEY", "")
if not jq_key:
    st.warning("⚠️ J-Quants APIキー未設定（`JQUANTS_API_KEY` を Secrets に追加してください）")
else:
    jq_code_input = st.sidebar.text_input("🏦 J-Quants 銘柄コード", value="72030", key="jq_code")
    jq_code  = jq_code_input.strip()
    jq_period = st.sidebar.selectbox("J-Quants 期間", ["3ヶ月","6ヶ月","1年"], index=1, key="jq_period")
    period_days_jq = {"3ヶ月":90,"6ヶ月":180,"1年":365}[jq_period]
    jq_date_to   = end_date.strftime("%Y%m%d")
    jq_date_from = (end_date - relativedelta(days=period_days_jq)).strftime("%Y%m%d")

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
                _plot_candlestick(df_bars, f"{jq_code} 株価")
        with col_p2:
            with st.spinner("TOPIX取得中..."):
                df_topix = jq_fetch_topix(jq_date_from, jq_date_to)
            if not df_topix.empty:
                close_col = next((c for c in df_topix.columns if c.lower() in ["close","closeprice"]), None)
                if close_col:
                    fig_t, ax_t = plt.subplots(figsize=(6,4))
                    ax_t.plot(df_topix["Date"], df_topix[close_col], color="#1565c0", linewidth=1.5)
                    ax_t.fill_between(df_topix["Date"], df_topix[close_col], df_topix[close_col].min(), alpha=0.1, color="#1565c0")
                    ax_t.set_title("TOPIX", fontsize=11)
                    ax_t.grid(True, alpha=0.25)
                    plt.xticks(rotation=45); plt.tight_layout()
                    st.pyplot(fig_t, clear_figure=True)

    with jq_t2:
        st.caption("Lightプラン以上が必要")
        with st.spinner("投資部門別データ取得中..."):
            df_inv = jq_fetch_investor_types(jq_date_from, jq_date_to)
        if df_inv.empty:
            st.warning("データ取得失敗（Lightプラン以上が必要）")
        else:
            section_col = next((c for c in df_inv.columns if "section" in c.lower()), None)
            buy_col  = next((c for c in df_inv.columns if "buy" in c.lower()), None)
            sell_col = next((c for c in df_inv.columns if "sell" in c.lower()), None)
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

    with jq_t3:
        st.caption("Standardプラン以上が必要")
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
                st.metric("最新 信用倍率", f"{ratio.iloc[-1]:.2f}倍",
                          delta=f"{ratio.iloc[-1]-ratio.iloc[-2]:+.2f}" if len(ratio)>1 else None)
            else:
                st.dataframe(df_mg, use_container_width=True)

    with jq_t4:
        st.caption("Standardプラン以上が必要")
        S33_OPTIONS = {
            "3650 電気機器":"3650","3700 輸送用機器":"3700","5250 情報・通信":"5250",
            "7050 銀行":"7050","3200 化学":"3200","3600 機械":"3600",
            "6100 小売":"6100","8050 不動産":"8050","9050 サービス":"9050",
        }
        selected_s33_label = st.selectbox("業種コード", list(S33_OPTIONS.keys()), key="jq_s33_auto")
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
                latest = float(df_sr[ratio_col].iloc[-1])*100
                avg    = float(df_sr[ratio_col].mean())*100
                c1, c2 = st.columns(2)
                c1.metric("最新空売り比率", f"{latest:.1f}%")
                c2.metric("期間平均", f"{avg:.1f}%", delta=f"{latest-avg:+.1f}%")

    with jq_t5:
        st.caption("Freeプラン以上で利用可能")
        with st.spinner("財務情報取得中..."):
            df_fins = jq_fetch_fins(jq_code)
        if df_fins.empty:
            st.warning("財務データ取得失敗")
        else:
            key_cols = [c for c in df_fins.columns if any(k in c.lower() for k in
                ["date","period","sales","profit","income","eps","revenue","operating","net","equity"])]
            st.dataframe(df_fins[key_cols].tail(8) if key_cols else df_fins.tail(8),
                         use_container_width=True)

st.divider()
st.caption("データソース: Yahoo Finance / J-Quants / TDnet / 株探 / みんかぶ / 日経 / Reuters | 投資判断は自己責任で")
