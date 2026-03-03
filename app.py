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
GROQ_MODEL = "llama-3.1-70b-versatile"

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
        
    if groq_client is None:
        raise RuntimeError("Geminiクォータ超過 & GROQ_API_KEY 未設定")
    chat = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    )
    return chat.choices[0].message.content, "Groq"

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
    株探の銘柄ニュースページをスクレイピング。
    ticker_code: '7203' など
    """
    code = ticker_code.replace(".T", "")
    url = f"https://kabutan.jp/stock/news?code={code}"
    try:
        r = requests.get(url, headers=_NEWS_HEADERS, timeout=12)
        if r.status_code != 200:
            return []
        # タイトルと日付を正規表現で抽出
        # 株探の構造: <a href="/news/...">タイトル</a> と <time>日付</time>
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


# ── ③ みんかぶ 銘柄別ニュース ───────────────────────────────────
@st.cache_data(ttl=600)
def fetch_minkabu_news(ticker_code: str, max_items: int = 6) -> list[dict]:
    """みんかぶの銘柄ニュースを取得"""
    code = ticker_code.replace(".T", "")
    url = f"https://minkabu.jp/stock/{code}/news"
    try:
        r = requests.get(url, headers=_NEWS_HEADERS, timeout=12)
        if r.status_code != 200:
            return []
        # ニュースタイトルと日付の抽出
        titles = re.findall(
            r'<a[^>]+href="(/stock/[^"]+/news/[^"]+)"[^>]*>\s*<[^>]+>\s*([^<]{5,120})\s*</[^>]+>',
            r.text,
        )
        if not titles:
            # より広いパターン
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


# ── ④ TDnet（適時開示情報）銘柄別 ─────────────────────────────
@st.cache_data(ttl=900)
def fetch_tdnet_news(ticker_code: str, max_items: int = 6) -> list[dict]:
    """
    TDnet（東京証券取引所 適時開示情報）から銘柄の最新開示を取得。
    JPX の開示検索API（非公式）を使用。
    """
    code = ticker_code.replace(".T", "")
    url = (
        "https://www.release.tdnet.info/inbs/I_list_001_"
        f"{datetime.today().strftime('%Y%m%d')}.html"
    )
    # TDnet の検索エンドポイント（銘柄コード指定）
    search_url = f"https://www.release.tdnet.info/inbs/I_main_00.html?target-code={code}"
    try:
        r = requests.get(search_url, headers=_NEWS_HEADERS, timeout=12)
        if r.status_code != 200:
            return []
        # タイトルと PDF リンクを抽出
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


# ── ⑤ 日経新聞 マーケット RSS ───────────────────────────────────
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


# ── ⑥ Reuters Japan RSS ────────────────────────────────────────
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


# ── ⑦ 統合ニュース取得（銘柄別 + 全体）─────────────────────────
def fetch_all_news(ticker_code: str, max_per_source: int = 5) -> list[dict]:
    """
    全ニュースソースを並列取得してまとめる。
    返り値: [{source, title, link, date, summary}, ...]
    """
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

    # 重複除去（タイトルが完全一致するもの）
    seen, unique = set(), []
    for item in all_items:
        if item["title"] not in seen:
            seen.add(item["title"])
            unique.append(item)

    return unique


# ── ⑧ AI によるニュース要約・センチメント ───────────────────────
def ai_news_summary(news_items: list[dict], company_name: str, ticker: str) -> str:
    """ニュース一覧をAIで日本語要約・センチメント分析"""
    if not news_items:
        return "ニュースが取得できませんでした。"

    headlines = "\n".join(
        f"[{it['source']}] {it['title']}" for it in news_items[:15]
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
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 2:
                continue
            close = df["Close"].dropna()
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
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 5:
                continue
            close = df["Close"].dropna()
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
    ax.set_title("📈 買われているセクター（累積リターン）", fontsize=11, fontweight="bold")
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
    ax.set_title("📉 売られているセクター（累積リターン）", fontsize=11, fontweight="bold")
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
tab_analysis, tab_sector, tab_news, tab_market_news = st.tabs([
    "📊 パフォーマンス分析",
    "🔄 セクターローテーション",
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

# ─── Tab2: セクターローテーション ──────────────────────────────
with tab_sector:
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
        run_rotation = st.button("▶ セクターローテーション分析を実行", type="primary")

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
                title=f"セクター別平均リターン（{period_label}）― 買われ🟢 / 売られ🔴",
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
                    if val > 2:   return "background-color: rgba(56,142,60,0.45); color: white; font-weight:bold"
                    elif val > 0: return "color: #388e3c; font-weight:bold"
                    elif val < -2: return "background-color: rgba(211,47,47,0.45); color: white; font-weight:bold"
                    elif val < 0: return "color: #d32f2f; font-weight:bold"
                return ""

            styled = df_display.style.format({
                "平均リターン(%)": "{:+.2f}",
                "中央値リターン(%)": "{:+.2f}",
                "上昇率(%)": "{:.1f}",
            }).applymap(color_return, subset=["平均リターン(%)", "中央値リターン(%)"])
            st.dataframe(styled, use_container_width=True, height=500)

            st.divider()
            st.subheader("🤖 AIによるセクターローテーション解説")
            top5_str    = df_sector.head(5)[["業種", "騰落率(%)"]].to_string(index=False)
            bottom5_str = df_sector.tail(5)[["業種", "騰落率(%)"]].to_string(index=False)
            prompt_rotation = f"""
あなたは日本株の機関投資家向けストラテジストです。
以下は直近{period_label}のJPX上場主要銘柄のセクター別平均リターンです。

【買われているセクター（上位5）】
{top5_str}

【売られているセクター（下位5）】
{bottom5_str}

以下の観点で400文字以内で分析してください：
1. 現在のセクターローテーションの特徴（どんな相場環境を示唆するか）
2. 買われているセクターの背景・理由（マクロ要因・テーマ）
3. 売られているセクターの背景・理由
4. 投資家へのアドバイス（注目セクター・リスク）
"""
            with st.spinner("AI分析中..."):
                try:
                    comment, ai_name = generate_ai_comment(prompt_rotation)
                    st.info(f"{comment}\n\n_AI: {ai_name}_")
                except Exception as e:
                    st.warning(f"AI APIエラー: {e}")

    else:
        st.info(
            "「▶ セクターローテーション分析を実行」ボタンを押すと分析が始まります。\n\n"
            "**表示されるグラフ:**\n"
            "- 🟢🔴 セクター別平均リターン棒グラフ（買われ・売られ色分け）\n"
            "- 📈📉 上位・下位セクターの累積リターン時系列グラフ\n"
            "- 🌡️ 1週間 / 1ヶ月 / 3ヶ月 ヒートマップ（期間比較）\n"
            "- 📋 セクター別詳細テーブル（上昇銘柄数・上昇率など）\n"
            "- 🤖 AIによるローテーション解説とアドバイス"
        )
        st.caption("⚠️ 全銘柄データ取得のため、初回実行には数十秒かかる場合があります。結果は30分キャッシュされます。")
        
# ─── Tab2: 銘柄別ニュース ────────────────────────────────────────
with tab_news:
    st.subheader("📰 銘柄別ニュース・適時開示")

    # 銘柄選択
    ticker_options = {f"{name}（{t}）": t for t, (name, _) in ticker_name_map.items()}
    selected_label = st.selectbox("銘柄を選択", list(ticker_options.keys()),
                                  index=list(ticker_options.keys()).index("トヨタ（7203.T）") if "トヨタ（7203.T）" in ticker_options else 0)
    selected_ticker = ticker_options[selected_label]
    selected_name   = ticker_name_map[selected_ticker][0]

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_news = st.button("▶ ニュースを取得", type="primary")
    with col_btn2:
        run_ai   = st.checkbox("🤖 AIによる要約・センチメント分析も行う", value=True)

    if run_news:
        with st.spinner(f"{selected_name} のニュースを全ソースから取得中..."):
            all_news = fetch_all_news(selected_ticker, news_max_per_source)

        # フィルタリング（サイドバーで選択したソースのみ）
        filtered = [n for n in all_news if n["source"] in show_news_sources] if show_news_sources else all_news

        if not filtered:
            st.warning("ニュースが取得できませんでした（ソース設定を確認してください）")
        else:
            # ソース別に色分け表示
            source_colors = {
                "Yahoo!Finance JP":  "🟦",
                "株探(Kabutan)":     "🟩",
                "みんかぶ":          "🟨",
                "TDnet（適時開示）": "🟥",
                "日経新聞":          "⬛",
                "Reuters JP":        "🟫",
            }

            # ソース別集計
            from collections import Counter
            src_counts = Counter(n["source"] for n in filtered)
            cols_stat  = st.columns(len(src_counts))
            for i, (src, cnt) in enumerate(src_counts.items()):
                icon = source_colors.get(src, "⚪")
                cols_stat[i].metric(f"{icon} {src}", f"{cnt}件")

            st.divider()

            # ニュース一覧表示
            for item in filtered:
                icon = source_colors.get(item["source"], "⚪")
                with st.expander(f"{icon} [{item['source']}] {item['title'][:60]}{'…' if len(item['title'])>60 else ''}"):
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

            # AI 分析
            if run_ai:
                st.divider()
                st.subheader("🤖 AI ニュース分析（センチメント）")
                with st.spinner("AI分析中..."):
                    ai_result = ai_news_summary(filtered, selected_name, selected_ticker)
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
