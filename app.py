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
GROQ_MODEL   = "llama3-70b-8192"

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
    """Gemini → Groq フォールバック"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text, "Gemini"
    except Exception as e:
        err_str = str(e)
        is_quota = "429" in err_str or "quota" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str
        if not is_quota:
            raise
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


# ── ③ みんかぶ 銘柄別ニュース（RSS使用） ────────────────────────
@st.cache_data(ttl=600)
def fetch_minkabu_news(ticker_code: str, max_items: int = 6) -> list[dict]:
    """
    みんかぶの銘柄ニュース。
    みんかぶは /stock/{code}/news ページで銘柄固有のニュースを提供。
    """
    code = ticker_code.replace(".T", "")
    url = f"https://minkabu.jp/stock/{code}/news"
    try:
        r = requests.get(url, headers=_NEWS_HEADERS, timeout=12)
        if r.status_code != 200:
            return []
        html = r.text

        # みんかぶの銘柄ニュース構造:
        # <li class="news_list_item"> ... <a href="/news/...">タイトル</a>
        # または <a href="/stock/XXXX/news/XXXXX">
        pattern = re.compile(
            r'<a\s+href="((?:/stock/' + code + r'/news/|/news/)[^"]+)"[^>]*>\s*([^<]{4,120})\s*</a>',
        )
        matches = pattern.findall(html)

        # 日付抽出
        dates = re.findall(r'(\d{4}/\d{2}/\d{2}|\d{2}/\d{2}\s+\d{2}:\d{2})', html)

        items = []
        seen = set()
        for i, (path, title) in enumerate(matches[:max_items * 2]):
            title = title.strip()
            if len(title) < 4 or title in seen:
                continue
            # ナビゲーション系を除外
            if any(kw in title for kw in ["ログイン", "会員登録", "みんかぶ", "詳しく見る"]):
                continue
            seen.add(title)
            link = f"https://minkabu.jp{path}" if path.startswith("/") else path
            date = dates[i] if i < len(dates) else ""
            items.append({
                "source": "みんかぶ",
                "title": title,
                "link": link,
                "date": date,
                "summary": "",
                "ticker_specific": True,
            })
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


# ── ④ TDnet（適時開示）銘柄別 ─────────────────────────────────
@st.cache_data(ttl=900)
def fetch_tdnet_news(ticker_code: str, max_items: int = 6) -> list[dict]:
    """
    EDINET/JPXが提供するTDnet開示情報。
    JPXの適時開示情報閲覧サービスで銘柄コード指定検索を使用。
    URL: https://www.release.tdnet.info/inbs/I_list_001_{date}.html
    銘柄コードでフィルタリング。
    """
    code = ticker_code.replace(".T", "")
    today = datetime.today().strftime("%Y%m%d")

    # 当日の開示一覧から銘柄コードで絞る
    url = f"https://www.release.tdnet.info/inbs/I_list_001_{today}.html"
    try:
        r = requests.get(url, headers={
            **_NEWS_HEADERS,
            "Host": "www.release.tdnet.info",
            "Referer": "https://www.release.tdnet.info/",
        }, timeout=15)
        if r.status_code != 200:
            # 前日も試す
            import datetime as dt
            yesterday = (dt.date.today() - dt.timedelta(days=1)).strftime("%Y%m%d")
            r = requests.get(
                f"https://www.release.tdnet.info/inbs/I_list_001_{yesterday}.html",
                headers=_NEWS_HEADERS, timeout=15
            )
            if r.status_code != 200:
                return []

        html = r.text

        # TDnetの表構造: <td>証券コード</td><td>会社名</td><td>開示タイトル</td>
        # コードでマッチする行を探す
        # 行単位でパース: <tr>...</tr> の中に code が含まれるものを抽出
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
        items = []
        for row in rows:
            # 証券コードセルを確認
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            clean_cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            if not any(code in c for c in clean_cells):
                continue
            # PDFリンク取得
            pdf_match = re.search(r'href="([^"]+\.pdf)"', row)
            # タイトル取得（証券コードの次のセルあたり）
            title = ""
            for j, c in enumerate(clean_cells):
                if code in c and j + 2 < len(clean_cells):
                    title = clean_cells[j + 2]  # コード→会社名→タイトルの順
                    break
            if not title:
                # クラス名 kjTitle のセルを探す
                title_match = re.search(r'class="[^"]*kjTitle[^"]*"[^>]*>(.*?)</td>', row, re.DOTALL)
                if title_match:
                    title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
            # 日時
            time_match = re.search(r'(\d{2}:\d{2})', row)
            time_str = time_match.group(1) if time_match else ""

            if not title or len(title) < 2:
                continue
            link = ""
            if pdf_match:
                pdf_path = pdf_match.group(1)
                link = f"https://www.release.tdnet.info{pdf_path}" if pdf_path.startswith("/") else pdf_path

            items.append({
                "source": "TDnet（適時開示）",
                "title": title,
                "link": link,
                "date": f"本日 {time_str}" if time_str else "本日",
                "summary": "📄 適時開示PDF",
                "ticker_specific": True,
            })
            if len(items) >= max_items:
                break
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


# ── ⑦ 統合ニュース取得（銘柄別）────────────────────────────────
def fetch_all_news(
    ticker_code: str,
    company_name: str,
    max_per_source: int = 5,
) -> list[dict]:
    """
    全ニュースソースを並列取得。
    - 銘柄固有ソース（Yahoo!JP / 株探 / みんかぶ / TDnet）: そのまま使用
    - 全体ニュース（日経 / Reuters）: 銘柄名・コードを含む記事のみ残す
    返り値: [{source, title, link, date, summary, ticker_specific}, ...]
    """
    import concurrent.futures
    code = ticker_code.replace(".T", "")

    tasks = {
        "yahoo_jp": lambda: fetch_yahoo_jp_news(code, max_per_source),
        "kabutan":  lambda: fetch_kabutan_news(code, max_per_source),
        "minkabu":  lambda: fetch_minkabu_news(code, max_per_source),
        "tdnet":    lambda: fetch_tdnet_news(code, max_per_source),
        "nikkei":   lambda: fetch_nikkei_market_rss(max_per_source * 3),  # 多め取得してフィルタ
        "reuters":  lambda: fetch_reuters_jp_rss(max_per_source * 3),
    }

    # 銘柄マッチ用キーワード（コード・会社名の一部）
    # 会社名の括弧・特殊文字を除いたシンプルな形にする
    company_short = re.sub(r"[　ＨＤ（）()ホールディングス]", "", company_name)[:4]
    match_keywords = {code, company_name, company_short}
    match_keywords = {k for k in match_keywords if len(k) >= 2}

    all_items = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(fn): key for key, fn in tasks.items()}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                results = future.result()
                # 全体ニュース（日経・Reuters）は銘柄関連記事のみ残す
                if key in ("nikkei", "reuters"):
                    results = [
                        item for item in results
                        if any(kw in item.get("title", "") for kw in match_keywords)
                    ]
                    for item in results:
                        item["ticker_specific"] = False
                all_items.extend(results)
            except Exception:
                pass

    # 重複除去
    seen, unique = set(), []
    for item in all_items:
        key = item["title"][:30]
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # ソート: 銘柄固有を先頭、日時の新しい順
    unique.sort(key=lambda x: (not x.get("ticker_specific", True), x.get("date", "")), reverse=False)
    unique.sort(key=lambda x: not x.get("ticker_specific", True))

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
        with st.spinner(f"{selected_name}（{selected_ticker}）のニュースを全ソースから取得中..."):
            all_news = fetch_all_news(selected_ticker, selected_name, news_max_per_source)

        # フィルタリング（サイドバーで選択したソースのみ）
        filtered = [n for n in all_news if n["source"] in show_news_sources] if show_news_sources else all_news

        source_colors = {
            "Yahoo!Finance JP":  "🟦",
            "株探(Kabutan)":     "🟩",
            "みんかぶ":          "🟨",
            "TDnet（適時開示）": "🟥",
            "日経新聞":          "⬛",
            "Reuters JP":        "🟫",
        }

        if not filtered:
            st.warning("ニュースが取得できませんでした")
            st.info(
                "**考えられる原因:**\n"
                f"- {selected_name}（{selected_ticker}）の最新ニュースが各ソースに存在しない\n"
                "- サイドバーの「表示するニュースソース」で絞り込みすぎている\n"
                "- スクレイピング先のサイト構造が変更された"
            )
        else:
            # 銘柄固有 / 市場全体 の内訳を表示
            ticker_specific = [n for n in filtered if n.get("ticker_specific", True)]
            market_wide     = [n for n in filtered if not n.get("ticker_specific", True)]

            col_a, col_b = st.columns(2)
            col_a.metric("📌 銘柄固有ニュース", f"{len(ticker_specific)}件",
                         help="Yahoo!Finance/株探/みんかぶ/TDnetの銘柄ページから取得")
            col_b.metric("🌐 市場全体（銘柄言及あり）", f"{len(market_wide)}件",
                         help="日経・Reutersから銘柄名・コードを含む記事のみ抽出")

            # ソース別集計
            from collections import Counter
            src_counts = Counter(n["source"] for n in filtered)
            cols_stat  = st.columns(min(len(src_counts), 6))
            for i, (src, cnt) in enumerate(src_counts.items()):
                icon = source_colors.get(src, "⚪")
                cols_stat[i % len(cols_stat)].metric(f"{icon} {src}", f"{cnt}件")

            st.divider()

            # ── 銘柄固有ニュースを先に表示 ──
            if ticker_specific:
                st.markdown(f"#### 📌 {selected_name} 銘柄固有ニュース")
                for item in ticker_specific:
                    icon = source_colors.get(item["source"], "⚪")
                    badge = "🟥 **適時開示**" if item["source"] == "TDnet（適時開示）" else ""
                    title_short = item["title"][:70] + ("…" if len(item["title"]) > 70 else "")
                    with st.expander(f"{icon} [{item['source']}]　{title_short}"):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            badge_text = item.get("badge", "")
                            badge_map = {
                                "特集": "🟠 特集", "材料": "🟢 材料", "決算": "🔵 決算",
                                "開示": "🔴 開示", "テク": "⚪ テク", "速報": "🟡 速報",
                            }
                            badge_label = badge_map.get(badge_text, f"◾ {badge_text}" if badge_text else "")
                            if badge_label:
                                st.caption(badge_label)
                            st.markdown(f"**{item['title']}**")
                            if item.get("summary") and item["summary"] not in ("📄 適時開示PDF", ""):
                                if not item["summary"].startswith("["):
                                    st.caption(item["summary"])
                        with c2:
                            if item.get("date"):
                                st.caption(f"🕐 {item['date']}")
                            if item.get("link"):
                                st.markdown(f"[🔗 記事を開く]({item['link']})")
                            elif item.get("source") == "TDnet（適時開示）":
                                st.caption("（PDF直リンク取得中）")

            # ── 市場全体から銘柄言及あり ──
            if market_wide:
                st.markdown(f"#### 🌐 市場ニュース（{selected_name}に言及）")
                for item in market_wide:
                    icon = source_colors.get(item["source"], "⚪")
                    title_short = item["title"][:70] + ("…" if len(item["title"]) > 70 else "")
                    with st.expander(f"{icon} [{item['source']}]　{title_short}"):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"**{item['title']}**")
                        with c2:
                            if item.get("date"):
                                st.caption(f"🕐 {item['date']}")
                            if item.get("link"):
                                st.markdown(f"[🔗 記事を開く]({item['link']})")

            if not ticker_specific and not market_wide:
                st.info(f"現時点で {selected_name} に関するニュースは見つかりませんでした")

            # AI 分析
            if run_ai and filtered:
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
