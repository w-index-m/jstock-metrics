# -*- coding: utf-8 -*-
"""
analytics.py  — Market Dashboard 自前アクセス解析 v3
======================================================
v3の変更点:
  - JavaScriptでクライアント側からIPとUser-Agentを取得
  - ipinfo.io API でIPから地域情報を取得
  - st.query_params 経由でPythonに送信
  - Google Sheets に永続保存

これにより:
  country / city → 正確に取得可能
  device / browser → User-Agentから正確に取得
======================================================
"""

import os
import json
import logging
import datetime
import re
from datetime import timedelta
from typing import Dict, List, Tuple

import pytz
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import requests

logger = logging.getLogger(__name__)
JST = pytz.timezone("Asia/Tokyo")

_SESSION_KEY  = "_anl_session_id"
_TRACKED_KEY  = "_anl_tracked"
_PV_LOG_KEY   = "_anl_pv_log"
_UA_KEY       = "_anl_ua"
_IP_KEY       = "_anl_ip"
REALTIME_MIN  = 5


# ===========================
# ユーティリティ
# ===========================
def _secret(key: str, default: str = "") -> str:
    try:
        return str(st.secrets[key])
    except Exception:
        return os.getenv(key, default)


def _detect_backend() -> str:
    if _secret("GOOGLE_SHEETS_ID") and _secret("GOOGLE_SERVICE_ACCOUNT_JSON"):
        return "sheets"
    return "session"


def _session_id() -> str:
    if _SESSION_KEY not in st.session_state:
        import uuid
        st.session_state[_SESSION_KEY] = str(uuid.uuid4())[:12]
    return st.session_state[_SESSION_KEY]


# ===========================
# JavaScript でIP・UAを取得
# ===========================
def inject_client_info_collector():
    """
    JavaScriptでクライアントのIPとUser-Agentを取得し
    st.query_params 経由でPythonに送信。
    track_pageview() より前に呼び出す。
    """
    if st.session_state.get("_anl_client_collected"):
        return

    sid = _session_id()
    ipinfo_token = _secret("IPINFO_TOKEN", "")

    components.html(
        f"""
        <script>
        (function() {{
            var ipinfo_token = "{ipinfo_token}";
            var ipinfo_url = ipinfo_token
                ? "https://ipinfo.io/json?token=" + ipinfo_token
                : "https://ipinfo.io/json";

            var ua = encodeURIComponent(
                (navigator.userAgent || "").substring(0, 300)
            );

            fetch(ipinfo_url)
                .then(function(r) {{ return r.json(); }})
                .then(function(d) {{
                    var url = new URL(window.parent.location.href);
                    if (!url.searchParams.get("_anl_done")) {{
                        url.searchParams.set("_anl_done",    "1");
                        url.searchParams.set("_anl_country", encodeURIComponent(d.country || "??"));
                        url.searchParams.set("_anl_city",    encodeURIComponent(d.city    || "??"));
                        url.searchParams.set("_anl_ua",      ua);
                        url.searchParams.set("_anl_sid",     "{sid}");
                        window.parent.location.replace(url.toString());
                    }}
                }})
                .catch(function() {{
                    var url = new URL(window.parent.location.href);
                    if (!url.searchParams.get("_anl_done")) {{
                        url.searchParams.set("_anl_done",    "1");
                        url.searchParams.set("_anl_country", "??");
                        url.searchParams.set("_anl_city",    "??");
                        url.searchParams.set("_anl_ua",      ua);
                        url.searchParams.set("_anl_sid",     "{sid}");
                        window.parent.location.replace(url.toString());
                    }}
                }});
        }})();
        </script>
        """,
        height=1, width=0,
    )


def collect_client_params():
    """
    query_params からクライアント情報を回収して session_state に保存。
    track_pageview() の先頭で呼び出される。
    """
    import urllib.parse
    params = st.query_params

    if not params.get("_anl_done"):
        return

    country = urllib.parse.unquote(params.get("_anl_country", "??"))
    city    = urllib.parse.unquote(params.get("_anl_city",    "??"))
    ua      = urllib.parse.unquote(params.get("_anl_ua",      ""))

    st.session_state["_anl_country"]      = country
    st.session_state["_anl_city"]         = city
    st.session_state[_UA_KEY]             = ua
    st.session_state["_user_agent"]       = ua
    st.session_state["_anl_client_collected"] = True

    # query_params をクリア（URLを綺麗にする）
    try:
        for k in ["_anl_done", "_anl_country", "_anl_city",
                  "_anl_ua", "_anl_sid"]:
            st.query_params.pop(k, None)
    except Exception:
        pass


# ===========================
# User-Agent 解析
# ===========================
def _parse_ua(ua: str) -> Tuple[str, str]:
    if not ua:
        return "Unknown", "Unknown"
    ua_l = ua.lower()
    if any(k in ua_l for k in ["iphone", "android", "mobile"]):
        device = "Mobile"
    elif any(k in ua_l for k in ["ipad", "tablet"]):
        device = "Tablet"
    else:
        device = "PC"
    for b, pat in [
        ("Edge",    r"Edg/"),
        ("Chrome",  r"Chrome/"),
        ("Firefox", r"Firefox/"),
        ("Safari",  r"Safari/"),
        ("Opera",   r"OPR/"),
    ]:
        if re.search(pat, ua):
            return device, b
    return device, "Other"


# ===========================
# Google Sheets
# ===========================
@st.cache_resource
def _sheets_client():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        sa_json = _secret("GOOGLE_SERVICE_ACCOUNT_JSON")
        if not sa_json:
            return None
        sa_info = json.loads(sa_json)
        scopes  = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        logger.debug(f"sheets client: {e}")
        return None


def _sheets_ws(tab: str, headers: List[str]):
    try:
        client = _sheets_client()
        if not client:
            return None
        sp = client.open_by_key(_secret("GOOGLE_SHEETS_ID"))
        try:
            ws = sp.worksheet(tab)
        except Exception:
            ws = sp.add_worksheet(title=tab, rows=10000, cols=len(headers))
            ws.append_row(headers)
            return ws
        # 1行目が空またはヘッダーと異なる場合のみ書き込む
        row1 = ws.row_values(1)
        if not row1:
            ws.append_row(headers)
        elif row1 != headers:
            # ヘッダーが違う場合は1行目を更新
            ws.update(range_name="A1", values=[headers])
        return ws
    except Exception as e:
        logger.debug(f"sheets_ws({tab}): {e}")
        return None


# ===========================
# PV 記録
# ===========================
def track_pageview(page: str = "dashboard"):
    """ページビューを記録。main()の先頭で呼び出す。"""
    collect_client_params()

    if st.session_state.get(_TRACKED_KEY):
        return
    st.session_state[_TRACKED_KEY] = True

    now     = datetime.datetime.now(JST)
    sid     = _session_id()
    country = st.session_state.get("_anl_country", "??")
    city    = st.session_state.get("_anl_city",    "??")
    ua      = st.session_state.get(_UA_KEY, "")
    device, browser = _parse_ua(ua)

    row = {
        "ts":         now.isoformat(),
        "date":       now.strftime("%Y-%m-%d"),
        "hour":       now.hour,
        "session_id": sid,
        "country":    country,
        "city":       city,
        "device":     device,
        "browser":    browser,
    }

    if _PV_LOG_KEY not in st.session_state:
        st.session_state[_PV_LOG_KEY] = []
    st.session_state[_PV_LOG_KEY].append(row)

    logger.info(f"[analytics] track_pageview: country={country} city={city} device={device} browser={browser}")
    try:
        if _detect_backend() == "sheets":
            _write_sheets(row, sid)
        else:
            logger.warning(f"[analytics] backend=session（Sheets未設定）")
    except Exception as e:
        logger.warning(f"track_pageview error: {e}")


def _write_sheets(row: dict, sid: str):
    headers = ["ts", "date", "hour", "session_id",
               "country", "city", "device", "browser"]
    try:
        ws = _sheets_ws("pageviews", headers)
        if ws:
            ws.append_row([row.get(k, "") for k in headers],
                          value_input_option="USER_ENTERED")
            logger.info(f"[analytics] Sheets書き込み成功: {row.get('country')}/{row.get('city')} {row.get('device')}")
        else:
            logger.warning("[analytics] Sheets ws=None（認証失敗の可能性）")
    except Exception as e:
        logger.warning(f"[analytics] Sheets書き込みエラー: {e}")
    rt_ws = _sheets_ws("realtime", ["session_id", "last_seen"])
    if rt_ws:
        now_str = row["ts"]
        for i, rec in enumerate(rt_ws.get_all_records(), start=2):
            if rec.get("session_id") == sid:
                rt_ws.update_cell(i, 2, now_str)
                return
        rt_ws.append_row([sid, now_str])


# ===========================
# データ読み込み
# ===========================
@st.cache_data(ttl=120, show_spinner=False)
def _load_df() -> pd.DataFrame:
    if _detect_backend() == "sheets":
        try:
            c = _sheets_client()
            if c:
                sp = c.open_by_key(_secret("GOOGLE_SHEETS_ID"))
                ws = sp.worksheet("pageviews")
                recs = ws.get_all_records()
                if recs:
                    df = pd.DataFrame(recs)
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    return df
        except Exception as e:
            logger.warning(f"sheets load: {e}")

    log = st.session_state.get(_PV_LOG_KEY, [])
    if log:
        df = pd.DataFrame(log)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    return pd.DataFrame()


@st.cache_data(ttl=30, show_spinner=False)
def _load_realtime() -> int:
    now    = datetime.datetime.now(JST)
    cutoff = now - timedelta(minutes=REALTIME_MIN)

    if _detect_backend() == "sheets":
        try:
            c = _sheets_client()
            if c:
                sp  = c.open_by_key(_secret("GOOGLE_SHEETS_ID"))
                ws  = sp.worksheet("realtime")
                cnt = sum(
                    1 for r in ws.get_all_records()
                    if _is_recent(r.get("last_seen", ""), cutoff)
                )
                return max(cnt, 1)
        except Exception:
            pass
    return 1


def _is_recent(ts_str: str, cutoff) -> bool:
    try:
        ls = datetime.datetime.fromisoformat(ts_str)
        if ls.tzinfo is None:
            ls = JST.localize(ls)
        return ls >= cutoff
    except Exception:
        return False


# ===========================
# ダッシュボード描画
# ===========================
def render_analytics_dashboard():
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    st.header("📊 アクセス解析")

    backend = _detect_backend()
    st.caption(f"保存先: {'🟢 Google Sheets（永続保存中）' if backend == 'sheets' else '🔴 session_stateのみ'}")

    # 現在のクライアント情報
    country = st.session_state.get("_anl_country", "取得中...")
    city    = st.session_state.get("_anl_city",    "取得中...")
    ua      = st.session_state.get(_UA_KEY, "")
    device, browser = _parse_ua(ua)

    st.markdown(
        f'<div style="background:#f0f9ff;border:1px solid #bae6fd;'
        f'border-radius:8px;padding:10px 16px;margin-bottom:12px;font-size:13px;">'
        f'🌍 あなたの情報: <b>{country}</b> / {city} &nbsp;|&nbsp; '
        f'💻 {device} / {browser}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # リアルタイム
    rt = _load_realtime()
    st.markdown(
        f'<div style="display:inline-flex;align-items:center;gap:10px;'
        f'background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;'
        f'padding:10px 20px;margin-bottom:16px;">'
        f'<span style="font-size:20px;">🟢</span>'
        f'<span style="font-size:18px;font-weight:700;color:#15803d;">'
        f'現在 {rt} 人がオンライン</span>'
        f'<span style="font-size:11px;color:#888;">（直近{REALTIME_MIN}分）</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("🔄 データ更新", key="anl_refresh"):
        st.cache_data.clear()
        st.rerun()

    df = _load_df()
    if df is None or df.empty:
        st.info("まだデータがありません。")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    today  = pd.Timestamp.now().normalize()
    last7  = today - pd.Timedelta(days=7)
    last30 = today - pd.Timedelta(days=30)
    has_sid = "session_id" in df.columns

    k = st.columns(6)
    k[0].metric("本日 PV",        f"{len(df[df['date'] >= today]):,}")
    k[1].metric("本日 Session",   f"{df[df['date'] >= today]['session_id'].nunique() if has_sid else '-'}")
    k[2].metric("7日 PV",         f"{len(df[df['date'] >= last7]):,}")
    k[3].metric("7日 Session",    f"{df[df['date'] >= last7]['session_id'].nunique() if has_sid else '-'}")
    k[4].metric("累計 PV",        f"{len(df):,}")
    k[5].metric("累計 Session",   f"{df['session_id'].nunique() if has_sid else '-'}")

    st.divider()

    t_trend, t_geo, t_dev, t_hour, t_raw = st.tabs(
        ["📅 トレンド", "🌍 地域", "💻 デバイス", "⏰ 時間帯", "📋 生データ"]
    )

    with t_trend:
        period = st.radio("期間", ["7日", "30日", "90日", "全期間"],
                          index=1, horizontal=True, key="anl_period")
        cutoff = {"7日": last7, "30日": last30,
                  "90日": today - pd.Timedelta(days=90),
                  "全期間": pd.Timestamp("2000-01-01")}[period]
        df_t = df[df["date"] >= cutoff].copy()
        if not df_t.empty:
            agg = {"PV": ("date", "count")}
            if has_sid:
                agg["Sessions"] = ("session_id", "nunique")
            daily = df_t.groupby("date").agg(**agg).reset_index()
            daily["date"] = pd.to_datetime(daily["date"])
            fig, ax1 = plt.subplots(figsize=(12, 4))
            ax1.bar(daily["date"], daily["PV"], color="#1976d2",
                    alpha=0.75, label="PV", width=0.6)
            ax1.set_ylabel("PV", color="#1976d2")
            if "Sessions" in daily.columns:
                ax2 = ax1.twinx()
                ax2.plot(daily["date"], daily["Sessions"],
                         color="#e91e63", linewidth=2,
                         marker="o", markersize=4, label="Sessions")
                ax2.set_ylabel("Sessions", color="#e91e63")
                l1, lb1 = ax1.get_legend_handles_labels()
                l2, lb2 = ax2.get_legend_handles_labels()
                ax1.legend(l1+l2, lb1+lb2, loc="upper left", fontsize=9)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax1.grid(True, axis="y", alpha=0.25)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

    with t_geo:
        if "country" not in df.columns or df["country"].isin(["??", "Local", ""]).all():
            st.warning("地域データ取得中... 次回アクセス時から記録されます。")
        else:
            df_v = df[~df["country"].isin(["??", "Local", ""])]
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**🌏 国別 Top10**")
                cnt_c = df_v["country"].value_counts().head(10).reset_index()
                cnt_c.columns = ["国", "PV"]
                cnt_c["割合"] = (cnt_c["PV"]/cnt_c["PV"].sum()*100).round(1).astype(str)+"%"
                st.dataframe(cnt_c, use_container_width=True, hide_index=True)
                fig_c, ax_c = plt.subplots(figsize=(6, 4))
                cs = ["#1565c0" if x == "JP" else "#90caf9" for x in cnt_c["国"]]
                ax_c.barh(cnt_c["国"][::-1], cnt_c["PV"][::-1], color=cs[::-1])
                ax_c.grid(True, axis="x", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_c, clear_figure=True)
            with g2:
                st.markdown("**🏙️ 都市別 Top10**")
                df_ci = df_v[~df_v.get("city", pd.Series(dtype=str)).isin(["??","Local",""])]
                cnt_ci = df_ci["city"].value_counts().head(10).reset_index()
                cnt_ci.columns = ["都市", "PV"]
                st.dataframe(cnt_ci, use_container_width=True, hide_index=True)
                fig_ci, ax_ci = plt.subplots(figsize=(6, 4))
                ax_ci.barh(cnt_ci["都市"][::-1], cnt_ci["PV"][::-1], color="#4caf50")
                ax_ci.grid(True, axis="x", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_ci, clear_figure=True)

    with t_dev:
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**💻 デバイス別**")
            if "device" in df.columns:
                df_dv = df[~df["device"].isin(["Unknown",""])]
                if not df_dv.empty:
                    cnt_d = df_dv["device"].value_counts().reset_index()
                    cnt_d.columns = ["デバイス", "PV"]
                    st.dataframe(cnt_d, use_container_width=True, hide_index=True)
                    fig_d, ax_d = plt.subplots(figsize=(4, 4))
                    cm = {"PC":"#1976d2","Mobile":"#e91e63","Tablet":"#ff9800"}
                    ax_d.pie(cnt_d["PV"], labels=cnt_d["デバイス"],
                             colors=[cm.get(x,"#888") for x in cnt_d["デバイス"]],
                             autopct="%1.1f%%", startangle=90)
                    plt.tight_layout()
                    st.pyplot(fig_d, clear_figure=True)
                else:
                    st.info("次回アクセス時から記録されます")
        with d2:
            st.markdown("**🌐 ブラウザ別**")
            if "browser" in df.columns:
                df_br = df[~df["browser"].isin(["Unknown",""])]
                if not df_br.empty:
                    cnt_b = df_br["browser"].value_counts().head(8).reset_index()
                    cnt_b.columns = ["ブラウザ", "PV"]
                    st.dataframe(cnt_b, use_container_width=True, hide_index=True)
                    fig_b, ax_b = plt.subplots(figsize=(6, 4))
                    ax_b.barh(cnt_b["ブラウザ"][::-1], cnt_b["PV"][::-1], color="#7b1fa2")
                    ax_b.grid(True, axis="x", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_b, clear_figure=True)
                else:
                    st.info("次回アクセス時から記録されます")

    with t_hour:
        if "hour" in df.columns:
            hcnt = df["hour"].value_counts().reindex(range(24), fill_value=0).sort_index()
            fig_h, ax_h = plt.subplots(figsize=(12, 4))
            ax_h.bar(range(24), hcnt.values,
                     color=["#ff5722" if (7<=i<=9 or 19<=i<=23) else "#1976d2" for i in range(24)],
                     width=0.7)
            ax_h.set_xticks(range(24))
            ax_h.set_xticklabels([str(i) for i in range(24)], fontsize=9)
            ax_h.set_xlabel("Hour (JST)")
            ax_h.set_ylabel("PV")
            ax_h.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_h, clear_figure=True)

    with t_raw:
        show_cols = [c for c in ["ts","date","hour","session_id","country","city","device","browser"] if c in df.columns]
        sort_col = "ts" if "ts" in df.columns else "date"
        df_sorted = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        total = len(df_sorted)
        n_show = st.slider("表示件数", min_value=10, max_value=min(total, 1000),
                           value=min(100, total), step=10, key="anl_raw_n")
        st.caption(f"全 {total} 件中、新しい順に {n_show} 件表示")
        st.dataframe(df_sorted.head(n_show)[show_cols],
                     use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 全データ CSV ダウンロード", data=csv,
                           file_name=f"analytics_{datetime.datetime.now(JST).strftime('%Y%m%d')}.csv",
                           mime="text/csv")
