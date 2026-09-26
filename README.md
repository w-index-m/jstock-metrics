# jstock-metrics

jstock-metrics

日本語 / Japanese

## 概要

jstock-metrics は、日本株の市場データ・適時開示・財務指標を収集し、Streamlit 上で可視化・分析できるダッシュボードです。

本プロジェクトでは、金融情報APIと公開データを組み合わせて、以下を実現します。

- 日本株の価格・財務・企業情報の確認
- TDnet / EDINET などからの適時開示の収集
- AI による開示スコアリング
- 重要度順のランキング表示
- 絞り込み・期間選択・CSV出力
- Claude を利用した LLM as a Judge

## 主な機能

- Streamlit ベースの Web UI
- 日本株銘柄の一覧・業種別フィルタ
- 金融情報 API 連携
- 適時開示の収集と表示
- AI による重要度評価
- 重要度 / インパクト / センチメント / 緊急度 / テーマ の評価
- JSON 解析と正規表現によるフォールバック
- 評価結果の CSV ダウンロード
- 重要度分布チャートと開示カード表示

## LLM as a Judge

本アプリでは、取得した適時開示リストを Claude にまとめて渡し、次の観点でスコアリングします。

- importance: 1-5
- impact: 高 / 中 / 低
- sentiment: ポジティブ / 中立 / ネガティブ
- urgency: 高 / 中 / 低
- themes: 主要テーマ（AI、業績修正、M&A、株主還元など）

JSON 形式で評価結果を受け取り、失敗時には正規表現ベースのフォールバックを使って最低限のスコアを復元します。

## 画面機能

- 業種絞り込み
- 取得期間: 1〜5 営業日
- フィルタ: 重要度 / インパクト / ポジネガ
- 重要度分布バーチャート
- 開示カード表示（重要度順・カラーコーディング）
- AI根拠の表示
- テーマタグの表示
- CSV ダウンロード

## 技術スタック

- Python
- Streamlit
- Claude API
- Pandas
- NumPy
- Matplotlib
- Plotly
- Requests
- yfinance
- BeautifulSoup
- pdfplumber

## データソース

- 金融情報 API
- TDnet
- EDINET
- Finnhub
- Alpha Vantage
- J-Quants（必要に応じて）

## 実行方法

```bash
pip install -r requirements.txt
streamlit run app.py
```

必要に応じて、環境変数または Streamlit Secrets に API キーを設定してください。

```bash
export CLAUDE_API_KEY="your_api_key"
```

## 注意事項

- AI による評価は補助的な判断であり、投資助言ではありません。
- 重要な投資判断は、公式開示書類と自社の分析を必ず確認してください。

---

Overview / English

## Overview

jstock-metrics is a Japanese stock dashboard for collecting market data, disclosure information, and financial indicators, then visualizing and analyzing them in a Streamlit app.

This project combines financial APIs and public data sources to support:

- checking Japanese stock prices, financials, and company information
- collecting disclosure information from TDnet / EDINET
- AI-based disclosure scoring
- ranking important disclosures by priority
- filtering by sector and date range
- exporting evaluation results as CSV
- using Claude for LLM as a Judge

## Main Features

- Streamlit-based web UI
- Japanese stock lists and sector filtering
- financial API integration
- disclosure collection and display
- AI scoring for disclosure importance
- evaluation across importance, impact, sentiment, urgency, and themes
- JSON parsing with regex fallback when parsing fails
- CSV export of scored results
- importance distribution charts and disclosure card views

## LLM as a Judge

The app sends a batch of disclosure items to Claude and scores each item based on:

- importance: 1-5
- impact: High / Medium / Low
- sentiment: Positive / Neutral / Negative
- urgency: High / Medium / Low
- themes: key topics such as AI, earnings revision, M&A, shareholder returns, etc.

The scoring result is processed as structured JSON, and if parsing fails, a regex-based fallback restores a usable result.

## UI Features

- sector filtering
- date range selection: 1-5 business days
- filters: importance, impact, sentiment
- importance distribution bar chart
- disclosure cards sorted by importance
- color-coded prioritization
- AI rationale display
- theme tags
- CSV download

## Tech Stack

- Python
- Streamlit
- Claude API
- Pandas
- NumPy
- Matplotlib
- Plotly
- Requests
- yfinance
- BeautifulSoup
- pdfplumber

## Data Sources

- financial information API
- TDnet
- EDINET
- Finnhub
- Alpha Vantage
- J-Quants (if configured)

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set API keys in environment variables or Streamlit Secrets as needed.

```bash
export CLAUDE_API_KEY="your_api_key"
```

## Disclaimer

- AI scoring is a decision-support tool only and is not investment advice.
- Always verify important investment decisions against official disclosure documents and your own analysis.
