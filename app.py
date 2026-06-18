from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


APP_TITLE = "股票德州扑克量化决策工具"
DATA_DIR = Path("data")
SECTOR_MAP_PATH = DATA_DIR / "sector_map.csv"
MANUAL_SCORES_PATH = DATA_DIR / "manual_scores.csv"
SUPPORTED_CONFIG_SUFFIXES = {".csv", ".xlsx", ".xls"}
REFERENCE_SYMBOLS = ["QQQ", "SMH", "XLI", "XLU", "HYG", "LQD", "^VIX"]

DEFAULT_SECTOR_ROWS = [
    ("MU", "存储", "SMH"),
    ("SNDK", "存储", "SMH"),
    ("NVDA", "人工智能架构", "SMH"),
    ("AVGO", "人工智能架构", "SMH"),
    ("TSM", "人工智能架构", "SMH"),
    ("COHR", "光通信与共封装光学", "SMH"),
    ("LITE", "光通信与共封装光学", "SMH"),
    ("ETN", "电力与散热", "XLI"),
    ("VRT", "电力与散热", "XLI"),
    ("TSLA", "其他", "QQQ"),
]

DEFAULT_MANUAL_ROWS = [
    ("MU", 22, 19, 11, "人工智能内存瓶颈"),
    ("NVDA", 24, 20, 10, "人工智能工厂核心控制点"),
    ("AVGO", 23, 18, 11, "定制芯片与人工智能网络"),
    ("TSM", 23, 19, 11, "先进制程与先进封装"),
    ("ETN", 20, 17, 10, "电力基础设施"),
    ("COHR", 19, 17, 7, "光通信与磷化铟"),
    ("LITE", 19, 17, 7, "激光与光源"),
    ("VRT", 21, 16, 7, "电力与散热，估值偏高"),
    ("TSLA", 12, 10, 6, "叙事弹性"),
]


@dataclass
class ManualScore:
    fundamental_score: float = 12.5
    bottleneck_score: float = 10.0
    valuation_score: float = 7.5
    notes: str = "默认中性评分"
    is_default: bool = True


@dataclass
class SectorInfo:
    sector: str = "其他"
    sector_etf: str = "QQQ"


COLUMN_ALIASES = {
    "ticker": "股票代码",
    "symbol": "股票代码",
    "股票": "股票代码",
    "股票代码": "股票代码",
    "代码": "股票代码",
    "sector": "所属板块",
    "板块": "所属板块",
    "所属板块": "所属板块",
    "industry": "所属板块",
    "sector_etf": "板块基金",
    "sector etf": "板块基金",
    "etf": "板块基金",
    "板块基金": "板块基金",
    "板块etf": "板块基金",
    "fundamental_score": "基本面评分",
    "fundamental": "基本面评分",
    "基本面评分": "基本面评分",
    "基本面": "基本面评分",
    "bottleneck_score": "真实瓶颈评分",
    "bottleneck": "真实瓶颈评分",
    "真实瓶颈评分": "真实瓶颈评分",
    "瓶颈评分": "真实瓶颈评分",
    "valuation_score": "估值评分",
    "valuation": "估值评分",
    "估值评分": "估值评分",
    "估值": "估值评分",
    "notes": "备注",
    "note": "备注",
    "备注": "备注",
    "说明": "备注",
}


def ensure_default_files() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not SECTOR_MAP_PATH.exists():
        pd.DataFrame(DEFAULT_SECTOR_ROWS, columns=["ticker", "sector", "sector_etf"]).to_csv(
            SECTOR_MAP_PATH, index=False, encoding="utf-8-sig"
        )
    if not MANUAL_SCORES_PATH.exists():
        pd.DataFrame(
            DEFAULT_MANUAL_ROWS,
            columns=["ticker", "fundamental_score", "bottleneck_score", "valuation_score", "notes"],
        ).to_csv(MANUAL_SCORES_PATH, index=False, encoding="utf-8-sig")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    for col in out.columns:
        raw = str(col).strip()
        key = raw.lower().replace("-", "_")
        rename_map[col] = COLUMN_ALIASES.get(key, COLUMN_ALIASES.get(raw, raw))
    out = out.rename(columns=rename_map)
    if "股票代码" in out.columns:
        out["股票代码"] = out["股票代码"].astype(str).str.strip().str.upper()
    return out


def read_table(source: Any) -> pd.DataFrame:
    name = getattr(source, "name", str(source))
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(source, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(source, encoding="gbk")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    raise ValueError("不支持的文件格式")


def safe_read_table(source: Any) -> pd.DataFrame | None:
    try:
        return normalize_columns(read_table(source))
    except Exception:
        return None


def detect_table_type(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty or "股票代码" not in df.columns:
        return None
    cols = set(df.columns)
    if {"所属板块", "板块基金"}.issubset(cols):
        return "板块映射表"
    if {"基本面评分", "真实瓶颈评分", "估值评分"}.issubset(cols):
        return "人工评分表"
    return None


def config_files() -> list[Path]:
    files: list[Path] = []
    for folder in [Path("."), DATA_DIR]:
        if folder.exists():
            files.extend([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_CONFIG_SUFFIXES])
    return list(dict.fromkeys(files))


def auto_detect_config() -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    sector_df = None
    manual_df = None
    messages = []
    for file in config_files():
        df = safe_read_table(file)
        table_type = detect_table_type(df)
        if table_type == "板块映射表" and sector_df is None:
            sector_df = df
            messages.append(f"已自动识别板块映射表：{file.name}")
        elif table_type == "人工评分表" and manual_df is None:
            manual_df = df
            messages.append(f"已自动识别人工评分表：{file.name}")
    return sector_df, manual_df, messages


def default_sector_df() -> pd.DataFrame:
    return normalize_columns(pd.DataFrame(DEFAULT_SECTOR_ROWS, columns=["ticker", "sector", "sector_etf"]))


def default_manual_df() -> pd.DataFrame:
    return normalize_columns(
        pd.DataFrame(
            DEFAULT_MANUAL_ROWS,
            columns=["ticker", "fundamental_score", "bottleneck_score", "valuation_score", "notes"],
        )
    )


def load_config(uploaded_sector: Any, uploaded_manual: Any) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    ensure_default_files()
    sector_df, manual_df, messages = auto_detect_config()
    warnings = []

    if uploaded_sector is not None:
        uploaded = safe_read_table(uploaded_sector)
        if uploaded is None or detect_table_type(uploaded) != "板块映射表":
            warnings.append("文件读取失败，请检查文件格式或字段名。")
        else:
            sector_df = uploaded
            messages.append("已读取上传的板块映射表。")

    if uploaded_manual is not None:
        uploaded = safe_read_table(uploaded_manual)
        if uploaded is None or detect_table_type(uploaded) != "人工评分表":
            warnings.append("文件读取失败，请检查文件格式或字段名。")
        else:
            manual_df = uploaded
            messages.append("已读取上传的人工评分表。")

    if sector_df is None:
        local = safe_read_table(SECTOR_MAP_PATH)
        sector_df = local if local is not None else default_sector_df()
        messages.append("未识别到可用板块映射表，已使用默认板块配置。")

    if manual_df is None:
        local = safe_read_table(MANUAL_SCORES_PATH)
        manual_df = local if local is not None else default_manual_df()
        messages.append("未识别到可用人工评分表，已使用默认评分配置。")

    return sector_df, manual_df, messages, warnings


def get_sector_info(ticker: str, sector_df: pd.DataFrame) -> SectorInfo:
    if "股票代码" not in sector_df.columns:
        return SectorInfo()
    row = sector_df.loc[sector_df["股票代码"] == ticker]
    if row.empty:
        return SectorInfo()
    item = row.iloc[0]
    return SectorInfo(
        sector=str(item.get("所属板块", "其他")),
        sector_etf=str(item.get("板块基金", "QQQ")).upper(),
    )


def get_manual_score(ticker: str, manual_df: pd.DataFrame) -> ManualScore:
    if "股票代码" not in manual_df.columns:
        return ManualScore()
    row = manual_df.loc[manual_df["股票代码"] == ticker]
    if row.empty:
        return ManualScore()
    item = row.iloc[0]
    try:
        return ManualScore(
            fundamental_score=float(item.get("基本面评分", 12.5)),
            bottleneck_score=float(item.get("真实瓶颈评分", 10)),
            valuation_score=float(item.get("估值评分", 7.5)),
            notes=str(item.get("备注", "")),
            is_default=False,
        )
    except Exception:
        return ManualScore()


def get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        return None
    return None


@st.cache_data(ttl=60, show_spinner=False)
def latest_price_from_finnhub(symbol: str, api_key: str) -> float | None:
    try:
        response = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol, "token": api_key},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        price = float(data.get("c", 0))
        return price if price > 0 else None
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def latest_price_from_yfinance(symbol: str, refresh_key: str) -> float | None:
    try:
        data = yf.download(symbol, period="5d", interval="1d", auto_adjust=False, progress=False, threads=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        close = data["Close"].dropna()
        if close.empty:
            return None
        return float(close.iloc[-1])
    except Exception:
        return None


def get_latest_price(symbol: str, refresh_key: str) -> tuple[float | None, str]:
    api_key = get_secret("FINNHUB_API_KEY")
    if api_key:
        price = latest_price_from_finnhub(symbol, api_key)
        if price is not None:
            return price, "Finnhub"
    return latest_price_from_yfinance(symbol, refresh_key), "yfinance"


@st.cache_data(ttl=3600, show_spinner=False)
def download_history(symbol: str, refresh_key: str) -> pd.DataFrame:
    try:
        data = yf.download(symbol, period="5y", interval="1d", auto_adjust=False, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns={col: str(col).title() for col in data.columns})
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in data.columns for col in required):
        return pd.DataFrame()
    data = data[required].dropna(subset=["High", "Low", "Close"])
    data.index = pd.to_datetime(data.index)
    return data


def apply_latest_price(data: pd.DataFrame, latest_price: float | None) -> pd.DataFrame:
    if data.empty or latest_price is None or not np.isfinite(latest_price):
        return data
    out = data.copy()
    last_index = out.index[-1]
    out.loc[last_index, "Close"] = latest_price
    out.loc[last_index, "High"] = max(float(out.loc[last_index, "High"]), latest_price)
    out.loc[last_index, "Low"] = min(float(out.loc[last_index, "Low"]), latest_price)
    return out


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = data["Close"].shift(1)
    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - prev_close).abs(),
            (data["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    out = data.copy()
    out["二十日均线"] = out["Close"].rolling(20).mean()
    out["五十日均线"] = out["Close"].rolling(50).mean()
    out["二百日均线"] = out["Close"].rolling(200).mean()
    out["相对强弱指标"] = calculate_rsi(out["Close"], 14)
    out["真实波幅"] = calculate_atr(out, 14)
    out["二十日涨幅"] = out["Close"].pct_change(20)
    out["六十日涨幅"] = out["Close"].pct_change(60)
    out["一百二十日涨幅"] = out["Close"].pct_change(120)
    out["五日涨幅"] = out["Close"].pct_change(5)
    out["二十日平均成交量"] = out["Volume"].rolling(20).mean()
    out["六十日平均成交量"] = out["Volume"].rolling(60).mean()
    return out


def prepare_market_data(ticker: str, sector_etf: str, force_refresh: bool) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], str]:
    refresh_key = pd.Timestamp.utcnow().isoformat() if force_refresh else "缓存"
    symbols = list(dict.fromkeys([ticker, *REFERENCE_SYMBOLS, sector_etf]))
    latest_price, latest_source = get_latest_price(ticker, refresh_key)
    market: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        history = download_history(symbol, refresh_key)
        if symbol == ticker:
            history = apply_latest_price(history, latest_price)
        market[symbol] = add_indicators(history)
    return market.get(ticker, pd.DataFrame()), market, latest_source


def latest_aligned_value(data: pd.DataFrame, column: str, date: pd.Timestamp) -> float:
    if data.empty or column not in data.columns:
        return float("nan")
    values = data.loc[data.index <= date, column].dropna()
    return float(values.iloc[-1]) if not values.empty else float("nan")


def calculate_score(stock: pd.DataFrame, qqq: pd.DataFrame, sector: pd.DataFrame, manual: ManualScore) -> dict[str, Any]:
    usable = stock.dropna(
        subset=["二百日均线", "真实波幅", "相对强弱指标", "二十日涨幅", "六十日涨幅", "六十日平均成交量"]
    )
    if usable.empty:
        raise ValueError("历史数据不足")
    latest = usable.iloc[-1]
    date = latest.name
    close = float(latest["Close"])
    qqq_return20 = latest_aligned_value(qqq, "二十日涨幅", date)
    qqq_return60 = latest_aligned_value(qqq, "六十日涨幅", date)
    sector_return60 = latest_aligned_value(sector, "六十日涨幅", date)

    technical_score = 0
    technical_score += 5 if close > float(latest["五十日均线"]) else 0
    technical_score += 5 if close > float(latest["二百日均线"]) else 0
    technical_score += 5 if float(latest["二十日涨幅"]) > qqq_return20 else 0
    technical_score += 5 if float(latest["六十日涨幅"]) > qqq_return60 else 0
    technical_score += 5 if float(latest["六十日涨幅"]) > sector_return60 else 0
    technical_score += 5 if float(latest["二十日平均成交量"]) > float(latest["六十日平均成交量"]) else 0

    risk_penalty = 0
    risk_penalty += 2 if float(latest["相对强弱指标"]) > 85 else 0
    risk_penalty += 2 if close > float(latest["二十日均线"]) + 2.5 * float(latest["真实波幅"]) else 0
    risk_penalty += 2 if float(latest["二十日涨幅"]) > 0.40 else 0
    risk_penalty += 2 if float(latest["五日涨幅"]) > 0.20 else 0

    raw_score = technical_score + manual.fundamental_score + manual.bottleneck_score + manual.valuation_score - risk_penalty
    total_score = min(max(raw_score / 90 * 100, 0), 100)
    return {
        "日期": pd.Timestamp(date),
        "最新价格": close,
        "技术趋势评分": technical_score,
        "风险扣分": risk_penalty,
        "综合评分": total_score,
        "相对强弱指标": float(latest["相对强弱指标"]),
        "真实波幅": float(latest["真实波幅"]),
        "二十日涨幅": float(latest["二十日涨幅"]),
        "六十日涨幅": float(latest["六十日涨幅"]),
        "五日涨幅": float(latest["五日涨幅"]),
        "纳指二十日涨幅": qqq_return20,
        "纳指六十日涨幅": qqq_return60,
        "板块六十日涨幅": sector_return60,
    }


def rating(total_score: float) -> str:
    if total_score >= 85:
        return "强势主线，可持有或回调加仓"
    if total_score >= 75:
        return "可买候选，适合分批建仓"
    if total_score >= 65:
        return "观察，等待突破或回调"
    if total_score >= 50:
        return "弱观察，暂不参与"
    return "排除，不参与"


def market_state(score: dict[str, Any]) -> str:
    if score["相对强弱指标"] > 85 or score["二十日涨幅"] > 0.40:
        return "趋势强但短线过热"
    if score["技术趋势评分"] >= 25:
        return "趋势强势"
    if score["技术趋势评分"] >= 15:
        return "趋势中性"
    return "趋势偏弱"


def calculate_trade_plan(stock: pd.DataFrame, portfolio_value: float, risk_per_trade: float) -> dict[str, Any]:
    latest = stock.dropna(subset=["五十日均线", "真实波幅"]).iloc[-1]
    close = float(latest["Close"])
    atr = float(latest["真实波幅"])
    sma50 = float(latest["五十日均线"])
    recent_low = float(stock["Low"].tail(20).min())
    stop_candidates = [close - 2.2 * atr, recent_low * 0.99, sma50 * 0.98]
    valid_stops = [value for value in stop_candidates if np.isfinite(value) and value < close]
    stop_loss = max(valid_stops) if valid_stops else close - 2.2 * atr
    risk_per_share = max(close - stop_loss, 0)
    shares_by_risk = math.floor(portfolio_value * risk_per_trade / risk_per_share) if risk_per_share > 0 else 0
    max_shares_by_weight = math.floor(portfolio_value * 0.15 / close) if close > 0 else 0
    shares = max(min(shares_by_risk, max_shares_by_weight), 0)
    position_value = shares * close
    max_loss = shares * risk_per_share
    return {
        "参考买入价": close,
        "止损位": stop_loss,
        "每股风险": risk_per_share,
        "第一止盈位": close + 2 * risk_per_share,
        "第二止盈位": close + 3 * risk_per_share,
        "建议买入股数": shares,
        "建议仓位金额": position_value,
        "建议仓位比例": position_value / portfolio_value if portfolio_value > 0 else 0,
        "最大亏损金额": max_loss,
        "最大亏损比例": max_loss / portfolio_value if portfolio_value > 0 else 0,
        "移动止盈规则": "第一止盈位兑现部分仓位后，将止损上移至参考买入价；若继续创阶段新高，则用二十日均线或两倍真实波幅跟踪保护利润。",
    }


def calculate_odds(stock: pd.DataFrame, holding_days: int) -> dict[str, Any]:
    data = stock.copy()
    for days in [20, 60, 120]:
        data[f"未来{days}日收益"] = data["Close"].shift(-days) / data["Close"] - 1
    latest = data.dropna(subset=["五十日均线", "二百日均线", "相对强弱指标", "六十日涨幅"]).iloc[-1]
    similar = data.loc[
        ((data["Close"] > data["五十日均线"]) == (latest["Close"] > latest["五十日均线"]))
        & ((data["Close"] > data["二百日均线"]) == (latest["Close"] > latest["二百日均线"]))
        & ((data["相对强弱指标"] - latest["相对强弱指标"]).abs() <= 12)
        & ((data["六十日涨幅"] - latest["六十日涨幅"]).abs() <= 0.18)
    ].dropna(subset=[f"未来{holding_days}日收益"])
    sample = similar if len(similar) >= 20 else data.dropna(subset=[f"未来{holding_days}日收益"]).tail(500)

    def win_rate(days: int) -> float:
        available = sample.dropna(subset=[f"未来{days}日收益"])
        return float((available[f"未来{days}日收益"] > 0).mean()) if not available.empty else float("nan")

    target_returns = sample[f"未来{holding_days}日收益"].dropna()
    gains = target_returns[target_returns > 0]
    losses = target_returns[target_returns < 0]
    avg_gain = float(gains.mean()) if not gains.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    selected_win_rate = win_rate(holding_days)
    payoff = avg_gain / abs(avg_loss) if avg_loss < 0 else float("nan")
    expectancy = selected_win_rate * avg_gain + (1 - selected_win_rate) * avg_loss if np.isfinite(selected_win_rate) else float("nan")
    return {
        "历史相似样本数": int(len(similar)),
        "二十日胜率": win_rate(20),
        "六十日胜率": win_rate(60),
        "一百二十日胜率": win_rate(120),
        "平均盈利": avg_gain,
        "平均亏损": avg_loss,
        "盈亏赔率": payoff,
        "期望值": expectancy,
        "样本不足": len(similar) < 30,
    }


def format_pct(value: float) -> str:
    return "无数据" if not np.isfinite(value) else f"{value * 100:.2f}%"


def format_money(value: float) -> str:
    return "无数据" if not np.isfinite(value) else f"${value:,.2f}"


def format_number(value: float) -> str:
    return "无数据" if not np.isfinite(value) else f"{value:.2f}"


def build_explanation(
    ticker: str,
    sector: SectorInfo,
    score: dict[str, Any],
    manual: ManualScore,
    trade_plan: dict[str, Any],
    odds: dict[str, Any],
) -> str:
    trend_text = "较强" if score["技术趋势评分"] >= 20 else "一般" if score["技术趋势评分"] >= 10 else "偏弱"
    heat_text = "存在短线过热，需要等待回踩或降低仓位" if score["风险扣分"] > 0 else "暂未触发主要过热扣分"
    odds_text = "具备正期望" if odds["期望值"] > 0 else "期望值不占优，需要更谨慎"
    return (
        f"{ticker} 属于{sector.sector}，综合评分为 {score['综合评分']:.1f} 分。"
        f"技术趋势评分为 {score['技术趋势评分']}/30，基本面、真实瓶颈、估值评分分别为 "
        f"{manual.fundamental_score:g}、{manual.bottleneck_score:g}、{manual.valuation_score:g}。"
        f"当前趋势{trend_text}，{heat_text}。"
        f"历史相似样本数为 {odds['历史相似样本数']}，二十日胜率 {format_pct(odds['二十日胜率'])}，"
        f"六十日胜率 {format_pct(odds['六十日胜率'])}，盈亏赔率 {format_number(odds['盈亏赔率'])}，{odds_text}。"
        f"止损位设在 {format_money(trade_plan['止损位'])}，来自真实波幅止损、二十日结构低点和五十日均线保护位中低于现价且最靠近现价的位置。"
        f"如果股价跌破止损位、放量跌破五十日均线、相对 QQQ 与板块基金持续转弱，或产业逻辑发生变化，则本次逻辑失效。"
    )


def price_chart(stock: pd.DataFrame, trade_plan: dict[str, Any], ticker: str) -> go.Figure:
    data = stock.tail(260)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="收盘价", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data["二十日均线"], name="二十日均线"))
    fig.add_trace(go.Scatter(x=data.index, y=data["五十日均线"], name="五十日均线"))
    fig.add_trace(go.Scatter(x=data.index, y=data["二百日均线"], name="二百日均线"))
    for name, value, color in [
        ("止损位", trade_plan["止损位"], "#d62728"),
        ("第一止盈位", trade_plan["第一止盈位"], "#2ca02c"),
        ("第二止盈位", trade_plan["第二止盈位"], "#1f77b4"),
    ]:
        fig.add_hline(y=float(value), line_dash="dash", line_color=color, annotation_text=name)
    fig.update_layout(
        title=f"{ticker} 价格走势图",
        height=480,
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="日期",
        yaxis_title="价格",
    )
    return fig


def rsi_chart(stock: pd.DataFrame) -> go.Figure:
    data = stock.tail(260)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["相对强弱指标"], name="相对强弱指标"))
    fig.add_hline(y=70, line_dash="dot", line_color="orange", annotation_text="七十警戒线")
    fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="八十五过热线")
    fig.update_layout(
        title="相对强弱指标图",
        height=300,
        margin=dict(l=20, r=20, t=50, b=30),
        xaxis_title="日期",
        yaxis_title="相对强弱指标",
        yaxis=dict(range=[0, 100]),
    )
    return fig


def relative_strength_chart(stock: pd.DataFrame, qqq: pd.DataFrame, sector_df: pd.DataFrame, ticker: str, sector_etf: str) -> go.Figure:
    merged = pd.concat(
        [
            stock["Close"].rename("股票"),
            qqq["Close"].rename("纳指基金"),
            sector_df["Close"].rename("板块基金"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    fig = go.Figure()
    if not merged.empty:
        rel_qqq = (merged["股票"] / merged["股票"].iloc[0]) / (merged["纳指基金"] / merged["纳指基金"].iloc[0])
        rel_sector = (merged["股票"] / merged["股票"].iloc[0]) / (merged["板块基金"] / merged["板块基金"].iloc[0])
        fig.add_trace(go.Scatter(x=rel_qqq.tail(260).index, y=rel_qqq.tail(260), name=f"{ticker} / QQQ"))
        fig.add_trace(go.Scatter(x=rel_sector.tail(260).index, y=rel_sector.tail(260), name=f"{ticker} / {sector_etf}"))
    fig.add_hline(y=1, line_dash="dot", line_color="gray", annotation_text="基准线")
    fig.update_layout(
        title="相对强弱图",
        height=320,
        margin=dict(l=20, r=20, t=50, b=30),
        xaxis_title="日期",
        yaxis_title="相对强弱",
    )
    return fig


def volume_chart(stock: pd.DataFrame) -> go.Figure:
    data = stock.tail(260)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="成交量", marker_color="#9ecae1"))
    fig.add_trace(go.Scatter(x=data.index, y=data["二十日平均成交量"], name="二十日平均成交量"))
    fig.add_trace(go.Scatter(x=data.index, y=data["六十日平均成交量"], name="六十日平均成交量"))
    fig.update_layout(
        title="成交量图",
        height=300,
        margin=dict(l=20, r=20, t=50, b=30),
        xaxis_title="日期",
        yaxis_title="成交量",
    )
    return fig


def analyze_one(
    ticker: str,
    portfolio_value: float,
    risk_per_trade: float,
    holding_days: int,
    force_refresh: bool,
    sector_df: pd.DataFrame,
    manual_df: pd.DataFrame,
) -> dict[str, Any] | None:
    sector = get_sector_info(ticker, sector_df)
    manual = get_manual_score(ticker, manual_df)
    if manual.is_default:
        st.warning("该股票缺少人工评分，已使用默认中性评分。")

    with st.spinner(f"正在获取 {ticker}、QQQ、SMH、XLI、XLU、HYG、LQD、波动率指数等行情数据..."):
        stock, market, latest_source = prepare_market_data(ticker, sector.sector_etf, force_refresh)

    if stock.empty:
        st.error("未获取到该股票数据，请确认股票代码是否正确。")
        return None
    qqq = market.get("QQQ", pd.DataFrame())
    sector_market = market.get(sector.sector_etf, pd.DataFrame())
    if qqq.empty:
        st.error("行情数据获取失败，请检查股票代码或稍后重试。")
        return None
    if sector_market.empty:
        st.warning("板块基金行情数据获取失败，已使用 QQQ 作为替代。")
        sector_market = qqq.copy()

    try:
        score = calculate_score(stock, qqq, sector_market, manual)
        trade_plan = calculate_trade_plan(stock, portfolio_value, risk_per_trade)
        odds = calculate_odds(stock, holding_days)
    except Exception:
        st.error("行情数据获取失败，请检查股票代码或稍后重试。")
        return None

    if odds["样本不足"]:
        st.warning("历史相似样本不足，胜率可信度较低。")

    action_rating = rating(score["综合评分"])
    state = market_state(score)
    explanation = build_explanation(ticker, sector, score, manual, trade_plan, odds)

    st.subheader("核心结论")
    cols = st.columns(6)
    cols[0].metric("股票代码", ticker)
    cols[1].metric("最新价格", format_money(score["最新价格"]))
    cols[2].metric("所属板块", sector.sector)
    cols[3].metric("市场状态", state)
    cols[4].metric("综合评分", f"{score['综合评分']:.1f}")
    cols[5].metric("操作评级", action_rating)
    st.caption(f"最新价格来源：{latest_source}")

    st.subheader("分项评分")
    cols = st.columns(5)
    cols[0].metric("技术趋势评分", f"{score['技术趋势评分']}/30")
    cols[1].metric("基本面评分", f"{manual.fundamental_score:g}")
    cols[2].metric("真实瓶颈评分", f"{manual.bottleneck_score:g}")
    cols[3].metric("估值评分", f"{manual.valuation_score:g}")
    cols[4].metric("风险扣分", f"-{score['风险扣分']}")

    st.subheader("胜率与赔率")
    cols = st.columns(8)
    cols[0].metric("历史相似样本数", f"{odds['历史相似样本数']}")
    cols[1].metric("二十日胜率", format_pct(odds["二十日胜率"]))
    cols[2].metric("六十日胜率", format_pct(odds["六十日胜率"]))
    cols[3].metric("一百二十日胜率", format_pct(odds["一百二十日胜率"]))
    cols[4].metric("平均盈利", format_pct(odds["平均盈利"]))
    cols[5].metric("平均亏损", format_pct(odds["平均亏损"]))
    cols[6].metric("盈亏赔率", format_number(odds["盈亏赔率"]))
    cols[7].metric("期望值", format_pct(odds["期望值"]))

    st.subheader("止盈止损")
    cols = st.columns(5)
    cols[0].metric("参考买入价", format_money(trade_plan["参考买入价"]))
    cols[1].metric("止损位", format_money(trade_plan["止损位"]))
    cols[2].metric("第一止盈位", format_money(trade_plan["第一止盈位"]))
    cols[3].metric("第二止盈位", format_money(trade_plan["第二止盈位"]))
    cols[4].metric("每股风险", format_money(trade_plan["每股风险"]))
    st.info(trade_plan["移动止盈规则"])

    st.subheader("仓位建议")
    cols = st.columns(5)
    cols[0].metric("建议买入股数", f"{trade_plan['建议买入股数']}")
    cols[1].metric("建议仓位金额", format_money(trade_plan["建议仓位金额"]))
    cols[2].metric("建议仓位比例", format_pct(trade_plan["建议仓位比例"]))
    cols[3].metric("最大亏损金额", format_money(trade_plan["最大亏损金额"]))
    cols[4].metric("最大亏损比例", format_pct(trade_plan["最大亏损比例"]))

    st.subheader("中文解释")
    st.write(explanation)

    st.subheader("图表")
    st.plotly_chart(price_chart(stock, trade_plan, ticker), use_container_width=True)
    chart_col_1, chart_col_2 = st.columns(2)
    chart_col_1.plotly_chart(rsi_chart(stock), use_container_width=True)
    chart_col_2.plotly_chart(relative_strength_chart(stock, qqq, sector_market, ticker, sector.sector_etf), use_container_width=True)
    st.plotly_chart(volume_chart(stock), use_container_width=True)

    return {
        "股票代码": ticker,
        "所属板块": sector.sector,
        "最新价格": score["最新价格"],
        "综合评分": score["综合评分"],
        "操作评级": action_rating,
        "技术趋势评分": score["技术趋势评分"],
        "基本面评分": manual.fundamental_score,
        "真实瓶颈评分": manual.bottleneck_score,
        "估值评分": manual.valuation_score,
        "风险扣分": score["风险扣分"],
        "二十日胜率": odds["二十日胜率"],
        "六十日胜率": odds["六十日胜率"],
        "期望值": odds["期望值"],
        "止损位": trade_plan["止损位"],
        "第一止盈位": trade_plan["第一止盈位"],
        "第二止盈位": trade_plan["第二止盈位"],
        "建议仓位比例": trade_plan["建议仓位比例"],
    }


def scan_many(
    tickers_text: str,
    portfolio_value: float,
    risk_per_trade: float,
    holding_days: int,
    force_refresh: bool,
    sector_df: pd.DataFrame,
    manual_df: pd.DataFrame,
) -> None:
    tickers = [item.strip().upper() for item in tickers_text.replace("，", ",").split(",") if item.strip()]
    if not tickers:
        st.info("请输入至少一个股票代码。")
        return
    results = []
    progress = st.progress(0, text="正在扫描股票...")
    for index, ticker in enumerate(tickers, start=1):
        sector = get_sector_info(ticker, sector_df)
        manual = get_manual_score(ticker, manual_df)
        stock, market, _ = prepare_market_data(ticker, sector.sector_etf, force_refresh)
        qqq = market.get("QQQ", pd.DataFrame())
        sector_market = market.get(sector.sector_etf, pd.DataFrame())
        if stock.empty:
            results.append({"股票代码": ticker, "操作评级": "未获取到该股票数据，请确认股票代码是否正确。"})
            progress.progress(index / len(tickers), text=f"正在扫描：{ticker}")
            continue
        if qqq.empty:
            results.append({"股票代码": ticker, "操作评级": "行情数据获取失败，请检查股票代码或稍后重试。"})
            progress.progress(index / len(tickers), text=f"正在扫描：{ticker}")
            continue
        if sector_market.empty:
            sector_market = qqq.copy()
        try:
            score = calculate_score(stock, qqq, sector_market, manual)
            trade_plan = calculate_trade_plan(stock, portfolio_value, risk_per_trade)
            odds = calculate_odds(stock, holding_days)
            results.append(
                {
                    "股票代码": ticker,
                    "所属板块": sector.sector,
                    "最新价格": score["最新价格"],
                    "综合评分": score["综合评分"],
                    "操作评级": rating(score["综合评分"]),
                    "技术趋势评分": score["技术趋势评分"],
                    "基本面评分": manual.fundamental_score,
                    "真实瓶颈评分": manual.bottleneck_score,
                    "估值评分": manual.valuation_score,
                    "风险扣分": score["风险扣分"],
                    "二十日胜率": odds["二十日胜率"],
                    "六十日胜率": odds["六十日胜率"],
                    "期望值": odds["期望值"],
                    "止损位": trade_plan["止损位"],
                    "第一止盈位": trade_plan["第一止盈位"],
                    "第二止盈位": trade_plan["第二止盈位"],
                    "建议仓位比例": trade_plan["建议仓位比例"],
                }
            )
        except Exception:
            results.append({"股票代码": ticker, "操作评级": "行情数据获取失败，请检查股票代码或稍后重试。"})
        progress.progress(index / len(tickers), text=f"正在扫描：{ticker}")
    progress.empty()

    result_df = pd.DataFrame(results)
    if result_df.empty:
        st.error("没有可展示的扫描结果。")
        return
    if "综合评分" in result_df.columns:
        result_df = result_df.sort_values("综合评分", ascending=False, na_position="last")

    display_df = result_df.copy()
    for col in ["最新价格", "止损位", "第一止盈位", "第二止盈位"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda value: format_money(float(value)) if pd.notna(value) else "无数据")
    for col in ["二十日胜率", "六十日胜率", "期望值", "建议仓位比例"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda value: format_pct(float(value)) if pd.notna(value) else "无数据")
    if "综合评分" in display_df.columns:
        display_df["综合评分"] = display_df["综合评分"].map(lambda value: format_number(float(value)) if pd.notna(value) else "无数据")

    st.subheader("多股票扫描结果")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.download_button(
        "下载扫描结果",
        data=result_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name="股票扫描结果.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.info("本工具为在线量化研究工具，仅用于辅助决策，不构成投资建议。")

    with st.sidebar:
        st.header("参数设置")
        tickers_text = st.text_area("股票代码输入框", value="MU", height=90)
        portfolio_value = st.number_input("账户规模", min_value=1000.0, value=100000.0, step=5000.0)
        risk_percent = st.number_input("单笔风险比例（百分比）", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        holding_label = st.selectbox("持有周期选择", ["二十日", "六十日", "一百二十日"], index=0)
        mode = st.radio("分析模式", ["单只股票分析", "多股票扫描"], index=0)
        refresh_clicked = st.button("刷新行情数据", use_container_width=True)
        if refresh_clicked:
            st.cache_data.clear()
            st.session_state["下次分析强制刷新"] = True
            st.success("已刷新行情缓存，下次分析将重新获取数据。")
        uploaded_sector = st.file_uploader("上传板块映射表", type=["xlsx", "xls", "csv"])
        uploaded_manual = st.file_uploader("上传人工评分表", type=["xlsx", "xls", "csv"])
        start = st.button("开始分析", type="primary", use_container_width=True)

    holding_days = {"二十日": 20, "六十日": 60, "一百二十日": 120}[holding_label]
    force_refresh = bool(st.session_state.pop("下次分析强制刷新", False)) if start else False
    sector_df, manual_df, messages, warnings = load_config(uploaded_sector, uploaded_manual)
    for message in messages:
        st.sidebar.caption(message)
    for warning in warnings:
        st.sidebar.warning(warning)

    if not start and mode == "单只股票分析":
        st.write("请在左侧输入股票代码后点击“开始分析”。")
    elif start and mode == "单只股票分析":
        ticker = tickers_text.replace("，", ",").split(",")[0].strip().upper()
        if not ticker:
            st.error("请输入股票代码。")
        else:
            analyze_one(ticker, portfolio_value, risk_percent / 100, holding_days, force_refresh, sector_df, manual_df)
    elif start and mode == "多股票扫描":
        scan_many(tickers_text, portfolio_value, risk_percent / 100, holding_days, force_refresh, sector_df, manual_df)
    else:
        st.write("请在左侧输入股票代码后点击“开始分析”。")

    st.divider()
    st.caption(
        "当前免费数据源使用 yfinance，可能不是交易所级实时行情，可能存在延迟。"
        "若需要更实时行情，后续可接入 Finnhub、Polygon、Financial Modeling Prep 等 API。"
    )


if __name__ == "__main__":
    main()
