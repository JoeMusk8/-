# 股票德州扑克量化决策工具

这是一个可以部署到 Streamlit Cloud 的在线量化研究工具。用户输入美股股票代码，例如 MU、NVDA、AVGO、TSM、ETN、COHR、LITE、TSLA，即可查看综合评分、操作评级、止损止盈、仓位建议、胜率赔率和中文解释。

## 在线部署

1. 把本项目上传到 GitHub。
2. 打开 Streamlit Cloud。
3. 选择对应的 GitHub 仓库。
4. Main file path 填写：

```text
app.py
```

5. 点击 Deploy。
6. 部署完成后，会得到一个可以直接访问的网页链接。

## 项目结构

```text
app.py
requirements.txt
README.md
runtime.txt
.gitignore
data/sector_map.csv
data/manual_scores.csv
```

## 数据说明

默认行情数据使用 yfinance，不需要 API Key。免费数据源可能不是交易所级实时行情，可能存在延迟。

如果需要更实时的最新价格，可以在 Streamlit Cloud 的 Secrets 中添加：

```text
FINNHUB_API_KEY = "你的 Finnhub 密钥"
```

没有 FINNHUB_API_KEY 时，程序会自动使用 yfinance，不会报错。

## 免责声明

本工具仅用于在线量化研究和辅助决策，不构成投资建议。市场有风险，交易决策需自行承担。
