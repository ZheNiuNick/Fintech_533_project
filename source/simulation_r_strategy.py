#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:00:00 2024

@author: hakan
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from mplfinance.original_flavor import candlestick_ohlc
import talib
import mplfinance as mpf
import os
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import shutil
from datetime import datetime

def find_time_index(time, df):
    df = pd.DataFrame(df)
    try:
        index = df.index.get_loc(time)
        print(f"找到时间索引: {index}")
        return index
    except:
        print(f"无法找到时间 {time}，使用默认值")
        return 0


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=7492)]
    )
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")


dd = './BTCUSDT_M15_final.csv'

firstDate = "2025-01-03 09:00:00"
lastDate = "2025-04-10 20:00:00"

data = pd.read_csv(dd, delimiter=',', index_col='Time', parse_dates=True)


data['SMA20'] = talib.SMA(data['Close'], timeperiod=20)
data['EMA30'] = talib.EMA(data['Close'], timeperiod=30)

data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(
    data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)

profit_percent = 0;

try:
    initialTime_index = find_time_index(firstDate, data)
    finalTime_index = find_time_index(lastDate, data)
    data = data[initialTime_index:finalTime_index]
except:
    print("use all the data")

output_dir = "test_for_signal"
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)

window_size = 5
shift_size = 2
for i in range(0, len(data) - window_size, shift_size):
    window = data.iloc[i:i+window_size]
    save_path = os.path.join(output_dir, f"{window.iloc[-1].name}.png")
    ap = [mpf.make_addplot(window['SMA20'], color='blue', secondary_y=False)]
    mpf.plot(window, type='candle', style='yahoo', addplot=ap, volume=True, axisoff=True, ylabel='',
            savefig=save_path)
    plt.close()

df = pd.read_csv(dd, delimiter=',', parse_dates=True)
df = df[initialTime_index:finalTime_index]
df = pd.DataFrame(df)
df['Date'] = pd.to_datetime(df['Time'])
df['Date'] = df['Date'].map(mdates.date2num)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
candlestick_ohlc(ax1, df[['Date', 'Open', 'High', 'Low', 'Close']].values, width=0.01, colorup='green', colordown='red')

dataset_path = "test_for_signal"
X = []
image_names = []

for name in sorted(os.listdir(dataset_path)):
    image_path = os.path.join(dataset_path, name)
    image1 = load_img(image_path, color_mode='rgb', interpolation="bilinear", target_size=(150, 150))
    image1 = img_to_array(image1)
    image1 = image1 / 255
    X.append(image1)
    image_names.append(name)

X = np.array(X)

try:
    model = load_model("chart_classification_model_BTC.h5")
    predictions = model.predict(X)

    indicator_xcoordinates = []
    indicator_trends = []

    for idx, i in enumerate(predictions):
        time_str = os.path.splitext(image_names[idx])[0]
        if i >= 0.5:
            indicator_xcoordinates.append(time_str)
            indicator_trends.append("U")
        else:
            indicator_xcoordinates.append(time_str)
            indicator_trends.append("D")

    signal_x = [indicator_xcoordinates[0]]
    signal_label = [indicator_trends[0]]

    for i in range(1, len(indicator_trends)):
        if indicator_trends[i] != indicator_trends[i - 1]:
            signal_x.append(indicator_xcoordinates[i])
            signal_label.append(indicator_trends[i])

    indicator_xcoordinates = signal_x
    indicator_trends = signal_label

    for time, label in zip(indicator_xcoordinates, indicator_trends):
        if time in df['Time'].values:
            result = df.isin([time])
            locations = result.stack()[result.stack()]
            row = locations.index[0][0]
            row = df.loc[row]
            timestamp = mdates.date2num(pd.to_datetime(time))

            if label == 'D':
                y_position = row['High'] + 0.00022
                color = 'red'
            else:
                y_position = row['Low'] - 0.00032
                color = 'green'

            ax1.annotate(label,
                        xy=(timestamp, y_position),
                        xytext=(0, 2),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2',
                                 fc=color,
                                 alpha=0.7)
                        )
except Exception as e:
    print(f"wrong in model generation: {e}")
    indicator_xcoordinates = []
    indicator_trends = []

ax1.xaxis_date()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.set_title('BTC/USDT with Trend Signals')
ax1.set_ylabel('Price')
ax1.grid(True)

df_macd = data.reset_index()
dates = [mdates.date2num(d) for d in pd.to_datetime(df_macd['Time'])]
ax2.bar(dates, df_macd['MACD_hist'], color=['green' if x >= 0 else 'red' for x in df_macd['MACD_hist']], alpha=0.5)
ax2.plot(dates, df_macd['MACD'], color='blue', label='MACD')
ax2.plot(dates, df_macd['MACD_signal'], color='red', label='Signal')
ax2.set_ylabel('MACD')
ax2.grid(True)
ax2.xaxis_date()
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax2.legend()

plt.tight_layout()
plt.savefig('trading_chart.png', dpi=300)
plt.show()

# 创建交易结果文件
trade_log_filename = f"trade_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(trade_log_filename, 'w') as f:
    f.write("=== MACD+ATR平衡策略交易记录 ===\n\n")
    f.write(f"数据源: {dd}\n")
    f.write(f"时间范围: {data.index[0]} 至 {data.index[-1]}\n\n")
    f.write("策略参数:\n")
    f.write("- MACD: (12,26,9)\n")
    f.write("- EMA: 50日\n")
    f.write("- ATR: 14日\n")
    f.write("- ATR乘数: 2.5\n")
    f.write("- 信号确认: 需要2个确认\n\n")

# ============ 交易策略执行 ============
# 策略参数
initial_amount_usd = 1000000  # 初始资金
current_amount_usd = initial_amount_usd
amount_in_btc = 0  # 持仓数量

# 关键策略参数
atr_multiplier = 3.2  # ATR倍数，控制止损宽度
trailing_stop = True  # 启用追踪止损
signal_confirmation = 2  # 需要连续信号确认数

stop_loss_price = 0  # 止损价格
entry_price = 0  # 入场价格
peak_price = 0  # 持仓后的最高价
confirmation_count = 0  # 信号确认计数
current_signal = None  # 当前信号

number_changes = 0  # 交易次数
trades = []  # 交易记录

# 模拟交易
print("\ntrading begin...\n")
with open(trade_log_filename, 'a') as f:
    f.write("===== trading record =====\n\n")

for i in range(1, len(data)):
    current_date = data.index[i]
    current_price = data['Close'].iloc[i]
    current_atr = data['ATR'].iloc[i]

    if data['MACD_hist'].iloc[i] > 0 and data['Close'].iloc[i] > data['EMA30'].iloc[i]:
        trend_signal = "U"
    elif data['MACD_hist'].iloc[i] < 0 and data['Close'].iloc[i] < data['EMA30'].iloc[i]:
        trend_signal = "D"
    else:
        trend_signal = "N"

    if current_signal != trend_signal:
        current_signal = trend_signal
        confirmation_count = 1
    else:
        confirmation_count += 1

    if amount_in_btc > 0 and trailing_stop:
        if current_price > peak_price:
            peak_price = current_price
            stop_loss_price = peak_price - (atr_multiplier * current_atr)

    if amount_in_btc > 0 and current_price < stop_loss_price:
        profit_percent = ((current_price / entry_price) - 1) * 100
        current_amount_usd = amount_in_btc * current_price

        trade_info = f"stop loss sell - date: {current_date}, amount remaining: ${current_amount_usd:.2f}, profit and loss: {profit_percent:.2f}%"
        print(trade_info)

        with open(trade_log_filename, 'a') as f:
            f.write(f"{trade_info}\n")
            f.write(f"remaing amount: ${current_amount_usd:.2f}\n\n")

        trades.append({
            'type': 'stop-loss sell',
            'date': current_date,
            'price': current_price,
            'profit_percent': profit_percent
        })

        amount_in_btc = 0
        number_changes += 1
        confirmation_count = 0

    # 买入信号处理
    elif trend_signal == "U" and confirmation_count >= signal_confirmation and current_amount_usd > 0:
        entry_price = current_price
        amount_in_btc = current_amount_usd / entry_price
        current_amount_usd = 0

        # 设置初始止损
        stop_loss_price = entry_price - (atr_multiplier * current_atr)
        peak_price = entry_price

        trade_info = f"stop loss sell - date: {current_date}, amount remaining: ${current_amount_usd:.2f}, profit and loss: {profit_percent:.2f}%"
        print(trade_info)

        with open(trade_log_filename, 'a') as f:
            f.write(f"{trade_info}\n")
            f.write(f"stop loss sell: ${stop_loss_price:.2f}\n\n")

        trades.append({
            'type': 'buy in',
            'date': current_date,
            'price': entry_price,
            'profit_percent': profit_percent
        })

        number_changes += 1

    # 卖出信号处理
    elif trend_signal == "D" and confirmation_count >= signal_confirmation and amount_in_btc > 0:
        profit_percent = ((current_price / entry_price) - 1) * 100
        current_amount_usd = amount_in_btc * current_price

        trade_info = f"stop loss sell - date: {current_date}, amount remaining: ${current_amount_usd:.2f}, profit and loss: {profit_percent:.2f}%"
        print(trade_info)

        with open(trade_log_filename, 'a') as f:
            f.write(f"{trade_info}\n")
            f.write(f"remaining amount: ${current_amount_usd:.2f}\n\n")

        trades.append({
            'type': 'sold out',
            'date': current_date,
            'price': current_price,
            'profit_percent': profit_percent
        })

        amount_in_btc = 0
        number_changes += 1
        confirmation_count = 0


if amount_in_btc > 0:
    final_price = data['Close'].iloc[-1]
    profit_percent = ((final_price / entry_price) - 1) * 100
    current_amount_usd = amount_in_btc * final_price

    trade_info = f"stop loss sell - date: {current_date}, amount remaining: ${current_amount_usd:.2f}, profit and loss: {profit_percent:.2f}%"
    print(trade_info)

    with open(trade_log_filename, 'a') as f:
        f.write(f"{trade_info}\n")
        f.write(f"total amount : ${current_amount_usd:.2f}\n\n")

    trades.append({
        'type': 'total amount',
        'date': data.index[-1],
        'price': final_price,
        'profit_percent': profit_percent
    })

# 计算策略统计数据
total_return = (current_amount_usd / initial_amount_usd - 1) * 100
win_trades = [t for t in trades if t.get('profit_percent', 0) > 0]
loss_trades = [t for t in trades if t.get('profit_percent', 0) <= 0]

win_count = len(win_trades)
loss_count = len(loss_trades)
total_trades = win_count + loss_count

win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

avg_win = sum(t['profit_percent'] for t in win_trades) / win_count if win_count > 0 else 0
avg_loss = sum(t['profit_percent'] for t in loss_trades) / loss_count if loss_count > 0 else 0

profit_factor = abs(sum(t['profit_percent'] for t in win_trades) / sum(t['profit_percent'] for t in loss_trades)) if sum(t['profit_percent'] for t in loss_trades) != 0 else float('inf')

# Output final results
print(f"Initial Capital: ${initial_amount_usd:.2f}")
print(f"Final Capital:   ${current_amount_usd:.2f}")
print(f"Total Return:    {total_return:.2f}%")
print(f"Trades Executed: {total_trades}")
print(f"Win Rate:        {win_rate:.2f}%")
print(f"Avg. Profit:     {avg_win:.2f}%")
print(f"Avg. Loss:       {avg_loss:.2f}%")
print(f"Profit Factor:   {profit_factor:.2f}")

# Write performance summary to log
with open(trade_log_filename, 'a') as f:
    f.write("\n===== Strategy Performance =====\n")
    f.write(f"Initial Capital: ${initial_amount_usd:.2f}\n")
    f.write(f"Final Capital:   ${current_amount_usd:.2f}\n")
    f.write(f"Total Return:    {total_return:.2f}%\n")
    f.write(f"Trades Executed: {total_trades}\n")
    f.write(f"Win Rate:        {win_rate:.2f}%\n")
    f.write(f"Avg. Profit:     {avg_win:.2f}%\n")
    f.write(f"Avg. Loss:       {avg_loss:.2f}%\n")
    f.write(f"Profit Factor:   {profit_factor:.2f}\n")

print(f"\nTrade log saved to {trade_log_filename}")


# Build DataFrame of all executed trades
trade_df = pd.DataFrame(trades)

# 1) Price chart with Buy/Sell markers
plt.figure(figsize=(12, 5))
plt.plot(data.index, data['Close'], label='Close Price', linewidth=1)

# Plot buy‑in points (type == 'buy in')
buys = trade_df[trade_df['type'] == 'buy in']
plt.scatter(pd.to_datetime(buys['date']), buys['price'],
            marker='^', color='green', s=100, label='Buy')

# Plot sell points (stop‑loss and regular sells)
sells = trade_df[trade_df['type'].isin(['sold out', 'stop-loss sell', 'final close'])]
plt.scatter(pd.to_datetime(sells['date']), sells['price'],
            marker='v', color='red', s=100, label='Sell')

plt.title('BTC/USDT Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === Generate Blotter ===
blotter_records = []
for trade in trades:
    if trade['type'] in ['buy in', 'sold out', 'stop-loss sell', 'total amount']:
        blotter_records.append({
            'timestamp': trade['date'],
            'action': trade['type'],
            'price': trade['price'],
            'profit_percent': trade.get('profit_percent', None),
            'btc_qty': round(
                trade['price'] and (initial_amount_usd / trade['price']) if trade['type'] == 'buy in' else 0, 6)
        })

blotter_df = pd.DataFrame(blotter_records)
blotter_df.to_csv('blotter.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt

balance = initial_amount_usd
btc_holding = 0
ledger_records = []

for trade in trades:
    date = trade['date']
    price = trade['price']
    if trade['type'] == 'buy in':
        btc_holding = balance / price
        balance = 0
    elif trade['type'] in ['sold out', 'stop-loss sell', 'total amount']:
        balance = btc_holding * price
        btc_holding = 0

    market_value = balance + btc_holding * price
    ledger_records.append({
        'timestamp': date,
        'cash': round(balance, 2),
        'btc_holding': round(btc_holding, 6),
        'price': round(price, 2),
        'market_value': round(market_value, 2)
    })

ledger_df = pd.DataFrame(ledger_records)
ledger_df['timestamp'] = pd.to_datetime(ledger_df['timestamp'])

ledger_df.to_csv('ledger.csv', index=False)

plt.figure(figsize=(10, 4))
plt.plot(ledger_df['timestamp'], ledger_df['market_value'], marker='o', linestyle='-')
plt.title('Remaining USD (Market Value) After Each Trade')
plt.xlabel('Date')
plt.ylabel('Balance (USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



