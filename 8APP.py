import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from tvDatafeed import TvDatafeed, Interval
from datetime import datetime, time as datetime_time
from math import floor
import time



SYMBOL = "XAUUSDm" 
MAGIC_NUMBER = 123456
RISK_PERCENT = 0.05
CHECK_INTERVAL_SECONDS = 60 * 60
DATA_RETRY_SECONDS = 6
CHECK_DISPLAY_DELAY_SECONDS = 2
TRADING_START_TIME = datetime_time(10, 0)
TRADING_END_TIME = datetime_time(23, 55)



USER_NAME = "JESSE"
DEVELOPER_NAME = "JESSE NJOROGE GITAKA"
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"


MT5_LOGIN = 435783355
MT5_PASSWORD = "jesse150"
MT5_SERVER = "ExnessKE-MT5Trial9"


def color_text(text, color):
    return f"{color}{text}{RESET}"


def print_banner():
    print(color_text("\n============================================================", CYAN))
    print(color_text("XAUUSD MARKOV BOT", BOLD + GREEN))
    print(color_text("XAUUSD MARKOV BOT", BOLD + GREEN))
    print(color_text("XAUUSD MARKOV BOT", BOLD + GREEN))
    print(color_text(f"Developer: {DEVELOPER_NAME}", BOLD + CYAN))
    print(color_text(f"Welcome to today, {USER_NAME}. Wish you all the best with today's trades.", BOLD + GREEN))
    print(color_text("Risk disclaimer: Trading involves risk. Losses can happen, so only trade with money you can afford to lose.", YELLOW))
    print(color_text("============================================================\n", CYAN))

def print_check(label, status="CHECK", detail="", delay=True):
    colors = {
        "CHECK": GREEN,
        "WAIT": YELLOW,
        "FAIL": RED,
        "INFO": CYAN,
        "RUN": BLUE,
    }
    color = colors.get(status, CYAN)
    detail_text = f" - {detail}" if detail else ""
    print(f"{color_text(f'[{status}]', color)} {label}{detail_text}")
    if delay:
        time.sleep(CHECK_DISPLAY_DELAY_SECONDS)

def print_section(title):
    print(color_text(f"\n--- {title} ---", MAGENTA))



class MarkovRegime:
    def __init__(self):
        self.n_states = 3
        self.current_state = 0 
        self.colors = ['#3fb950', '#d29922', '#f85149'] 
        self.bg_colors = ['rgba(63, 185, 80, 0.25)', 'rgba(210, 153, 34, 0.25)', 'rgba(248, 81, 73, 0.25)']
        self.state_probs = np.array([1/3, 1/3, 1/3])
        self.transition_matrix = np.array([
            [0.90, 0.08, 0.02],  
            [0.10, 0.80, 0.10],  
            [0.02, 0.08, 0.90]   
        ])
        self.emission_means = np.array([0.0005, 0.002, 0.005])
        self.emission_stds = np.array([0.0003, 0.001, 0.003])

    def calibrate(self, hist_bars):
        if len(hist_bars) < 20: return
        vols = np.array([(b['h'] - b['l']) / b['c'] if b['c'] > 0 else 0 for b in hist_bars])
        vols = vols[vols > 0]
        if len(vols) < 20: return

        p33, p67 = np.percentile(vols, 33), np.percentile(vols, 67)
        regime_assignments = np.zeros(len(vols), dtype=int)
        regime_assignments[vols >= p33] = 1
        regime_assignments[vols >= p67] = 2

        for regime in range(self.n_states):
            regime_vols = vols[regime_assignments == regime]
            if len(regime_vols) >= 3:
                self.emission_means[regime] = np.mean(regime_vols)
                self.emission_stds[regime] = max(np.std(regime_vols), 1e-6)

        sorted_indices = np.argsort(self.emission_means)
        self.emission_means = self.emission_means[sorted_indices]
        self.emission_stds = self.emission_stds[sorted_indices]

        transition_counts = np.zeros((self.n_states, self.n_states))
        for t in range(1, len(regime_assignments)):
            prev_regime = regime_assignments[t-1]
            curr_regime = regime_assignments[t]
            transition_counts[prev_regime, curr_regime] += 1

        for i in range(self.n_states):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                self.transition_matrix[i] = (transition_counts[i] + 0.1) / (row_sum + 0.3)
        self.state_probs = np.array([1/3, 1/3, 1/3])

    def _gaussian_likelihood(self, vol, regime):
        mean = self.emission_means[regime]
        std = self.emission_stds[regime]
        coeff = 1 / (std * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((vol - mean) / std) ** 2
        return coeff * np.exp(exponent)

    def get_regime(self, bars):
        if not bars: return self.current_state
        current_bar = bars[-1]
        vol = current_bar.volatility
        if vol <= 0:
            current_bar.regime = self.current_state
            return self.current_state

        prior_probs = self.transition_matrix.T @ self.state_probs
        likelihoods = np.array([self._gaussian_likelihood(vol, i) for i in range(self.n_states)])
        posterior_probs = prior_probs * likelihoods

        prob_sum = posterior_probs.sum()
        if prob_sum > 0:
            posterior_probs = posterior_probs / prob_sum
        else:
            posterior_probs = prior_probs

        self.state_probs = posterior_probs
        self.current_state = int(np.argmax(posterior_probs))
        current_bar.regime = self.current_state
        return self.current_state

class Bar:
    def __init__(self, datetime, open_price, high, low, close, volume):
        self.datetime = datetime
        self.o = open_price
        self.h = high
        self.l = low
        self.c = close
        self.v = volume
        self.volatility = (high - low) / close if close > 0 else 0
        self.regime = 0


def init_datafeed():
    tv = TvDatafeed()
    return tv

def get_hist_with_retry(tv_session, symbol, exchange, bars_to_pull):
    while True:
        try:
            df = tv_session.get_hist(symbol, exchange, interval=Interval.in_daily, n_bars=bars_to_pull)
            if df is not None and not df.empty:
                return df
            print_check("TradingView data fetch", "WAIT", f"{symbol} returned no data. Retrying in {DATA_RETRY_SECONDS} seconds", delay=False)
        except Exception as exc:
            print_check("TradingView data fetch", "WAIT", f"{symbol} error: {exc}. Retrying in {DATA_RETRY_SECONDS} seconds", delay=False)

        time.sleep(DATA_RETRY_SECONDS)

def fetch_market_GOLD(tv_session,symbol='GOLD', exchange='TVC', bars_to_pull=600):
    df = get_hist_with_retry(tv_session, symbol, exchange, bars_to_pull)
    df.index = pd.to_datetime(df.index).date 
    df.columns = ['Symbol','XAUUSD_Open', 'XAUUSD_High', 'XAUUSD_Low', 'XAUUSD_Close', 'XAUUSD_Volume']
    # CHANGED: We now need OHLC, not just Close, for the Volatility formula
    df = df[['XAUUSD_Open', 'XAUUSD_High', 'XAUUSD_Low', 'XAUUSD_Close']].dropna()
    return df

def fetch_market_DXY(tv_session,symbol='DXY', exchange='TVC', bars_to_pull=600):
    df = get_hist_with_retry(tv_session, symbol, exchange, bars_to_pull)
    df.index = pd.to_datetime(df.index).date 
    df.columns = ['Symbol','DXY_Open', 'DXY_High', 'DXY_Low', 'DXY_Close', 'DXY_Volume']
    df = df[["DXY_Close"]].dropna()
    return df 

def fetch_market_SPX(tv_session,symbol='SPX', exchange='SP', bars_to_pull=600):
    df = get_hist_with_retry(tv_session, symbol, exchange, bars_to_pull)
    df.index = pd.to_datetime(df.index).date 
    df.columns = ['Symbol','SPX_Open', 'SPX_High', 'SPX_Low', 'SPX_Close', 'SPX_Volume']
    df = df[["SPX_Close"]].dropna()
    return df 

def fetch_market_US10Y(tv_session,symbol='US10Y', exchange='TVC', bars_to_pull=600):
    df = get_hist_with_retry(tv_session, symbol, exchange, bars_to_pull)
    df.index = pd.to_datetime(df.index).date 
    df.columns = ['Symbol','US10Y_Open', 'US10Y_High', 'US10Y_Low', 'US10Y_Close', 'US10Y_Volume']
    df = df[["US10Y_Close"]].dropna()
    return df 

def fetch_market_VIX(tv_session,symbol='VIX', exchange='CBOE', bars_to_pull=600):
    df = get_hist_with_retry(tv_session, symbol, exchange, bars_to_pull)
    df.index = pd.to_datetime(df.index).date 
    df.columns = ['Symbol','VIX_Open', 'VIX_High', 'VIX_Low', 'VIX_Close', 'VIX_Volume']
    df = df[["VIX_Close"]].dropna()
    return df
    
def fetch_market_data():
    session = init_datafeed()
    df_XAUUSD =   fetch_market_GOLD(session)
    time.sleep(1)
    df_DXY =   fetch_market_DXY(session)
    time.sleep(1)
    df_SPX =   fetch_market_SPX(session)
    time.sleep(1)
    df_US10Y =   fetch_market_US10Y(session)
    time.sleep(1)
    df_VIX =   fetch_market_VIX(session)
    
    df_market_data = pd.concat([df_XAUUSD, df_DXY, df_SPX, df_US10Y, df_VIX], axis=1, join='inner')
    return df_market_data

def calculate_features(df_market_data):
    df = df_market_data.copy()
    df['XAU_Returns'] = np.log(df['XAUUSD_Close'] / df['XAUUSD_Close'].shift(1))
    df['DXY_Returns'] = np.log(df['DXY_Close'] / df['DXY_Close'].shift(1))
    df['SPX_Returns'] = np.log(df['SPX_Close'] / df['SPX_Close'].shift(1))

    df['US10Y_Diff'] = df['US10Y_Close'].diff()
    df['VIX_Diff'] = df['VIX_Close'].diff()

    df['XAU_Returns_Lag1'] = df['XAU_Returns'].shift(1)
    df_features = df.dropna()

    return df_features

def detect_market_regime(df_features):
    """
    CHANGED: Replaces statsmodels. Uses the new Volatility Markov Model.
    Calibrates on recent history and filters to find today's regime.
    """
    # Use the last 100 days to calibrate the model effectively
    historical_data = df_features.iloc[-100:]
    
    bars = []
    for date_index, row in historical_data.iterrows():
        bar = Bar(
            datetime=date_index,
            open_price=row['XAUUSD_Open'],
            high=row['XAUUSD_High'],
            low=row['XAUUSD_Low'],
            close=row['XAUUSD_Close'],
            volume=0
        )
        bars.append(bar)

    regime_model = MarkovRegime()
    
  
    calibration_size = min(100, len(bars))
    regime_model.calibrate([{'h': b.h, 'l': b.l, 'c': b.c} for b in bars[:calibration_size]])
    
    
    current_regime = regime_model.get_regime(bars)
    
    return current_regime

def calculate_drift_signal(regime, df_features):
   
    latest = df_features.iloc[-1]
    
    coef_map = {
        0: {'intercept': 0.000409, 'xau': 0.130505, 'dxy': 0.207511, 'spx': -0.035197, 'us10y': 0.006922, 'vix': -0.000186},
        1: {'intercept': 0.000025, 'xau': 0.012422, 'dxy': 0.121750, 'spx': 0.027322, 'us10y': 0.010282, 'vix': 0.000118},
        2: {'intercept': 0.000468, 'xau': 0.015236, 'dxy': -0.048405, 'spx': -0.114452, 'us10y': -0.000895, 'vix': -0.000736}
    }
    
    beta = coef_map[regime]
    
    drift = (
        beta['intercept'] +
        (latest['XAU_Returns_Lag1'] * beta['xau']) +
        (latest['DXY_Returns'] * beta['dxy']) +
        (latest['SPX_Returns'] * beta['spx']) +
        (latest['US10Y_Diff'] * beta['us10y']) +
        (latest['VIX_Diff'] * beta['vix'])
    )
    
    return drift


def get_atr(symbol, period=14):
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        return None
        
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, period + 1)
    if rates is None or len(rates) < period:
        print_check("4H ATR check", "FAIL", f"Failed to get rates. Error: {mt5.last_error()}")
        return None

    df_atr = pd.DataFrame(rates)
    df_atr['prev_close'] = df_atr['close'].shift(1)
    df_atr['tr'] = np.maximum(df_atr['high'] - df_atr['low'], 
                     np.maximum(abs(df_atr['high'] - df_atr['prev_close']), 
                                abs(df_atr['low'] - df_atr['prev_close'])))
    
    atr = df_atr['tr'].rolling(window=period).mean().iloc[-1]
    return atr

def normalize_lot_size(lot, symbol_info):
    volume_step = symbol_info.volume_step
    volume_min = symbol_info.volume_min
    volume_max = symbol_info.volume_max

    lot = floor(lot / volume_step) * volume_step
    lot = max(volume_min, min(lot, volume_max))

    step_text = f"{volume_step:.10f}".rstrip("0")
    decimals = len(step_text.split(".")[1]) if "." in step_text else 0
    return round(lot, decimals)

def calculate_lot_size(symbol_info, entry_price, stop_loss):
    account_info = mt5.account_info()
    if account_info is None:
        return None, None

    balance = account_info.balance
    risk_amount = balance * RISK_PERCENT
    stop_distance = abs(entry_price - stop_loss)

    if stop_distance <= 0 or symbol_info.trade_tick_size <= 0 or symbol_info.trade_tick_value <= 0:
        return None, risk_amount

    ticks_to_stop = stop_distance / symbol_info.trade_tick_size
    risk_per_lot = ticks_to_stop * symbol_info.trade_tick_value

    if risk_per_lot <= 0:
        return None, risk_amount

    lot = normalize_lot_size(risk_amount / risk_per_lot, symbol_info)
    return lot, risk_amount

def has_traded_today(symbol):
    today_start = datetime.combine(datetime.now().date(), datetime.min.time())
    deals = mt5.history_deals_get(today_start, datetime.now())

    if deals is None:
        return False

    for deal in deals:
        if deal.symbol == symbol and deal.magic == MAGIC_NUMBER:
            return True

    return False

def is_trading_session():
    now_time = datetime.now().time()
    return TRADING_START_TIME <= now_time < TRADING_END_TIME

def should_close_for_day():
    return datetime.now().time() >= TRADING_END_TIME

def seconds_until_next_check():
    close_datetime = datetime.combine(datetime.now().date(), TRADING_END_TIME)
    seconds_to_close = (close_datetime - datetime.now()).total_seconds()

    if seconds_to_close > 0:
        return max(1, min(CHECK_INTERVAL_SECONDS, int(seconds_to_close)))

    return CHECK_INTERVAL_SECONDS

def close_bot_positions():
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print_check("End-of-day close check", "FAIL", f"Login error code: {mt5.last_error()}")
        return False

    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        print_check("End-of-day close check", "FAIL", f"Could not get open positions. Error: {mt5.last_error()}")
        mt5.shutdown()
        return False

    closed_any = False

    for position in positions:
        if position.magic != MAGIC_NUMBER:
            continue

        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            print_check("End-of-day close check", "FAIL", "Could not get symbol tick to close position")
            continue

        if position.type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            close_price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            close_price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": close_price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "End of day close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print_check("End-of-day close check", "CHECK", f"Closed position {position.ticket}")
            closed_any = True
        else:
            print_check("End-of-day close check", "FAIL", f"Position {position.ticket}, retcode={result.retcode} | {result.comment}")

    mt5.shutdown()
    return closed_any

def execute_trade(drift_prediction, current_regime):
    if not is_trading_session():
        print_check("Trading hours check", "WAIT", "Outside London-to-New-York session")
        return False

    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print_check("MT5 login check", "FAIL", f"Error code: {mt5.last_error()}")
        return False

    terminal_info = mt5.terminal_info()
    if terminal_info is None or not terminal_info.connected:
        print_check("MT5 connection check", "FAIL", "Not connected to server")
        mt5.shutdown()
        return False

    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print_check("Symbol check", "FAIL", f"{SYMBOL} not found")
        mt5.shutdown()
        return False

    if has_traded_today(SYMBOL):
        print_check("Daily trade limit check", "WAIT", "Already traded today")
        mt5.shutdown()
        return False
    
    digits = symbol_info.digits 
    atr_value = get_atr(SYMBOL, period=14)
    
    if atr_value is None:
        print_check("4H ATR check", "FAIL", "Could not calculate ATR")
        mt5.shutdown()
        return False

    threshold = 0.0005
    
    print_section("Trade Logic")
    print_check("Trading hours check", "CHECK", f"{TRADING_START_TIME.strftime('%H:%M')} to {TRADING_END_TIME.strftime('%H:%M')} Nairobi time")
    print_check("Regime check", "INFO", f"Current regime: {current_regime}")
    print_check("Signal generation check", "CHECK", f"Drift signal: {drift_prediction:.6f}")
    print_check("4H ATR check", "CHECK", f"ATR: {atr_value:.2f}")
    
    if drift_prediction > threshold:
        order_type = mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            print_check("Price tick check", "FAIL", "Could not get current symbol tick")
            mt5.shutdown()
            return False
        price = tick.ask
        sl = round(price - (1 * atr_value), digits)
        tp = round(price + (2 * atr_value), digits)
        print_check("Trade action check", "CHECK", f"BUY | SL: {sl} | TP: {tp}")

    elif drift_prediction < -threshold:
        order_type = mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            print_check("Price tick check", "FAIL", "Could not get current symbol tick")
            mt5.shutdown()
            return False
        price = tick.bid
        sl = round(price + (1 * atr_value), digits)
        tp = round(price - (2 * atr_value), digits)
        print_check("Trade action check", "CHECK", f"SELL | SL: {sl} | TP: {tp}")

    else:
        print_check("Trade action check", "WAIT", "Signal below threshold")
        mt5.shutdown()
        return False

    lot, risk_amount = calculate_lot_size(symbol_info, price, sl)

    if lot is None:
        print_check("Risk and lot check", "FAIL", "Could not calculate lot size from account balance and symbol tick values")
        mt5.shutdown()
        return False

    print_check("Risk check", "CHECK", f"${risk_amount:.2f} risk ({RISK_PERCENT * 100:.1f}% of account)")
    print_check("Lot size check", "CHECK", f"{lot} lots")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,           
        "tp": tp,           
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": f"Regime {current_regime} Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC, 
    }

    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print_check("Order execution check", "FAIL", f"retcode={result.retcode} | {result.comment}")
        trade_placed = False
    else:
        print_check("Order execution check", "CHECK", f"Trade {result.order} placed")
        trade_placed = True

    mt5.shutdown()
    return trade_placed


def run_bot_cycle():
    print_section("XAUUSD Markov Bot Cycle")

    print_check("Market data check", "RUN", "Fetching TradingView data", delay=False)
    data = fetch_market_data()
    print_check("Market data check", "CHECK", "Data received")

    print_check("Feature calculation check", "RUN", "Calculating features")
    feats = calculate_features(data)
    print_check("Feature calculation check", "CHECK", f"{len(feats)} usable rows")

    print_check("Regime detection check", "RUN", "Detecting market regime")
    state = detect_market_regime(feats)
    print_check("Regime detection check", "CHECK", f"Regime {state}")

    print_check("Signal generation check", "RUN", "Calculating drift signal")
    signal = calculate_drift_signal(state, feats)
    print_check("Signal generation check", "CHECK", f"Signal {signal:.6f}")

    print_check("Trade execution check", "RUN", "Checking entry rules")
    trade_placed = execute_trade(signal, state)

    print_section("Cycle Finished")
    return trade_placed


if __name__ == "__main__":
    last_trade_date = None
    last_close_date = None

    print_banner()
    print_section("Startup Checks")
    print_check("User check", "CHECK", USER_NAME)
    print_check("Symbol check", "INFO", SYMBOL)
    print_check("Risk settings check", "CHECK", f"{RISK_PERCENT * 100:.1f}% risk per trade")
    print_check("Trading hours check", "INFO", f"{TRADING_START_TIME.strftime('%H:%M')} to {TRADING_END_TIME.strftime('%H:%M')} Nairobi time")
    print_check("Loop check", "CHECK", f"Runs every {CHECK_INTERVAL_SECONDS // 60} minutes")

    while True:
        today = datetime.now().date()

        if should_close_for_day() and last_close_date != today:
            print_check("End-of-day check", "RUN", f"{today} close time reached")
            close_bot_positions()
            last_close_date = today

        if last_trade_date == today:
            print_check("Daily trade limit check", "WAIT", f"Trade already placed for {today}")
            time.sleep(seconds_until_next_check())
            continue

        if not is_trading_session():
            print_check("Trading hours check", "WAIT", "Outside London-to-New-York session")
            time.sleep(seconds_until_next_check())
            continue

        try:
            trade_placed = run_bot_cycle()
            if trade_placed:
                last_trade_date = today
        except Exception as exc:
            print_check("Bot cycle check", "FAIL", str(exc))

        time.sleep(seconds_until_next_check())





   
