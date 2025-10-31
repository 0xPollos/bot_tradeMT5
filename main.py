import os
import json
import time
import colorama
from colorama import Fore, Style
import threading
import metatrader5 as mt5
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
import yfinance as yf
from google import genai
from google.genai.errors import APIError
from google.genai.types import GenerationConfig
from analyzers.market_analyzer import MarketAnalyzer
from trading.trade_manager import TradeManager
from bot.bot_manager import TradingBot
from utils.symbol_detector import SymbolDetector
from typing import Optional, Dict, Any


colorama.init(autoreset=True)

# Membuat string dengan warna, gaya, dan emoji
# 🎨 Warna dan gaya untuk "Author"
author_text = Fore.CYAN + Style.BRIGHT + "Author" + Style.RESET_ALL
# 📘 Warna dan emoji untuk FB
fb_info = Fore.BLUE + Style.BRIGHT + "👾 Pollos" + Style.RESET_ALL
# ✈️ Warna dan emoji untuk Tele
tele_info = Fore.MAGENTA + Style.BRIGHT + "✈️ Tele @RayyXBT" + Style.RESET_ALL

# --------------------------
# 1. LOAD KONFIG & ENV
# --------------------------
def load_environment():
    """
    Load .env and return a dict with multiple key formats to avoid
    mismatches between UPPER and snake_case usages in the code.
    """
    load_dotenv()  # load .env into os.environ

    # Basic presence debug (masked) - will not print the full key
    raw_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek_api_key")
    if raw_key:
        masked = raw_key[:4] + "..." + raw_key[-4:] if len(raw_key) > 8 else "****"
        print(f"\n✅ DEEPSEEK_API_KEY found in environment (masked): {masked}")
    else:
        print("\n⚠️ DEEPSEEK_API_KEY not found in environment variables")

    env = {
        # UPPER keys (if other parts of code expect these)
        "MT5_LOGIN": os.getenv("MT5_LOGIN"),
        "MT5_PASSWORD": os.getenv("MT5_PASSWORD"),
        "MT5_SERVER": os.getenv("MT5_SERVER"),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
        "NEWS_API_KEY": os.getenv("NEWS_API_KEY"),
        "TRADING_ECONOMICS_KEY": os.getenv("TRADING_ECONOMICS_KEY", "guest:guest"),

        # snake_case keys (common in this project)
        "mt5_login": int(os.getenv("MT5_LOGIN")) if os.getenv("MT5_LOGIN") and os.getenv("MT5_LOGIN").isdigit() else 0,
        "mt5_password": os.getenv("MT5_PASSWORD", ""),
        "mt5_server": os.getenv("MT5_SERVER", ""),
        "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY"),
        "news_api_key": os.getenv("NEWS_API_KEY", ""),
        "trading_economics_key": os.getenv("TRADING_ECONOMICS_KEY", "guest:guest"),
    }
    return env


def load_config(config_path: str = "config.json") -> dict:
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: config.json tidak ditemukan!")
        return None


def save_config(config: dict, config_path: str = "config.json") -> None:
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Konfigurasi disimpan ke {config_path}")
    except Exception as e:
        print(f"❌ Gagal menyimpan konfigurasi: {str(e)}")

def init_deepseek() -> Optional[Dict[str, Any]]:
    """Inisialisasi DeepSeek API"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key or api_key == "":
        print("\n⚠️ DEEPSEEK_API_KEY tidak ditemukan di .env!")
        return None
        
    try:
        # Test connection dengan request sederhana
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": "Test connection - reply with 'OK' only"}
            ],
            "max_tokens": 5,
            "temperature": 0.1,
            "stream": False
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                print("\n✅ DeepSeek API siap digunakan!")
                return {
                    "api_key": api_key,
                    "base_url": "https://api.deepseek.com/v1/chat/completions",
                    "model": "deepseek-chat",
                    "is_active": True
                }
        else:
            print(f"\n❌ Gagal koneksi ke DeepSeek: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Gagal inisialisasi DeepSeek: {e}")
        print("ℹ️ Bot akan berjalan tanpa analisis AI")
    
    return None

# Inisialisasi global
deepseek_config = None

def get_conversion_rate(currency: str) -> float:
    """Mendapatkan rate konversi USD ke mata uang akun."""
    if currency == "USD":
        return 1.0
    
    pair = f"USD{currency}=X" # Cth: USDIDR=X
    
    try:
        # Coba ambil dari MT5 dulu (jika ada, cth: USDIDR)
        symbol_name = f"USD{currency}"
        if mt5.symbol_select(symbol_name, True):
            tick = mt5.symbol_info_tick(symbol_name)
            if tick and tick.ask > 0:
                print(f"   Rate from MT5 ({symbol_name}): {tick.ask}")
                return tick.ask
    except:
        pass # Lanjut ke yfinance
    
    try:
        # Coba yfinance (butuh koneksi internet)
        data = yf.Ticker(pair).history(period="1d")
        if not data.empty:
            rate = data['Close'].iloc[-1]
            print(f"   Rate from yfinance ({pair}): {rate:.2f}")
            return rate
    except Exception as e:
        print(f"   yfinance error (perlu internet): {e}")
    
    # Fallback jika semua gagal
    fallback_rate = 16000.0 if currency == "IDR" else 1.0
    print(f"⚠️ Gagal ambil rate, pakai fallback: {fallback_rate}")
    return fallback_rate

# --------------------------
# 2. KONEKSI & UTIL MT5
# --------------------------
env = load_environment()
mt5_terhubung = False
symbol_detector = None

def get_available_symbols() -> list:
    """Get list of available symbols with variations"""
    symbols = []
    if mt5_terhubung:
        # Get all symbols
        all_symbols = mt5.symbols_get()
        if all_symbols:
            for symbol_info in all_symbols:
                name = symbol_info.name
                # Check for common forex pairs and variations
                if any(pair in name for pair in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD', 'BTCUSD']):
                    symbols.append(name)
    return symbols

def init_mt5() -> bool:
    """Initialize MT5 connection with auto symbol detection"""
    global mt5_terhubung
    
    # Shutdown existing connection if any
    if mt5.initialize():
        mt5.shutdown()
    
    # Try multiple MT5 installation paths
    mt5_paths = [
        "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
        "C:\\Program Files (x86)\\MetaTrader 5\\terminal64.exe",
        "C:\\Program Files\\MetaTrader 5 IC Markets\\terminal64.exe",  # IC Markets
        "C:\\Program Files\\Admiral Markets MT5\\terminal64.exe",  # Admiral
        None  # Let MT5 find the path automatically
    ]
    
    connection_success = False
    
    for path in mt5_paths:
        try:
            # Attempt connection
            init_result = mt5.initialize(
                login=env["mt5_login"],
                password=env["mt5_password"],
                server=env["mt5_server"],
                path=path
            ) if path else mt5.initialize(
                login=env["mt5_login"],
                password=env["mt5_password"],
                server=env["mt5_server"]
            )
            
            if init_result:
                # Test connection by getting account info
                akun = mt5.account_info()
                if akun:
                    print(f"\n✅ Terhubung ke MT5")
                    print(f"   Akun: {akun.login}")
                    print(f"   Server: {akun.server}")
                    print(f"   Saldo: ${akun.balance:.2f} {akun.currency}")
                    print(f"   Leverage: 1:{akun.leverage}")
                    
                    if path:
                        print(f"   Path: {path}")
                    
                    mt5_terhubung = True
                    connection_success = True
                    
                    # Check AutoTrading status
                    terminal_info = mt5.terminal_info()
                    if terminal_info:
                        if not terminal_info.trade_allowed:
                            print(f"\n⚠️ WARNING: AutoTrading is DISABLED!")
                            print(f"   Enable it: Ctrl+E or click 'AutoTrading' button")
                        else:
                            print(f"✅ AutoTrading: ENABLED")
                    
                    # Get and subscribe to available symbols
                    print(f"\n🔍 Searching for available symbols...")
                    all_symbols = mt5.symbols_get()
                    
                    if all_symbols:
                        common_prefixes = ["XAUUSD", "GOLD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "BTCUSD"]
                        found_symbols = []
                        
                        print(f"   Total symbols available: {len(all_symbols)}")
                        
                        # Find and subscribe to common symbols
                        for prefix in common_prefixes:
                            best_match = None
                            
                            for sym in all_symbols:
                                sym_upper = sym.name.upper()
                                
                                # Exact match or starts with prefix
                                if sym_upper == prefix or sym_upper.startswith(prefix):
                                    # Prefer exact match
                                    if sym_upper == prefix:
                                        best_match = sym.name
                                        break
                                    # Otherwise take first match
                                    elif not best_match:
                                        best_match = sym.name
                            
                            # Subscribe to best match
                            if best_match:
                                if mt5.symbol_select(best_match, True):
                                    # Verify symbol has data
                                    tick = mt5.symbol_info_tick(best_match)
                                    if tick:
                                        found_symbols.append(best_match)
                                        print(f"   ✅ {best_match} - Bid: {tick.bid}")
                        
                        if found_symbols:
                            print(f"\n📊 Subscribed to {len(found_symbols)} symbols:")
                            print(f"   {', '.join(found_symbols)}")
                        else:
                            print(f"\n⚠️ No common symbols auto-detected")
                            print(f"   You can manually select symbols later")
                    
                    # Initialize symbol detector
                    print(f"\n🔧 Initializing symbol detector...")
                    init_symbol_detector()
                    
                    return True
                
        except Exception as e:
            error_msg = str(e)
            if "path" not in error_msg.lower():  # Don't spam path errors
                print(f"⚠️ Connection attempt failed: {error_msg}")
            continue
    
    # If we get here, all attempts failed
    if not connection_success:
        print(f"\n❌ Gagal terhubung ke MT5")
        
        last_error = mt5.last_error()
        if last_error:
            print(f"   Error code: {last_error[0]}")
            print(f"   Error message: {last_error[1]}")
        
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check MT5 is installed and running")
        print(f"   2. Verify credentials in .env file:")
        print(f"      MT5_LOGIN={env.get('mt5_login', 'NOT SET')}")
        print(f"      MT5_PASSWORD={'*' * len(env.get('mt5_password', '')) if env.get('mt5_password') else 'NOT SET'}")
        print(f"      MT5_SERVER={env.get('mt5_server', 'NOT SET')}")
        print(f"   3. Try logging in manually to MT5 first")
        print(f"   4. Check if account is active/not blocked")
        
        mt5_terhubung = False
        return False


def map_timeframe(timeframe_str: str) -> int:
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    return mapping.get(timeframe_str, mt5.TIMEFRAME_M30)


def init_symbol_detector():
    """Initialize symbol detector"""
    global symbol_detector
    
    if not mt5_terhubung:
        print("⚠️ MT5 not connected, skipping symbol detector")
        return None
    
    try:
        from utils.symbol_detector import SymbolDetector
        symbol_detector = SymbolDetector()
        
        # Quick test
        test_result = symbol_detector.find_symbol("XAUUSD")
        if test_result:
            print(f"✅ Symbol detector ready (test: XAUUSD → {test_result})")
        else:
            print(f"✅ Symbol detector ready")
        
        return symbol_detector
        
    except ImportError:
        print("⚠️ Symbol detector module not found")
        print("   Creating utils/symbol_detector.py...")
        
        # Create the file if it doesn't exist
        import os
        os.makedirs('utils', exist_ok=True)
        
        # Write the symbol detector code
        with open('utils/symbol_detector.py', 'w', encoding='utf-8') as f:
            f.write('''# utils/symbol_detector.py
import metatrader5 as mt5
from typing import Dict, List, Optional

class SymbolDetector:
    def __init__(self):
        self.symbol_cache = {}
        
    def find_symbol(self, base_symbol: str) -> Optional[str]:
        """Find the actual symbol name available in broker"""
        
        if base_symbol in self.symbol_cache:
            return self.symbol_cache[base_symbol]
        
        all_symbols = mt5.symbols_get()
        if not all_symbols:
            return None
        
        variations = self._generate_variations(base_symbol)
        
        # Try exact match first
        for symbol_info in all_symbols:
            name = symbol_info.name
            name_upper = name.upper()
            
            if name_upper == base_symbol.upper():
                if self._test_symbol(name):
                    self.symbol_cache[base_symbol] = name
                    return name
            
            for variant in variations:
                if name_upper == variant.upper():
                    if self._test_symbol(name):
                        self.symbol_cache[base_symbol] = name
                        return name
        
        # Try contains
        for symbol_info in all_symbols:
            name = symbol_info.name
            name_upper = name.upper()
            
            if 'XAU' in base_symbol.upper() or 'GOLD' in base_symbol.upper():
                if ('XAU' in name_upper or 'GOLD' in name_upper) and 'USD' in name_upper:
                    if self._test_symbol(name):
                        self.symbol_cache[base_symbol] = name
                        return name
        
        return None
    
    def _generate_variations(self, base: str) -> List[str]:
        base_upper = base.upper().replace('M', '').replace('-M', '')
        
        return [
            base, base_upper,
            f"{base_upper}m", f"{base_upper}M",
            f"{base_upper}-m", f"{base_upper}.a",
            f"{base_upper}_m"
        ]
    
    def _test_symbol(self, symbol: str) -> bool:
        try:
            if not mt5.symbol_select(symbol, True):
                return False
            tick = mt5.symbol_info_tick(symbol)
            return tick is not None
        except:
            return False
    
    def get_available_gold_symbols(self) -> List[str]:
        all_symbols = mt5.symbols_get()
        if not all_symbols:
            return []
        
        gold_symbols = []
        for symbol_info in all_symbols:
            name = symbol_info.name
            name_upper = name.upper()
            
            if ('XAU' in name_upper or 'GOLD' in name_upper) and 'USD' in name_upper:
                if self._test_symbol(name):
                    gold_symbols.append(name)
        
        return gold_symbols
''')
        
        print("✅ Created utils/symbol_detector.py")
        
        # Try again
        try:
            from utils.symbol_detector import SymbolDetector
            symbol_detector = SymbolDetector()
            print(f"✅ Symbol detector initialized")
            return symbol_detector
        except Exception as e2:
            print(f"❌ Still failed: {e2}")
            return None
            
    except Exception as e:
        print(f"❌ Symbol detector init failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def select_symbol(symbol_baru: str) -> bool:
    """Select a symbol in MT5 with fallback variations"""
    # Get available symbols
    all_symbols = mt5.symbols_get()
    available_symbols = []
    if all_symbols:
        for symbol_info in all_symbols:
            available_symbols.append(symbol_info.name)
    
    # Get uppercase input but preserve original for display
    symbol_upper = symbol_baru.upper()
    
    # Try exact match first (case insensitive)
    for available in available_symbols:
        if available.upper() == symbol_upper:
            if mt5.symbol_select(available, True):
                print(f"✅ Symbol {available} selected")
                return True
    
    # Try variations (case insensitive)
    variations = [
        symbol_upper,
        f"{symbol_upper}m",    # Some brokers use 'm' suffix for micro
        f"{symbol_upper}-m",   # Dash variation
        f"{symbol_upper}.a",   # Some use .a suffix
        f"{symbol_upper}-a",   # Dash variation
        f"{symbol_upper}_m",   # Underscore variation
        symbol_upper.replace('XAU', 'GOLD'),  # Special case for gold
        symbol_upper.replace('GOLD', 'XAU')   # Both directions
    ]
    
    # Try each variation with case-insensitive matching
    for variant in variations:
        for available in available_symbols:
            if available.upper() == variant:
                if mt5.symbol_select(available, True):
                    print(f"✅ Found matching symbol: {available}")
                    return True
    
    # If no exact match, look for similar symbols
    print(f"❌ Symbol {symbol_baru} not found")
    
    # First show exact matches ignoring case
    exact_matches = [s for s in available_symbols if s.upper() == symbol_upper]
    if exact_matches:
        print("\nExact matches (different case):")
        for sym in exact_matches:
            print(f"- {sym}")
    
    # Then show similar symbols
    print("\nSimilar symbols:")
    similar_symbols = [s for s in available_symbols if any(
        var in s.upper() for var in variations
    ) and s not in exact_matches]
    
    if similar_symbols:
        # Show gold-related symbols first if searching for gold
        if 'GOLD' in symbol_upper or 'XAU' in symbol_upper:
            gold_symbols = [s for s in similar_symbols if 'GOLD' in s.upper() or 'XAU' in s.upper()]
            if gold_symbols:
                for sym in sorted(gold_symbols)[:5]:
                    print(f"- {sym}")
    
        # Then show other similar symbols
        other_symbols = [s for s in similar_symbols if s not in gold_symbols] if 'gold_symbols' in locals() else similar_symbols
        if other_symbols:
            print("\nOther similar symbols:")
            for sym in sorted(other_symbols)[:5]:
                print(f"- {sym}")
    else:
        print("No similar symbols found")
    
    # Get most common forex pairs for suggestions
    common_pairs = [s for s in available_symbols if any(
        pair in s for pair in ['EUR', 'USD', 'GBP', 'JPY', 'XAU', 'GOLD']
    )]
    
    if common_pairs:
        print("\nPopular trading symbols:")
        for sym in sorted(common_pairs)[:10]:
            print(f"- {sym}")
    
    return False

def ambil_candle(symbol: str, timeframe: str, jumlah: int) -> pd.DataFrame:
    if not mt5_terhubung:
        print("❌ MT5 belum terhubung!")
        return pd.DataFrame()
    
    # Ensure symbol is selected with enhanced selection
    if not select_symbol(symbol):
        return pd.DataFrame()

    # Try different methods to get data
    mt5_tf = map_timeframe(timeframe)
    
    # Method 1: copy_rates_from_pos
    data = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, jumlah)
    
    if data is None:
        # Method 2: copy_rates_range
        from_date = pd.Timestamp.now() - pd.Timedelta(days=5)
        to_date = pd.Timestamp.now()
        
        data = mt5.copy_rates_range(
            symbol,
            mt5_tf,
            from_date.timetuple(),
            to_date.timetuple()
        )
    
    if data is None:
        print(f"❌ Gagal mengambil candle | Error: {mt5.last_error()}")
        # Reinitialize MT5 connection
        if init_mt5():
            print("✅ Berhasil reconnect MT5")
            # Try one more time
            data = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, jumlah)
    
    if data is None:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

def fetch_latest_news(api_key, symbol):
    """Mengambil berita terbaru yang relevan dengan simbol (misal: Gold/XAUUSD)"""
    if not api_key:
        return "No News API Key available."
        
    # Contoh URL untuk mencari berita tentang 'Gold' atau 'USD'
    url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&apiKey={api_key}&pageSize=5"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        if data['status'] == 'ok' and data['articles']:
            # Gabungkan judul dan deskripsi beberapa artikel menjadi satu string untuk dianalisis Gemini
            combined_text = "\n".join([f"- {a['title']}: {a['description']}" for a in data['articles']])
            return combined_text
        else:
            return f"No relevant news found for {symbol} today."
            
    except requests.RequestException as e:
        return f"News API Request Failed: {e}"
    
def analyze_with_deepseek_advanced(deepseek_config: Dict, analysis: dict, df: pd.DataFrame, symbol: str) -> dict:
    """
    Advanced DeepSeek AI analysis with structured output
    """
    if not deepseek_config:
        return {"recommendation": "WAIT", "confidence": 0, "reason": "DeepSeek not available"}

    # Prepare comprehensive data for AI
    last_close = df['close'].iloc[-1]
    price_change_5 = ((df['close'].iloc[-1] / df['close'].iloc[-5]) - 1) * 100 if len(df) >= 5 else 0
    price_change_20 = ((df['close'].iloc[-1] / df['close'].iloc[-20]) - 1) * 100 if len(df) >= 20 else 0
    
    # Get technical summary
    tech = analysis['technical']
    news = analysis['news']
    calendar = analysis['calendar']
    
    high = df['high'].tail(10).max()
    low = df['low'].tail(10).min()
    
    # Construct detailed prompt
    system_prompt = """Anda adalah AI analis trading Forex/Crypto yang ahli. 
    Analisis data market yang diberikan dan berikan rekomendasi trading dalam format JSON.

    Anda HARUS merespons dalam format JSON ini:
    {
      "recommendation": "BUY" atau "SELL" atau "WAIT",
      "confidence": 0-100,
      "entry_price": harga entry yang disarankan,
      "stop_loss": harga SL yang disarankan,
      "take_profit": harga TP yang disarankan,
      "risk_reward_ratio": rasio R:R yang dihitung,
      "key_factors": [daftar 3-5 faktor kunci yang mempengaruhi keputusan],
      "warnings": [daftar risiko atau kekhawatiran],
      "timeframe": "short-term" atau "medium-term" atau "long-term"
    }

    Dasarkan analisis Anda pada:
    1. Indikator teknikal dan tren
    2. Sentimen berita
    3. Event kalender ekonomi
    4. Prinsip manajemen risiko
    5. Kondisi market saat ini
    """
    
    market_data = f"""
=== ANALISIS MARKET UNTUK {symbol} ===

DATA HARGA:
- Harga Saat Ini: {last_close:.5f}
- Perubahan 5-bar: {price_change_5:+.2f}%
- Perubahan 20-bar: {price_change_20:+.2f}%
- High 10-bar: {high:.5f}
- Low 10-bar: {low:.5f}

ANALISIS TEKNIKAL:
- Sinyal: {tech['signal']}
- Sinyal Bullish: {tech['bullish']}
- Sinyal Bearish: {tech['bearish']}
- Confidence: {tech.get('confidence', 0):.1%}
- Indikator Kunci:
{chr(10).join(['  • ' + s for s in tech['signals'][:5]])}

SENTIMEN BERITA:
- Impact: {news['impact']}
- Skor Sentimen: {news.get('sentiment_score', 0)}
- Headline Terbaru:
{chr(10).join(['  • ' + h[:80] for h in news.get('headlines', [])[:3]])}

KALENDER EKONOMI:
- Level Impact: {calendar['impact']}
- Event High Impact: {calendar.get('high_impact_count', 0)}
- Event Mendatang:
{chr(10).join(['  • ' + e for e in calendar.get('events', [])[:3]])}

ANALISIS KESELURUHAN:
- Sinyal Gabungan: {analysis['overall']['signal']}
- Kekuatan Sinyal: {analysis['overall']['strength']:.1%}
- Alasan Kunci:
{chr(10).join(['  • ' + r for r in analysis['overall']['reasons'][:5]])}

=== TUGAS ===
Berdasarkan analisis komprehensif ini, berikan rekomendasi trading ahli Anda.
Pertimbangkan manajemen risiko, volatilitas saat ini, dan semua faktor di atas.
"""

    try:
        headers = {
            "Authorization": f"Bearer {deepseek_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": market_data
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
            "stream": False,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            deepseek_config["base_url"],
            headers=headers,
            json=payload,
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            
            # Parse JSON response
            ai_analysis = json.loads(ai_response)
            
            # Validate response
            required_keys = ['recommendation', 'confidence', 'key_factors']
            if all(key in ai_analysis for key in required_keys):
                # Convert confidence to 0-1 scale if needed
                if ai_analysis['confidence'] > 1:
                    ai_analysis['confidence'] = ai_analysis['confidence'] / 100.0
                
                return ai_analysis
            else:
                print("⚠️ AI response missing required fields")
                return {
                    "recommendation": "WAIT", 
                    "confidence": 0, 
                    "reason": "Invalid AI response",
                    "key_factors": []
                }
        else:
            print(f"❌ DeepSeek API error: {response.status_code}")
            return {
                "recommendation": "WAIT", 
                "confidence": 0, 
                "reason": f"API error: {response.status_code}",
                "key_factors": []
            }
            
    except requests.exceptions.Timeout:
        print("❌ DeepSeek API request timeout")
        return {
            "recommendation": "WAIT", 
            "confidence": 0, 
            "reason": "API timeout",
            "key_factors": []
        }
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse AI JSON response: {e}")
        return {
            "recommendation": "WAIT", 
            "confidence": 0, 
            "reason": "JSON parse error",
            "key_factors": []
        }
    except Exception as e:
        print(f"❌ Error in DeepSeek analysis: {e}")
        return {
            "recommendation": "WAIT", 
            "confidence": 0, 
            "reason": f"Analysis error: {str(e)}",
            "key_factors": []
        }
            
    except json.JSONDecodeError as e:
        print(f"❌ AI JSON decode error: {e}")
        print(f"Raw response: {response.text[:200]}")
        return {"recommendation": "WAIT", "confidence": 0, "reason": "JSON decode failed"}
    except Exception as e:
        print(f"❌ Gemini AI Error: {e}")
        return {"recommendation": "WAIT", "confidence": 0, "reason": str(e)}
    
def prepare_market_analysis(df: pd.DataFrame, symbol: str, technical_signals: Dict) -> Dict:
    """
    Siapkan analisis market yang komprehensif untuk AI
    """
    if len(df) < 20:
        return {
            "technical": {"signal": "HOLD", "confidence": 0, "signals": ["Insufficient data"]},
            "news": {"impact": "LOW", "sentiment_score": 0},
            "calendar": {"impact": "LOW", "high_impact_count": 0},
            "overall": {"signal": "HOLD", "strength": 0, "reasons": ["Insufficient data"]}
        }
    
    current = df.iloc[-1]
    
    # Technical analysis
    technical_signals_list = []
    
    # Trend analysis
    if current['close'] > current.get('ma_21', current['close']):
        technical_signals_list.append("Uptrend: Price above MA21")
    else:
        technical_signals_list.append("Downtrend: Price below MA21")
    
    # RSI analysis
    rsi = current.get('rsi', 50)
    if rsi > 70:
        technical_signals_list.append("RSI Overbought")
    elif rsi < 30:
        technical_signals_list.append("RSI Oversold")
    else:
        technical_signals_list.append("RSI Neutral")
    
    # Stochastic analysis
    stoch_k = current.get('STOCHk_14_3_3', 50)
    if stoch_k > 80:
        technical_signals_list.append("Stochastic Overbought")
    elif stoch_k < 20:
        technical_signals_list.append("Stochastic Oversold")
    
    # Bollinger Bands analysis
    if current['close'] > current.get('BBU_20_2.0', current['close']):
        technical_signals_list.append("Price above Bollinger Upper Band")
    elif current['close'] < current.get('BBL_20_2.0', current['close']):
        technical_signals_list.append("Price below Bollinger Lower Band")
    
    # Candlestick patterns
    if current.get('engulfing', 0) > 0:
        technical_signals_list.append("Bullish Engulfing Pattern")
    elif current.get('engulfing', 0) < 0:
        technical_signals_list.append("Bearish Engulfing Pattern")
    
    if current.get('doji', 0) > 0:
        technical_signals_list.append("Doji Pattern - Potential Reversal")
    
    # Count bullish vs bearish signals
    bullish_count = sum(1 for signal in technical_signals_list 
                       if any(word in signal.lower() for word in ['bullish', 'uptrend', 'oversold', 'below', 'lower']))
    bearish_count = sum(1 for signal in technical_signals_list 
                       if any(word in signal.lower() for word in ['bearish', 'downtrend', 'overbought', 'above', 'upper']))
    
    return {
        "technical": {
            "signal": technical_signals.get('signal', 'HOLD'),
            "bullish": bullish_count,
            "bearish": bearish_count,
            "confidence": technical_signals.get('confidence', 0),
            "signals": technical_signals_list
        },
        "news": {
            "impact": "MEDIUM",  # Placeholder - bisa diintegrasikan dengan news API
            "sentiment_score": 0.5,
            "headlines": [
                "Market volatility within normal range",
                "No major economic news affecting pair",
                "Technical factors driving price action"
            ]
        },
        "calendar": {
            "impact": "LOW",  # Placeholder - bisa diintegrasikan dengan calendar API
            "high_impact_count": 0,
            "events": [
                "No high impact events scheduled",
                "Normal trading conditions",
                "Monitor for unexpected news"
            ]
        },
        "overall": {
            "signal": technical_signals.get('signal', 'HOLD'),
            "strength": technical_signals.get('confidence', 0),
            "reasons": technical_signals_list[:3]  # Top 3 reasons
        }
    }
# --------------------------
# 3. TAMPILAN & INPUT MENU
# --------------------------
def get_account_summary() -> dict:
    """Get account summary including today's P&L"""
    if not mt5_terhubung:
        return None
        
    account = mt5.account_info()
    if not account:
        return None
    
    # Get today's deals for P&L calculation
    from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    to_date = datetime.now()
    
    deals = mt5.history_deals_get(from_date, to_date)
    
    today_profit = 0
    today_loss = 0
    total_trades = 0
    
    if deals:
        for deal in deals:
            # Skip balance operations
            if deal.type in [0, 1]:  # Buy or Sell deals only
                if deal.profit != 0:  # Closing deals
                    total_trades += 1
                    if deal.profit > 0:
                        today_profit += deal.profit
                    else:
                        today_loss += abs(deal.profit)
    
    # Calculate percentages
    starting_balance = account.balance - (today_profit - today_loss)
    profit_percent = (today_profit / starting_balance * 100) if starting_balance > 0 else 0
    loss_percent = (today_loss / starting_balance * 100) if starting_balance > 0 else 0
    
    # Get open positions
    positions = mt5.positions_get()
    open_positions = len(positions) if positions else 0
    floating_pl = sum(pos.profit for pos in positions) if positions else 0
    
    return {
        'balance': account.balance,
        'equity': account.equity,
        'margin': account.margin,
        'free_margin': account.margin_free,
        'currency': account.currency,
        'leverage': account.leverage,
        'today_profit': today_profit,
        'today_loss': today_loss,
        'today_net': today_profit - today_loss,
        'profit_percent': profit_percent,
        'loss_percent': loss_percent,
        'total_trades': total_trades,
        'open_positions': open_positions,
        'floating_pl': floating_pl
    }
    
# Letakkan ini di bagian atas file main.py, di bawah bagian import
last_backtest_result = None

# Gantikan fungsi cetak_menu() yang lama dengan yang ini
def cetak_menu(config: dict, rate: float, currency_symbol: str) -> None:
    global last_backtest_result
    current = config["current"]
    options = config["options"]
    
    # Helper untuk format ON/OFF
    def status(value):
        return "ON" if value else "OFF"

    # Helper untuk format baris menu
    def format_line(num, text, current_val, hint=""):
        # Menghapus 'Toggle' dan 'Set' untuk deskripsi yang lebih pendek
        text = text.replace("Toggle ", "").replace("Set ", "")
        
        # Membuat string utama
        line = f"{num:>2}) {text:<25}"
        
        # Menambahkan nilai saat ini
        current_str = f"(current: {current_val})"
        line = f"{line}{current_str:<25}"
        
        # Menambahkan hint jika ada
        if hint:
            line += f" {hint}"
        print(line)

    # Helper untuk format nilai mata uang
    def format_currency(usd_value):
        converted_value = usd_value * rate
        # Format IDR tanpa desimal, USD dengan 2 desimal
        if currency_symbol == "Rp":
            return f"{currency_symbol}{converted_value:,.0f}"
        else:
            return f"{currency_symbol}{converted_value:,.2f}"

    print("\n" + "="*90)

    # --- HEADER: Tampilkan hasil backtest terakhir atau status akun ---
    if last_backtest_result:
        print(last_backtest_result)
        last_backtest_result = None  # Reset agar hanya tampil sekali
    else:
        summary = get_account_summary()
        if summary:
            pl_sign = "+" if summary['today_net'] >= 0 else "-"
            # Ini sudah otomatis mendeteksi dari fungsi get_account_summary
            currency_symbol_header = "Rp" if summary['currency'] == 'IDR' else "$"
            print(f"💰 Balance: {currency_symbol_header}{summary['balance']:.2f} | Equity: {currency_symbol_header}{summary['equity']:.2f} | Today's P/L: {pl_sign}{currency_symbol_header}{abs(summary['today_net']):.2f} | Open: {summary['open_positions']}")

    print("="*90)
    print(f"\n{author_text}: {fb_info}, {tele_info} ✨")
    print("\n Menu:")
   

    # --- MENU UTAMA ---
    print(" 1) Analyze now")
    format_line(2, "Change SYMBOL", current['symbol'], f"available: {', '.join(get_available_symbols()[:5])}...")
    format_line(3, "Change TIMEFRAME", current['timeframe'], f"options: {', '.join(options['timeframes'])}")
    format_line(4, "Change CANDLES", current['candles'], "e.g. 50 / 100 / 200")
    format_line(5, "Switch ACCOUNT", current['account'], f"options: {', '.join(options['accounts'])}")
    format_line(6, "Change TRADE MODE", current['trade_mode'], f"options: {', '.join(options['trade_modes'])}")
    print(" 7) Launch external TRAINER window (every 10s, bars=800)")
    format_line(8, "Toggle AUTO-TRADE", status(current['auto_trade']))
    format_line(9, "Set AUTO lot", current['lot'])
    format_line(10, "Set AUTO slippage (dev)", current['slippage'])
    format_line(11, "Toggle AUTO-CLOSE profit", status(current['auto_close_profit']))
    print(" 12) AUTO-CLOSE settings (mode, targets, etc.)")  
    format_line(13, "Toggle AUTO-ANALYZE", status(current['auto_analyze']))
    format_line(14, "Set AUTO-ANALYZE interval minutes", current['auto_analyze_interval'])
    format_line(15, "Toggle BEP", status(current['bep']))
    format_line(16, f"BEP min profit ({currency_symbol})", format_currency(current['bep_min_profit']))
    format_line(17, "BEP spread multiplier", current['bep_spread_multiplier'])
    format_line(18, "Toggle STEP TRAILING", status(current['stpp_trailing']))
    format_line(19, f"STEP lock init ({currency_symbol})", format_currency(current['step_lock_init']))
    format_line(20, f"STEP step ({currency_symbol})", format_currency(current['step_step']))

    # --- KELOMPOK MENU LAINNYA ---
    print("\n-- Price Trigger --")
    print("21) Set ONE-SHOT price trigger (symbol, side, price, lot, SL, TP, int-match)")
    print("22) Cancel price trigger")
    format_line(23, "Set ENTRY match decimals", current['entry_decimals'], "e.g. None/0/1/2")

    print("\n-- Backtest --")
    print("24) Backtest (custom range) -> CSV")
    print("25) Backtest 1 minggu (last 7d)")
    print("26) Backtest 2 minggu (last 14d)")
    print("27) Backtest 1 bulan (last 30d)")
    print("28) Backtest 2 bulan (last 60d)")


    print("\n-- General --")
    format_line(29, "Toggle TRADE ALWAYS ON", status(current['trade_always_on']))
    print("30) Change mode (SCALPING/AGGRESSIVE/MODERATE/CONSERVATIVE)")
    print("31) Multi-position setup (RAPID FIRE mode)")
    print("32) Stop Loss Protection (limits, drawdown, emergency)")
    print("33) Safety Dashboard (real-time monitoring)")

    
    print("\n-- Optimization --")
    print("34) 💰 AUTO BALANCE OPTIMIZER (recommended!)")  # TAMBAH
    print("35) 🧮 SAFETY CALCULATOR") 
    print("36) ⚡ QUICK APPLY calculator settings")
    print("37) 📊 PROFIT PLANNER (set daily target)")  # NEW
    print("38) 📈 DYNAMIC TRADE LIMIT (auto-increase)")  # NEW
    print("39) 🎯 SIGNAL STRENGTH adjuster")

    print("\n-- Analytics --")
    print("40) 📊 TRADE STATISTICS (win rate, profit factor)")
    print("41) 🌊 MARKET CONDITION (trending/ranging)")
    print("42) 🌍 SESSION TRADING (London/NY/Asian)")
    
    print("\n-- Symbol Setup --")
    print("43) 🔍 AUTO-DETECT symbols (find broker format)")
    print("44) 🧪 TEST symbol connection")
        
    print("\n-- Controls --")
    print("99) START TRADING")
    print(" 0) Quit")
    print(f"\n{author_text}: {fb_info}, {tele_info} ✨")
    print("-" * 90)


def pilih_menu() -> int:
    pilihan = input("Select: ").strip()
    
    # Handle empty input
    if not pilihan:
        print("❌ Silakan masukkan pilihan menu!")
        return -1
        
    try:
        nilai = int(pilihan)
        
        # Validate menu option range
        valid_options = list(range(0, 45)) + [99]  # 0-29 plus 99
        if nilai not in valid_options:
            print(f"❌ Pilihan {nilai} tidak tersedia dalam menu!")
            return -1
            
        return nilai
        
    except ValueError:
        # Handle non-numeric input more gracefully
        print(f"❌ '{pilihan}' bukan pilihan yang valid! Masukkan nomor menu (0-30 atau 99)")
        return -1


# --------------------------
# 4. FUNGSI MENU (SESUI AI FOTO)
# --------------------------
def menu_1_analyze_now(config: dict, deepseek_config: Dict) -> None:
    """Analyze now - AGGRESSIVE MODE"""
    print(f"\n🔍 Analyzing {config['current']['symbol']} ({config['current']['timeframe']})...")
    print(f"Mode: {config['current']['trade_mode']} | Threshold: {config['current']['signal_threshold']}")
    
    # 1. Get market data
    df = ambil_candle(
        symbol=config["current"]["symbol"],
        timeframe=config["current"]["timeframe"],
        jumlah=config["current"]["candles"]
    )
    if df.empty: 
        print("❌ Cannot get candle data")
        return
    
    # 2. Initialize analyzer
    analyzer = MarketAnalyzer(
        news_api_key=env.get('news_api_key'),
        te_key=env.get('trading_economics_key')
    )
    
    # 3. Full analysis with config
    analysis = analyzer.analyze_market(df, config['current']['symbol'], config)
    
    # 4. Display results
    print("\n" + "="*80)
    print(f"📊 MARKET ANALYSIS - {config['current']['trade_mode']} MODE")
    print("="*80)
    
    # Technical
    print("\n🔧 TECHNICAL:")
    tech = analysis['technical']
    print(f"Signal: {tech['signal']} | Bullish: {tech['bullish']} | Bearish: {tech['bearish']}")
    for sig in tech['signals'][:5]:
        print(f"  • {sig}")
    
    # Patterns
    if analysis.get('patterns', {}).get('count', 0) > 0:
        print("\n📊 CANDLESTICK PATTERNS:")
        for pattern in analysis['patterns']['patterns']:
            print(f"  🔥 {pattern}")
    
    # Breakouts
    if analysis.get('breakout', {}).get('count', 0) > 0:
        print("\n💥 BREAKOUTS:")
        for bo in analysis['breakout']['breakouts']:
            print(f"  🚀 {bo}")
    
    # Scalping
    if config['current'].get('enable_scalping', True):
        scalp = analysis.get('scalping', {})
        if scalp.get('score', 0) != 0:
            print(f"\n⚡ SCALPING: Score {scalp['score']}")
            for sig in scalp.get('signals', []):
                print(f"  • {sig}")
    
    # AI Analysis (Optional - fast version) - DEEPSEEK VERSION
    if deepseek_config and config['current']['trade_mode'] != 'SCALPING':
        print("\n🤖 DEEPSEEK QUICK ANALYSIS:")
        try:
            ai_response = deepseek_quick_analysis(
                deepseek_config=deepseek_config,
                symbol=config['current']['symbol'],
                technical_signal=tech['signal'],
                bullish_count=tech['bullish'],
                bearish_count=tech['bearish'],
                patterns=analysis.get('patterns', {}).get('patterns', []),
                current_price=df['close'].iloc[-1]
            )
            if ai_response:
                print(f"  {ai_response[:150]}")
        except Exception as e:
            print(f"  ❌ AI analysis failed: {e}")
    
    # Overall
    print("\n🎯 FINAL DECISION:")
    overall = analysis['overall']
    print(f"Signal: {overall['signal']} | Strength: {overall['strength']:.1%}")
    for reason in overall['reasons'][:5]:
        print(f"  {reason}")
    
    # Market status
    symbol_info = mt5.symbol_info(config['current']['symbol'])
    if symbol_info:
        print(f"\n💱 Market: Bid {symbol_info.bid:.5f} | Ask {symbol_info.ask:.5f}")
    
    print("="*80)

def menu_2_change_symbol(config: dict) -> dict:
    """Change trading symbol"""
    print(f"\nCurrent Symbol: {config['current']['symbol']}")
    
    # Show available symbols
    if symbol_detector:
        print("\n📊 Available Symbols:")
        
        # Gold
        gold = symbol_detector.get_available_gold_symbols()
        if gold:
            print(f"\n💰 Gold: {', '.join(gold[:3])}")
        
        # Forex
        forex = symbol_detector.get_available_forex_pairs()
        if forex:
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
            print(f"\n💱 Forex:")
            for pair in major_pairs:
                if pair in forex:
                    print(f"   {forex[pair]} ({pair})")
        
        # Crypto
        all_symbols = mt5.symbols_get()
        if all_symbols:
            crypto = [s.name for s in all_symbols if 'BTC' in s.name.upper()]
            if crypto:
                print(f"\n₿ Crypto: {', '.join(crypto[:3])}")
    
    new_symbol = input("\nEnter new symbol: ").strip().upper()
    
    if not new_symbol:
        print("❌ Cancelled")
        return config
    
    # Validate symbol exists
    symbol_info = mt5.symbol_info(new_symbol)
    if not symbol_info:
        print(f"❌ Symbol {new_symbol} not found!")
        
        # Try to find alternative
        if symbol_detector:
            alternative = symbol_detector.find_symbol(new_symbol)
            if alternative:
                print(f"💡 Found similar: {alternative}")
                if input(f"Use {alternative}? (y/n): ").lower() == 'y':
                    new_symbol = alternative
                else:
                    return config
            else:
                print("💡 Use Menu 43 to auto-detect available symbols")
                return config
        else:
            return config
    
    # Update both main symbol AND symbols_to_trade
    config['current']['symbol'] = new_symbol
    
    # If multi-symbol is disabled, update symbols_to_trade to match
    if not config['current'].get('enable_multi_symbol', False):
        config['current']['symbols_to_trade'] = [new_symbol]
    else:
        # If multi-symbol enabled, ask if user wants to replace or add
        current_symbols = config['current'].get('symbols_to_trade', [])
        
        print(f"\nCurrent trading symbols: {', '.join(current_symbols)}")
        print("\n1) Replace all with new symbol")
        print("2) Add to list")
        print("3) Replace main symbol only")
        
        choice = input("\nSelect (1-3): ").strip()
        
        if choice == '1':
            config['current']['symbols_to_trade'] = [new_symbol]
            print(f"✅ Now trading: {new_symbol} only")
        elif choice == '2':
            if new_symbol not in current_symbols:
                current_symbols.append(new_symbol)
                config['current']['symbols_to_trade'] = current_symbols
            print(f"✅ Now trading: {', '.join(current_symbols)}")
        else:  # 3 or default
            # Keep symbols_to_trade as is, just change main
            print(f"✅ Main symbol changed, but still trading: {', '.join(current_symbols)}")
    
    save_config(config)
    
    # Show confirmation
    print(f"\n✅ Symbol Configuration:")
    print(f"   Main symbol: {config['current']['symbol']}")
    print(f"   Trading symbols: {', '.join(config['current'].get('symbols_to_trade', [new_symbol]))}")
    print(f"   Multi-symbol: {'ON' if config['current'].get('enable_multi_symbol') else 'OFF'}")
    
    return config


def menu_3_change_timeframe(config: dict) -> dict:
    new_tf = input(f"\nMasukkan timeframe baru (Opsi: {', '.join(config['options']['timeframes'])}): ").strip().upper()
    if new_tf in config["options"]["timeframes"]:
        config["current"]["timeframe"] = new_tf
        save_config(config)
        print(f"✅ Timeframe diubah menjadi: {new_tf}")
    else:
        print(f"❌ Timeframe {new_tf} tidak tersedia!")
    return config


def menu_4_change_candles(config: dict) -> dict:
    try:
        new_count = int(input("\nMasukkan jumlah candle (contoh: 50, 100): "))
        if new_count > 0:
            config["current"]["candles"] = new_count
            save_config(config)
            print(f"✅ Jumlah candle diatur menjadi: {new_count}")
        else:
            print("❌ Jumlah candle harus positif!")
    except ValueError:
        print("❌ Input harus angka!")
    return config


def menu_5_switch_account(config: dict) -> dict:
    new_acc = input(f"\nMasukkan tipe akun (Opsi: {', '.join(config['options']['accounts'])}): ").strip().upper()
    if new_acc in config["options"]["accounts"]:
        config["current"]["account"] = new_acc
        save_config(config)
        print(f"✅ Akun diubah menjadi: {new_acc}")
    else:
        print(f"❌ Akun {new_acc} tidak tersedia!")
    return config


def menu_6_change_trade_mode(config: dict) -> dict:
    new_mode = input(f"\nMasukkan mode trading (Opsi: {', '.join(config['options']['trade_modes'])}): ").strip().upper()
    if new_mode in config["options"]["trade_modes"]:
        config["current"]["trade_mode"] = new_mode
        save_config(config)
        print(f"✅ Mode Trading diubah menjadi: {new_mode}")
    else:
        print(f"❌ Mode Trading {new_mode} tidak tersedia!")
    return config


def menu_7_launch_trainer(config: dict) -> None:
    """Launch training visualization window"""
    print("\n📚 Launching Trainer Mode...")
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import matplotlib.dates as mdates
        
        # Create figure and axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle(f'Trading Trainer - {config["current"]["symbol"]} ({config["current"]["timeframe"]})')
        
        def update_chart(frame):
            # Get latest data
            df = ambil_candle(
                symbol=config["current"]["symbol"],
                timeframe=config["current"]["timeframe"],
                jumlah=config["current"]["candles"]
            )
            
            if df.empty:
                return
            
            # Clear axes
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            # Plot 1: Price and Moving Averages
            ax1.plot(df['time'], df['close'], label='Close', color='blue', linewidth=1)
            
            # Add SMA
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            ax1.plot(df['time'], df['SMA_20'], label='SMA 20', color='orange', alpha=0.7)
            ax1.plot(df['time'], df['SMA_50'], label='SMA 50', color='red', alpha=0.7)
            
            ax1.set_title('Price Action & Moving Averages')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            # Plot 2: Volume
            colors = ['g' if df['close'].iloc[i] > df['open'].iloc[i] else 'r' 
                     for i in range(len(df))]
            ax2.bar(df['time'], df['tick_volume'], color=colors, alpha=0.5)
            ax2.set_title('Volume')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: RSI
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            ax3.plot(df['time'], df['RSI'], label='RSI', color='purple')
            ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
            ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
            ax3.fill_between(df['time'], 30, 70, alpha=0.1)
            ax3.set_title('RSI (14)')
            ax3.set_ylim(0, 100)
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
            
            # Add latest price annotation
            last_price = df['close'].iloc[-1]
            ax1.annotate(f'${last_price:.2f}', 
                        xy=(df['time'].iloc[-1], last_price),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            plt.tight_layout()
        
        # Animation
        ani = FuncAnimation(fig, update_chart, interval=10000, cache_frame_data=False)  # Update every 10 seconds
        
        plt.show()
        
    except ImportError:
        print("❌ Matplotlib tidak terinstall. Install dengan: pip install matplotlib")
        print("\n💡 Alternatif: Gunakan menu 1 (Analyze Now) untuk analisis manual")
    except Exception as e:
        print(f"❌ Error: {e}")


def menu_8_toggle_auto_trade(config: dict) -> dict:
    config["current"]["auto_trade"] = not config["current"]["auto_trade"]
    save_config(config)
    print(f"✅ Auto Trade diubah menjadi: {'ON' if config['current']['auto_trade'] else 'OFF'}")
    return config


def menu_9_set_auto_lot(config: dict) -> dict:
    try:
        new_lot = float(input("\nMasukkan ukuran lot (contoh: 0.01, 0.1): "))
        if 0 < new_lot <= 10:
            config["current"]["lot"] = new_lot
            save_config(config)
            print(f"✅ Lot diatur menjadi: {new_lot}")
        else:
            print("❌ Lot harus antara 0.01 dan 10!")
    except ValueError:
        print("❌ Input harus angka!")
    return config


def menu_10_set_auto_slippage(config: dict) -> dict:
    try:
        new_slip = int(input("\nMasukkan slippage (points): "))
        if new_slip >= 0:
            config["current"]["slippage"] = new_slip
            save_config(config)
            print(f"✅ Slippage diatur menjadi: {new_slip}")
        else:
            print("❌ Slippage tidak bisa negatif!")
    except ValueError:
        print("❌ Input harus angka!")
    return config


def menu_11_toggle_auto_close_profit(config: dict) -> dict:
    config["current"]["auto_close_profit"] = not config["current"]["auto_close_profit"]
    save_config(config)
    print(f"✅ Auto Close Profit diubah menjadi: {'ON' if config['current']['auto_close_profit'] else 'OFF'}")
    return config


def menu_12_set_auto_close_settings(config: dict, rate: float, currency_symbol: str) -> dict:
    """Advanced Auto Close Settings"""
    print("\n💰 AUTO CLOSE PROFIT SETTINGS")
    print("="*70)
    
    current = config['current']
    
    # Konversi nilai USD dari config ke mata uang akun
    per_trade_target_display = current.get('auto_close_target', 0.4) * rate
    total_target_display = current.get('auto_close_total_target', 5.0) * rate
    
    print(f"Current Settings:")
    print(f"  Auto Close: {'ON' if current.get('auto_close_profit') else 'OFF'}")
    print(f"  Mode: {current.get('auto_close_mode', 'PER_TRADE')}")
    # Tampilkan nilai yang sudah dikonversi
    print(f"  Per Trade Target: {currency_symbol}{per_trade_target_display:,.2f}")
    print(f"  Total Target: {currency_symbol}{total_target_display:,.2f}")
    print(f"  Close All on Target: {'YES' if current.get('close_all_on_target') else 'NO'}")
    
    print("\n📋 OPTIONS:")
    print("1) Set MODE - How to calculate profit target")
    print(f"2) Set PER TRADE target (e.g., {currency_symbol}{0.4 * rate:,.0f})")
    print(f"3) Set TOTAL target (e.g., {currency_symbol}{5.0 * rate:,.0f})")
    print("4) Toggle CLOSE ALL on target")
    print("5) Quick presets")
    print("0) ← Back")
    
    choice = input("\nSelect option (0-5): ").strip()
    
    if choice == '0':
        return config
    
    if choice == '1':
        # Set Mode
        print("\n🎯 AUTO CLOSE MODE:")
        print("1) PER_TRADE   - Close each trade when it hits target (e.g., $0.4)")
        print("2) TOTAL       - Close all trades when TOTAL profit hits target (e.g., $5)")
        print("3) BOTH        - Use both conditions (whichever comes first)")
        
        mode_choice = input("\nSelect mode (1-3): ").strip()
        
        modes = {
            '1': 'PER_TRADE',
            '2': 'TOTAL',
            '3': 'BOTH'
        }
        
        if mode_choice in modes:
            config['current']['auto_close_mode'] = modes[mode_choice]
            save_config(config)
            
            print(f"\n✅ Mode set to: {modes[mode_choice]}")
            
            if modes[mode_choice] == 'PER_TRADE':
                print("   Each trade will close individually at target profit")
            elif modes[mode_choice] == 'TOTAL':
                print("   All trades will close together when total profit reaches target")
            else:
                print("   Trades close on EITHER per-trade OR total target")
    
    elif choice == '2':
        # Set Per Trade Target
        try:
            # Minta input dalam mata uang akun
            target_display = float(input(f"\nEnter per-trade profit target ({currency_symbol}, e.g., {0.4 * rate:,.0f}): {currency_symbol}"))
            
            # Konversi BALIK ke USD untuk disimpan
            target_usd = target_display / rate
            
            if 0.01 <= target_usd <= 100:
                config['current']['auto_close_target'] = target_usd
                save_config(config)
                print(f"\n✅ Per-trade target set to: {currency_symbol}{target_display:,.0f} (disimpan sebagai ${target_usd:,.2f} USD)")
            else:
                print(f"❌ Target harus antara $0.01 dan $100 USD")
        except ValueError:
            print("❌ Invalid input")
    
    elif choice == '3':
        # Set Total Target
        try:
            # Minta input dalam mata uang akun
            target_display = float(input(f"\nEnter total profit target ({currency_symbol}, e.g., {5.0 * rate:,.0f}): {currency_symbol}"))
            
            # Konversi BALIK ke USD untuk disimpan
            target_usd = target_display / rate
            
            if 0.1 <= target_usd <= 1000:
                config['current']['auto_close_total_target'] = target_usd
                save_config(config)
                print(f"\n✅ Total target set to: {currency_symbol}{target_display:,.0f} (disimpan sebagai ${target_usd:,.2f} USD)")
            else:
                print(f"❌ Target harus antara $0.1 dan $1000 USD")
        except ValueError:
            print("❌ Invalid input")
    
    elif choice == '4':
        # Toggle Close All
        current_val = config['current'].get('close_all_on_target', False)
        config['current']['close_all_on_target'] = not current_val
        save_config(config)
        
        new_val = config['current']['close_all_on_target']
        print(f"\n✅ Close All on Target: {'ON' if new_val else 'OFF'}")
        
        if new_val:
            print("   When ANY position hits target, ALL positions will close")
        else:
            print("   Only the profitable position will close")
    
    elif choice == '5':
        # Quick Presets (Biarkan ini dalam USD, tapi konversi saat diterapkan)
        print("\n⚡ QUICK PRESETS (Nilai dalam USD):")
        print("1) 🐌 CONSERVATIVE - $1/trade, $10 total")
        print("2) 📈 MODERATE      - $0.5/trade, $5 total")
        print("3) ⚡ AGGRESSIVE    - $0.3/trade, $3 total")
        print("4) 🔥 SCALPING      - $0.2/trade, $2 total")
        
        preset = input("\nSelect preset (1-4): ").strip()
        
        presets = {
            '1': {'per_trade': 1.0, 'total': 10.0, 'mode': 'BOTH'},
            '2': {'per_trade': 0.5, 'total': 5.0, 'mode': 'BOTH'},
            '3': {'per_trade': 0.3, 'total': 3.0, 'mode': 'BOTH'},
            '4': {'per_trade': 0.2, 'total': 2.0, 'mode': 'PER_TRADE'}
        }
        
        if preset in presets:
            p = presets[preset]
            # Simpan nilai USD ke config
            config['current']['auto_close_target'] = p['per_trade']
            config['current']['auto_close_total_target'] = p['total']
            config['current']['auto_close_mode'] = p['mode']
            save_config(config)
            
            print(f"\n✅ Preset applied!")
            # Tampilkan nilai USD dan nilai konversinya
            print(f"   Per trade: ${p['per_trade']:.2f} USD ({currency_symbol}{p['per_trade'] * rate:,.0f})")
            print(f"   Total: ${p['total']:.2f} USD ({currency_symbol}{p['total'] * rate:,.0f})")
            print(f"   Mode: {p['mode']}")
    
    return config


def menu_13_toggle_auto_analyze(config: dict) -> dict:
    config["current"]["auto_analyze"] = not config["current"]["auto_analyze"]
    save_config(config)
    print(f"✅ Auto Analyze diubah menjadi: {'ON' if config['current']['auto_analyze'] else 'OFF'}")
    return config


def menu_14_set_auto_analyze_interval(config: dict) -> dict:
    try:
        new_int = int(input("\nMasukkan interval (menit): "))
        if new_int > 0:
            config["current"]["auto_analyze_interval"] = new_int
            save_config(config)
            print(f"✅ Interval Auto Analyze diatur menjadi: {new_int} menit")
        else:
            print("❌ Interval harus positif!")
    except ValueError:
        print("❌ Input harus angka!")
    return config


def menu_15_toggle_bep(config: dict) -> dict:
    config["current"]["bep"] = not config["current"]["bep"]
    save_config(config)
    print(f"✅ BEP diubah menjadi: {'ON' if config['current']['bep'] else 'OFF'}")
    return config


def menu_16_set_bep_min_profit(config: dict, rate: float, currency_symbol: str) -> dict:
    try:
        current_val_display = config["current"]["bep_min_profit"] * rate
        new_profit_display = float(input(f"\nMasukkan BEP min profit ({currency_symbol}) (current: {current_val_display:,.0f}): {currency_symbol}"))
        
        new_profit_usd = new_profit_display / rate
        
        if new_profit_usd > 0:
            config["current"]["bep_min_profit"] = new_profit_usd
            save_config(config)
            print(f"✅ BEP Min Profit diatur menjadi: {currency_symbol}{new_profit_display:,.0f} (disimpan sebagai ${new_profit_usd:,.2f} USD)")
        else:
            print("❌ Profit harus positif!")
    except ValueError:
        print("❌ Input harus angka!")
    return config

def menu_17_set_bep_spread_multiplier(config: dict) -> dict:
    try:
        new_multi = float(input("\nMasukkan BEP spread multiplier (contoh: 1.0): "))
        if new_multi > 0:
            config["current"]["bep_spread_multiplier"] = new_multi
            save_config(config)
            print(f"✅ BEP Spread Multiplier diatur menjadi: {new_multi}")
        else:
            print("❌ Multiplier harus positif!")
    except ValueError:
        print("❌ Input harus angka!")
    return config


def menu_18_toggle_stpp_trailing(config: dict) -> dict:
    config["current"]["stpp_trailing"] = not config["current"]["stpp_trailing"]
    save_config(config)
    print(f"✅ STPP Trailing diubah menjadi: {'ON' if config['current']['stpp_trailing'] else 'OFF'}")
    return config


def menu_19_set_step_lock_init(config: dict, rate: float, currency_symbol: str) -> dict:
    try:
        current_val_display = config["current"]["step_lock_init"] * rate
        new_init_display = float(input(f"\nMasukkan STEP Lock Init ({currency_symbol}) (current: {current_val_display:,.0f}): {currency_symbol}"))
        
        new_init_usd = new_init_display / rate

        if new_init_usd > 0:
            config["current"]["step_lock_init"] = new_init_usd
            save_config(config)
            print(f"✅ STEP Lock Init diatur menjadi: {currency_symbol}{new_init_display:,.0f} (disimpan sebagai ${new_init_usd:,.2f} USD)")
        else:
            print("❌ Nilai harus positif!")
    except ValueError:
        print("❌ Input harus angka!")
    return config


def menu_20_set_step_step(config: dict, rate: float, currency_symbol: str) -> dict:
    try:
        current_val_display = config["current"]["step_step"] * rate
        new_step_display = float(input(f"\nMasukkan STEP Step ({currency_symbol}) (current: {current_val_display:,.0f}): {currency_symbol}"))

        new_step_usd = new_step_display / rate

        if new_step_usd > 0:
            config["current"]["step_step"] = new_step_usd
            save_config(config)
            print(f"✅ STEP Step diatur menjadi: {currency_symbol}{new_step_display:,.0f} (disimpan sebagai ${new_step_usd:,.2f} USD)")
        else:
            print("❌ Nilai harus positif!")
    except ValueError:
        print("❌ Input harus angka!")
    return config

def menu_0_quit() -> bool:
    mt5.shutdown()
    print("\n👋 Menutup MT5 dan keluar...")
    return False


# Global variable for price triggers
price_triggers = []

def menu_21_set_one_shot(config: dict) -> None:
    """Set one-shot price trigger order"""
    print("\n🎯 SET PRICE TRIGGER ORDER")
    print("="*50)
    
    try:
        # Get inputs
        print("Enter trigger details:")
        symbol = input(f"Symbol [{config['current']['symbol']}]: ").strip() or config['current']['symbol']
        
        # Validate symbol
        if not select_symbol(symbol):
            print("❌ Invalid symbol")
            return
            
        side = input("Side (BUY/SELL): ").strip().upper()
        if side not in ['BUY', 'SELL']:
            print("❌ Side must be BUY or SELL")
            return
            
        trigger_price = float(input("Trigger Price: "))
        lot = float(input(f"Lot Size [{config['current']['lot']}]: ") or config['current']['lot'])
        
        # Get current price for reference
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            print(f"\n📊 Current Market:")
            print(f"Bid: {tick.bid:.5f} | Ask: {tick.ask:.5f}")
            
        # Optional SL/TP
        sl_input = input("Stop Loss (price or leave empty): ").strip()
        sl = float(sl_input) if sl_input else None
        
        tp_input = input("Take Profit (price or leave empty): ").strip()
        tp = float(tp_input) if tp_input else None
        
        # Create trigger
        trigger = {
            'id': len(price_triggers) + 1,
            'symbol': symbol,
            'side': side,
            'trigger_price': trigger_price,
            'lot': lot,
            'sl': sl,
            'tp': tp,
            'created_at': datetime.now(),
            'status': 'PENDING',
            'triggered': False
        }
        
        price_triggers.append(trigger)
        
        print("\n✅ Price Trigger Created:")
        print(f"ID: {trigger['id']}")
        print(f"Symbol: {symbol}")
        print(f"Side: {side}")
        print(f"Trigger: {trigger_price:.5f}")
        print(f"Lot: {lot}")
        if sl: print(f"SL: {sl:.5f}")
        if tp: print(f"TP: {tp:.5f}")
        print(f"Status: PENDING")
        
        # Start monitoring in background
        import threading
        monitor_thread = threading.Thread(target=create_trigger_from_analysis, daemon=True)
        monitor_thread.start()
        print("\n🔍 Price monitoring started...")
        
    except ValueError:
        print("❌ Invalid input format")
    except Exception as e:
        print(f"❌ Error: {e}")

def menu_22_cancel_price_trigger() -> None:
    """Cancel pending price triggers"""
    global price_triggers
    
    if not price_triggers:
        print("\n❌ No active price triggers")
        return
        
    print("\n📋 ACTIVE PRICE TRIGGERS")
    print("="*50)
    
    for trigger in price_triggers:
        if trigger['status'] == 'PENDING':
            print(f"ID: {trigger['id']} | {trigger['symbol']} | {trigger['side']} @ {trigger['trigger_price']:.5f}")
            print(f"   Lot: {trigger['lot']} | Created: {trigger['created_at'].strftime('%H:%M:%S')}")
            print("-"*50)
    
    try:
        trigger_id = input("\nEnter Trigger ID to cancel (or 'all' to cancel all): ").strip()
        
        if trigger_id.lower() == 'all':
            # Cancel all pending triggers
            cancelled = 0
            for trigger in price_triggers:
                if trigger['status'] == 'PENDING':
                    trigger['status'] = 'CANCELLED'
                    cancelled += 1
            print(f"✅ Cancelled {cancelled} triggers")
        else:
            # Cancel specific trigger
            trigger_id = int(trigger_id)
            for trigger in price_triggers:
                if trigger['id'] == trigger_id and trigger['status'] == 'PENDING':
                    trigger['status'] = 'CANCELLED'
                    print(f"✅ Trigger {trigger_id} cancelled")
                    return
            print(f"❌ Trigger {trigger_id} not found or already triggered")
            
    except ValueError:
        print("❌ Invalid input")

def create_trigger_from_analysis(symbol: str, analysis: dict, lot_size: float = 0.02) -> Optional[Dict]:
    """
    Buat price trigger berdasarkan analisis teknikal
    
    Args:
        symbol: Trading symbol
        analysis: Hasil analisis dari MarketAnalyzer
        lot_size: Ukuran lot untuk trade
    
    Returns:
        Trigger dictionary atau None jika tidak ada sinyal
    """
    try:
        # Ambil sinyal dari analisis
        signal = analysis.get('overall', {}).get('signal', 'HOLD')
        strength = analysis.get('overall', {}).get('strength', 0)
        
        # Hanya buat trigger untuk sinyal BUY/SELL yang kuat
        if signal == 'HOLD' or strength < 0.6:
            return None
        
        # Dapatkan data harga terbaru
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None
        
        current_bid = tick.bid
        current_ask = tick.ask
        
        # Tentukan parameter trigger berdasarkan analisis
        if signal == 'BUY':
            # Untuk BUY: trigger ketika harga turun ke level support
            trigger_price = current_bid * 0.998  # 0.2% di bawah current price
            stop_loss = current_bid * 0.995      # 0.5% di bawah trigger
            take_profit = current_bid * 1.010    # 1.0% di atas trigger
            side = 'BUY'
            
        elif signal == 'SELL':
            # Untuk SELL: trigger ketika harga naik ke level resistance  
            trigger_price = current_ask * 1.002  # 0.2% di atas current price
            stop_loss = current_ask * 1.005      # 0.5% di atas trigger
            take_profit = current_ask * 0.990    # 1.0% di bawah trigger
            side = 'SELL'
        
        else:
            return None
        
        # Buat trigger object
        trigger = {
            'id': len(price_triggers) + 1,
            'symbol': symbol,
            'side': side,
            'trigger_price': round(trigger_price, 5),
            'lot': lot_size,
            'sl': round(stop_loss, 5),
            'tp': round(take_profit, 5),
            'status': 'PENDING',
            'created_at': datetime.now(),
            'signal_strength': strength,
            'analysis_reasons': analysis.get('overall', {}).get('reasons', [])[:3],
            'technical_signals': analysis.get('technical', {}).get('signals', [])[:3]
        }
        
        return trigger
        
    except Exception as e:
        print(f"❌ Error creating trigger from analysis: {e}")
        return None


def menu_23_set_entry_decimals(config: dict) -> dict:
    try:
        new_dec = input("\nMasukkan entry decimals (contoh: None/0/1/2): ").strip().lower()
        if new_dec == "none":
            config["current"]["entry_decimals"] = None
        else:
            new_dec = int(new_dec)
            if 0 <= new_dec <= 2:
                config["current"]["entry_decimals"] = new_dec
            else:
                print("❌ Decimals harus 0,1,2, atau 'None'!")
                return config
        save_config(config)
        print(f"✅ Entry Decimals diatur menjadi: {config['current']['entry_decimals']}")
    except ValueError:
        print("❌ Input harus 'None', 0, 1, atau 2!")
    return config


def backtest_umum(config: dict, hari: int) -> None:
    print(f"\n🔄 Backtest {config['current']['symbol']} ({config['current']['timeframe']}) selama {hari} hari...")
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=hari)
    
    mt5_start = start_date.timetuple()
    mt5_end = end_date.timetuple()
    mt5_tf = map_timeframe(config["current"]["timeframe"])
    
    data = mt5.copy_rates_range(
        config["current"]["symbol"],
        mt5_tf,
        mt5_start,
        mt5_end
    )
    
    if data is None:
        print(f"❌ Backtest gagal | Error: {mt5.last_error()}")
        return
    
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    print(f"✅ Data backtest loaded | Baris: {len(df)} | Tanggal: {df['time'].min()} to {df['time'].max()}")
    print(df[["time", "open", "high", "low", "close"]].head(5))
    
    csv_name = f"backtest_{config['current']['symbol']}_{hari}d.csv"
    df.to_csv(csv_name, index=False)
    print(f"✅ Backtest disimpan ke: {csv_name}")


def menu_24_backtest_custom(config: dict) -> None:
    print("\n📅 Backtest Kustom: Masukkan tanggal (YYYY-MM-DD)")
    try:
        start_str = input("Tanggal Mulai: ")
        end_str = input("Tanggal Akhir: ")
        start_date = pd.Timestamp(start_str)
        end_date = pd.Timestamp(end_str)
        
        mt5_start = start_date.timetuple()
        mt5_end = end_date.timetuple()
        mt5_tf = map_timeframe(config["current"]["timeframe"])
        
        data = mt5.copy_rates_range(
            config["current"]["symbol"],
            mt5_tf,
            mt5_start,
            mt5_end
        )
        
        if data is None:
            print(f"❌ Backtest gagal | Error: {mt5.last_error()}")
            return
        
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        csv_name = f"backtest_{config['current']['symbol']}_kustom.csv"
        df.to_csv(csv_name, index=False)
        print(f"✅ Backtest kustom disimpan ke: {csv_name}")
    except Exception as e:
        print(f"❌ Format tanggal salah | Error: {str(e)}")


def menu_25_backtest_7d(config: dict) -> None:
    backtest_umum(config, hari=7)


def menu_26_backtest_14d(config: dict) -> None:
    backtest_umum(config, hari=14)


def menu_27_backtest_30d(config: dict) -> None:
    backtest_umum(config, hari=30)


def menu_28_backtest_60d(config: dict) -> None:
    backtest_umum(config, hari=60)


def menu_29_toggle_trade_always_on(config: dict) -> dict:
    config["current"]["trade_always_on"] = not config["current"]["trade_always_on"]
    save_config(config)
    print(f"✅ Trade Always On diubah menjadi: {'ON' if config['current']['trade_always_on'] else 'OFF'}")
    return config

def menu_30_change_mode_settings(config: dict) -> dict:
    """Advanced mode settings - SCALPING/AGGRESSIVE"""
    print("\n⚙️ TRADING MODE SETTINGS")
    print("="*60)
    print(f"Current Mode: {config['current']['trade_mode']}")
    print(f"Signal Threshold: {config['current'].get('min_signal_strength', 0.2):.0%}")
    print(f"Check Interval: {config['current']['auto_analyze_interval']} min")
    print("="*60)
    
    print("\n📊 PRESET MODES:")
    print("1) 🔥 SCALPING     - Ultra fast (5% threshold, 1min check, 100 trades/day)")
    print("2) ⚡ AGGRESSIVE   - Fast signals (10% threshold, 2min check, 50 trades/day)")
    print("3) 📈 MODERATE     - Balanced (20% threshold, 5min check, 20 trades/day)")
    print("4) 🛡️  CONSERVATIVE - Safe (35% threshold, 15min check, 10 trades/day)")
    print("5) ⚙️  CUSTOM       - Set your own values")
    print("0) ← Back")
    
    choice = input("\nSelect preset (0-5): ").strip()
    
    if choice == '0':
        return config
    
    presets = {
        '1': {
            'trade_mode': 'SCALPING',
            'min_signal_strength': 0.05,
            'auto_analyze_interval': 1,
            'max_daily_trades': 100,
            'enable_scalping': True,
            'enable_pattern_trading': True,
            'enable_breakout_trading': True,
            'signal_threshold': 'LOW'
        },
        '2': {
            'trade_mode': 'AGGRESSIVE',
            'min_signal_strength': 0.1,
            'auto_analyze_interval': 2,
            'max_daily_trades': 50,
            'enable_scalping': True,
            'enable_pattern_trading': True,
            'enable_breakout_trading': True,
            'signal_threshold': 'LOW'
        },
        '3': {
            'trade_mode': 'MODERATE',
            'min_signal_strength': 0.2,
            'auto_analyze_interval': 5,
            'max_daily_trades': 20,
            'enable_scalping': False,
            'enable_pattern_trading': True,
            'enable_breakout_trading': True,
            'signal_threshold': 'MEDIUM'
        },
        '4': {
            'trade_mode': 'CONSERVATIVE',
            'min_signal_strength': 0.35,
            'auto_analyze_interval': 15,
            'max_daily_trades': 10,
            'enable_scalping': False,
            'enable_pattern_trading': True,
            'enable_breakout_trading': False,
            'signal_threshold': 'HIGH'
        }
    }
    
    if choice in presets:
        # Apply preset
        for key, value in presets[choice].items():
            config['current'][key] = value
        
        save_config(config)
        
        print("\n✅ Settings Applied:")
        print(f"Mode: {config['current']['trade_mode']}")
        print(f"Threshold: {config['current']['min_signal_strength']:.0%}")
        print(f"Interval: {config['current']['auto_analyze_interval']} min")
        print(f"Max Trades: {config['current']['max_daily_trades']}/day")
        print(f"Scalping: {'✅' if config['current']['enable_scalping'] else '❌'}")
        print(f"Patterns: {'✅' if config['current']['enable_pattern_trading'] else '❌'}")
        print(f"Breakouts: {'✅' if config['current']['enable_breakout_trading'] else '❌'}")
        
    elif choice == '5':
        # Custom settings
        print("\n⚙️ CUSTOM SETTINGS:")
        try:
            threshold = float(input("Signal threshold (0.05-0.50, e.g., 0.1 for 10%): "))
            interval = int(input("Check interval in minutes (1-60): "))
            max_trades = int(input("Max daily trades (1-200): "))
            
            if 0.05 <= threshold <= 0.5 and 1 <= interval <= 60 and 1 <= max_trades <= 200:
                config['current']['min_signal_strength'] = threshold
                config['current']['auto_analyze_interval'] = interval
                config['current']['max_daily_trades'] = max_trades
                
                # Ask for features
                scalp = input("Enable scalping signals? (y/n): ").lower() == 'y'
                patterns = input("Enable pattern trading? (y/n): ").lower() == 'y'
                breakouts = input("Enable breakout trading? (y/n): ").lower() == 'y'
                
                config['current']['enable_scalping'] = scalp
                config['current']['enable_pattern_trading'] = patterns
                config['current']['enable_breakout_trading'] = breakouts
                config['current']['trade_mode'] = 'CUSTOM'
                
                save_config(config)
                print("\n✅ Custom settings saved!")
            else:
                print("❌ Invalid range!")
        except ValueError:
            print("❌ Invalid input!")
    else:
        print("❌ Invalid choice!")
    
    return config

def menu_31_setup_multi_position(config: dict) -> dict:
    """Setup multi-position trading"""
    print("\n⚙️ MULTI-POSITION SETUP")
    print("="*60)
    
    current = config['current']
    
    print(f"Current Settings:")
    print(f"  Max positions per symbol: {current.get('max_positions_per_symbol', 1)}")
    print(f"  Max total positions: {current.get('max_total_positions', 5)}")
    print(f"  Multi-symbol: {current.get('enable_multi_symbol', False)}")
    print(f"  Rapid fire mode: {current.get('rapid_fire_mode', False)}")
    
    print("\n📋 PRESETS:")
    print("1) 🐌 CONSERVATIVE - 1 position per symbol, max 3 total")
    print("2) 📈 MODERATE      - 3 positions per symbol, max 10 total")
    print("3) ⚡ AGGRESSIVE    - 5 positions per symbol, max 20 total")
    print("4) 🔥 RAPID FIRE    - 10 positions per symbol, max 50 total, multi-symbol")
    print("5) ⚙️  CUSTOM        - Set your own")
    print("0) ← Back")
    
    choice = input("\nSelect preset (0-5): ").strip()
    
    presets = {
        '1': {
            'max_positions_per_symbol': 1,
            'max_total_positions': 3,
            'enable_multi_symbol': False,
            'rapid_fire_mode': False,
            'max_daily_trades': 20
        },
        '2': {
            'max_positions_per_symbol': 3,
            'max_total_positions': 10,
            'enable_multi_symbol': False,
            'rapid_fire_mode': False,
            'max_daily_trades': 50
        },
        '3': {
            'max_positions_per_symbol': 5,
            'max_total_positions': 20,
            'enable_multi_symbol': True,
            'rapid_fire_mode': False,
            'max_daily_trades': 100
        },
        '4': {
            'max_positions_per_symbol': 10,
            'max_total_positions': 50,
            'enable_multi_symbol': True,
            'rapid_fire_mode': True,
            'enable_multi_timeframe': True,
            'max_daily_trades': 200,
            'auto_analyze_interval': 1,
            'symbols_to_trade': ['XAUUSDm', 'EURUSDm', 'GBPUSDm'],
            'timeframes_to_check': ['M1', 'M5', 'M15']
        }
    }
    
    if choice == '0':
        return config
    
    if choice in presets:
        for key, value in presets[choice].items():
            config['current'][key] = value
        
        save_config(config)
        
        print("\n✅ Settings Applied!")
        print(f"  Max positions per symbol: {config['current']['max_positions_per_symbol']}")
        print(f"  Max total positions: {config['current']['max_total_positions']}")
        print(f"  Max daily trades: {config['current']['max_daily_trades']}")
        print(f"  Rapid fire: {'✅' if config['current'].get('rapid_fire_mode') else '❌'}")
        
    elif choice == '5':
        try:
            per_symbol = int(input("Max positions per symbol (1-20): "))
            total = int(input("Max total positions (1-100): "))
            multi = input("Enable multi-symbol? (y/n): ").lower() == 'y'
            rapid = input("Enable rapid fire mode? (y/n): ").lower() == 'y'
            
            config['current']['max_positions_per_symbol'] = per_symbol
            config['current']['max_total_positions'] = total
            config['current']['enable_multi_symbol'] = multi
            config['current']['rapid_fire_mode'] = rapid
            
            save_config(config)
            print("\n✅ Custom settings saved!")
        except:
            print("❌ Invalid input")
    
    return config

def menu_32_stop_loss_settings(config: dict, rate: float, currency_symbol: str) -> dict:
    """Complete Stop Loss Protection Settings - DYNAMIC PRESETS"""
    print("\n🛡️ STOP LOSS PROTECTION SETTINGS")
    print("="*70)
    
    current = config['current']
    
    # 1. Dapatkan info saldo
    account = mt5.account_info()
    balance_native = 0.0
    balance_usd = 0.0
    
    if account:
        balance_native = account.balance # Saldo dalam mata uang akun (cth: 322460.95 IDR)
        # Konversi ke USD untuk perhitungan persentase
        balance_usd = balance_native / rate 
    
    # Ambil nilai USD dari config
    usd_max_loss_trade = current.get('max_loss_per_trade', 1.0)
    usd_global_max_loss = current.get('global_max_loss', 5.0)
    usd_daily_loss_limit = current.get('daily_loss_limit', 10.0)
    
    # Konversi nilai USD dari config ke mata uang akun untuk ditampilkan
    max_loss_trade_display = usd_max_loss_trade * rate
    global_max_loss_display = usd_global_max_loss * rate
    daily_loss_limit_display = usd_daily_loss_limit * rate
    
    print(f"Current Protection Levels:")
    print(f"  Protection: {'ON' if current.get('stop_loss_protection') else 'OFF'}")
    print(f"  Max Loss per Trade: {currency_symbol}{max_loss_trade_display:,.0f} (${usd_max_loss_trade:,.2f} USD)")
    print(f"  Global Max Loss: {currency_symbol}{global_max_loss_display:,.0f} (${usd_global_max_loss:,.2f} USD)")
    print(f"  Daily Loss Limit: {currency_symbol}{daily_loss_limit_display:,.0f} (${usd_daily_loss_limit:,.2f} USD)")
    print(f"  Max Drawdown: {current.get('max_drawdown_percent', 20.0):.1f}%")
    print(f"  Auto Close on Loss: {'YES' if current.get('auto_close_on_loss') else 'NO'}")
    print(f"  SL Multiplier: {current.get('sl_multiplier', 1.5):.1f}x ATR")
    
    # 2. Perbaiki Tampilan "Based on Balance"
    if account and balance_usd > 0:
        print(f"\n💰 Based on Balance: {currency_symbol}{balance_native:,.0f} (${balance_usd:,.2f} USD)")
        # Hitung persentase berdasarkan balance_usd
        print(f"  Max loss/trade: ${usd_max_loss_trade:.2f} ({usd_max_loss_trade/balance_usd*100:.1f}%)")
        print(f"  Global max loss: ${usd_global_max_loss:.2f} ({usd_global_max_loss/balance_usd*100:.1f}%)")
        print(f"  Daily limit: ${usd_daily_loss_limit:.2f} ({usd_daily_loss_limit/balance_usd*100:.1f}%)")
    
    print("\n📋 OPTIONS:")
    print("1) Toggle STOP LOSS PROTECTION (enable/disable all)")
    print(f"2) Set MAX LOSS per trade (e.g., {currency_symbol}{1.0 * rate:,.0f})")
    print(f"3) Set GLOBAL MAX LOSS (total floating loss, e.g., {currency_symbol}{5.0 * rate:,.0f})")
    print(f"4) Set DAILY LOSS LIMIT (max loss per day, e.g., {currency_symbol}{10.0 * rate:,.0f})")
    print("5) Set MAX DRAWDOWN % (e.g., 20% of balance)")
    print("6) Toggle AUTO CLOSE on loss")
    print("7) Set SL MULTIPLIER (ATR multiplier, e.g., 1.5)")
    print("8) ⚡ Quick DYNAMIC PRESETS (Based on % Balance)")
    print("9) EMERGENCY STOP (stop all trading NOW)")
    print("0) ← Back")
    
    choice = input("\nSelect option (0-9): ").strip()
    
    if choice == '0':
        return config
    
    if choice == '1':
        current_val = config['current'].get('stop_loss_protection', False)
        config['current']['stop_loss_protection'] = not current_val
        save_config(config)
        new_val = config['current']['stop_loss_protection']
        print(f"\n✅ Stop Loss Protection: {'ON' if new_val else 'OFF'}")

    elif choice == '2':
        # Max Loss per Trade
        try:
            loss_display = float(input(f"\nEnter max loss per trade ({currency_symbol}, e.g., {1.0 * rate:,.0f}): {currency_symbol}"))
            loss_usd = loss_display / rate
            
            if 0.1 <= loss_usd <= 1000: # Naikkan batas atas
                config['current']['max_loss_per_trade'] = round(loss_usd, 2)
                save_config(config)
                print(f"\n✅ Max loss per trade: {currency_symbol}{loss_display:,.0f} (disimpan sebagai ${loss_usd:,.2f} USD)")
            else:
                print("❌ Loss harus antara $0.1 dan $1000 USD")
        except ValueError:
            print("❌ Invalid input")
    
    elif choice == '3':
        # Global Max Loss
        try:
            loss_display = float(input(f"\nEnter global max loss (total floating, e.g., {5.0 * rate:,.0f}): {currency_symbol}"))
            loss_usd = loss_display / rate
            
            if 1.0 <= loss_usd <= 10000: # Naikkan batas atas
                config['current']['global_max_loss'] = round(loss_usd, 2)
                save_config(config)
                print(f"\n✅ Global max loss: {currency_symbol}{loss_display:,.0f} (disimpan sebagai ${loss_usd:,.2f} USD)")
            else:
                print("❌ Loss harus antara $1 dan $10000 USD")
        except ValueError:
            print("❌ Invalid input")
    
    elif choice == '4':
        # Daily Loss Limit
        try:
            loss_display = float(input(f"\nEnter daily loss limit (e.g., {10.0 * rate:,.0f}): {currency_symbol}"))
            loss_usd = loss_display / rate
            
            if 1.0 <= loss_usd <= 10000: # Naikkan batas atas
                config['current']['daily_loss_limit'] = round(loss_usd, 2)
                save_config(config)
                print(f"\n✅ Daily loss limit: {currency_symbol}{loss_display:,.0f} (disimpan sebagai ${loss_usd:,.2f} USD)")
            else:
                print("❌ Loss harus antara $1 dan $10000 USD")
        except ValueError:
            print("❌ Invalid input")

    elif choice == '5':
        # Max Drawdown %
        try:
            dd = float(input("\nEnter max drawdown % (e.g., 20 for 20%): "))
            
            if 5.0 <= dd <= 50.0:
                config['current']['max_drawdown_percent'] = dd
                save_config(config)
                
                account = mt5.account_info()
                if account:
                    dd_usd = (account.balance / rate) * (dd / 100)
                    dd_native = account.balance * (dd / 100)
                    print(f"\n✅ Max drawdown: {dd:.1f}% ({currency_symbol}{dd_native:,.0f} / ${dd_usd:,.2f})")
                    print("   Trading will stop if equity drops this much")
                else:
                    print(f"\n✅ Max drawdown: {dd:.1f}%")
            else:
                print("❌ Drawdown must be between 5% and 50%")
        except ValueError:
            print("❌ Invalid input")

    elif choice == '6':
        # Toggle Auto Close on Loss
        current_val = config['current'].get('auto_close_on_loss', False)
        config['current']['auto_close_on_loss'] = not current_val
        save_config(config)
        new_val = config['current']['auto_close_on_loss']
        print(f"\n✅ Auto Close on Loss: {'ON' if new_val else 'OFF'}")

    elif choice == '7':
        # SL Multiplier
        try:
            mult = float(input("\nEnter SL multiplier (ATR multiplier, e.g., 1.5): "))
            
            if 0.5 <= mult <= 5.0:
                config['current']['sl_multiplier'] = mult
                save_config(config)
                print(f"\n✅ SL Multiplier: {mult:.1f}x ATR")
                print(f"   Tighter SL = less loss but more stop-outs")
                print(f"   Wider SL = more breathing room but bigger loss")
            else:
                print("❌ Multiplier must be between 0.5 and 5.0")
        except ValueError:
            print("❌ Invalid input")
    
    elif choice == '8':
        # --- INI DIA PERUBAHAN BESARNYA ---
        if balance_usd == 0:
            print("❌ Tidak bisa menghitung preset, balance $0 USD.")
            return config
            
        print(f"\n⚡ DYNAMIC SAFETY PRESETS (berdasarkan Saldo ${balance_usd:,.2f} USD)")
        
        # 1. Definisikan presets sebagai PERSENTASE (decimals)
        presets = {
            '1': {'name': '♟️ MICRO (1% risk/trade)',    'risk_trade': 0.01, 'global': 0.03, 'daily': 0.05, 'dd': 30.0, 'sl_mult': 1.0},
            '2': {'name': '🛡️ CONSERVATIVE (1.5% risk)', 'risk_trade': 0.015,'global': 0.05, 'daily': 0.10, 'dd': 15.0, 'sl_mult': 1.5},
            '3': {'name': '⚖️ BALANCED (2% risk)',       'risk_trade': 0.02, 'global': 0.08, 'daily': 0.15, 'dd': 20.0, 'sl_mult': 2.0},
            '4': {'name': '⚔️ AGGRESSIVE (3% risk)',     'risk_trade': 0.03, 'global': 0.10, 'daily': 0.20, 'dd': 25.0, 'sl_mult': 2.5},
            '5': {'name': '🔥 DEGEN (5% risk)',          'risk_trade': 0.05, 'global': 0.15, 'daily': 0.30, 'dd': 40.0, 'sl_mult': 3.0}
        }
        
        # 2. Tampilkan menu preset yang baru
        print("Pilih preset (akan dihitung dari saldo USD ekuivalen):")
        for key, p in presets.items():
            print(f"{key}) {p['name']}")
        
        preset = input("\nSelect preset (1-5): ").strip()
        
        if preset in presets:
            p = presets[preset]
            
            # 3. Hitung nilai absolut USD dari persentase
            calc_max_loss_trade_usd = balance_usd * p['risk_trade']
            calc_global_max_loss_usd = balance_usd * p['global']
            calc_daily_limit_usd = balance_usd * p['daily']
            
            # 4. Simpan nilai USD yang sudah dihitung ke config
            config['current']['max_loss_per_trade'] = round(calc_max_loss_trade_usd, 2)
            config['current']['global_max_loss'] = round(calc_global_max_loss_usd, 2)
            config['current']['daily_loss_limit'] = round(calc_daily_limit_usd, 2)
            config['current']['max_drawdown_percent'] = p['dd']
            config['current']['sl_multiplier'] = p['sl_mult']
            config['current']['stop_loss_protection'] = True # Selalu aktifkan
            
            save_config(config)
            
            print(f"\n✅ Preset '{p['name']}' diterapkan!")
            print(f"   (Berdasarkan saldo ${balance_usd:,.2f} USD)")
            print(f"   Max loss/trade: ${calc_max_loss_trade_usd:.2f} USD ({currency_symbol}{calc_max_loss_trade_usd * rate:,.0f})")
            print(f"   Global max loss: ${calc_global_max_loss_usd:.2f} USD ({currency_symbol}{calc_global_max_loss_usd * rate:,.0f})")
            print(f"   Daily limit: ${calc_daily_limit_usd:.2f} USD ({currency_symbol}{calc_daily_limit_usd * rate:,.0f})")
            print(f"   Max drawdown: {p['dd']:.1f}%")
            print(f"   SL Multiplier: {p['sl_mult']:.1f}x ATR")
        
        else:
            print("❌ Pilihan tidak valid")

    elif choice == '9':
        # Emergency Stop
        print("\n🚨 EMERGENCY STOP")
        print("="*70)
        print("This will:")
        print("  1. STOP all trading immediately")
        print("  2. CLOSE all open positions")
        print("  3. SET emergency_stop_active = True")
        print("="*70)
        
        confirm = input("\nType 'EMERGENCY' to confirm: ").strip()
        
        if confirm == 'EMERGENCY':
            config['current']['emergency_stop_active'] = True
            config['current']['auto_trade'] = False
            save_config(config)
            
            # Close all positions
            positions = mt5.positions_get()
            if positions:
                bot_positions = [p for p in positions if p.magic == 234000]
                
                print(f"\n🔴 EMERGENCY STOP ACTIVATED!")
                print(f"Closing {len(bot_positions)} position(s)...")
                
                from trading.trade_manager import TradeManager
                trader = TradeManager(config)
                
                for p in bot_positions:
                    print(f"  Closing {p.symbol} #{p.ticket} (${p.profit:+.2f})")
                    trader.close_position(p.ticket)
                    time.sleep(0.5)
                
                print(f"\n✅ All positions closed!")
            else:
                print("\n✅ No open positions to close")
            
            print(f"\n🚨 EMERGENCY STOP ACTIVE - Trading disabled!")
            print(f"To resume: Menu 32 → Reset emergency stop")
        else:
            print("\n❌ Emergency stop cancelled")
    
    return config

def menu_33_safety_dashboard(config: dict = None) -> None:
    """Real-time safety monitoring dashboard"""
    print("\n🛡️ SAFETY DASHBOARD")
    print("Press Ctrl+C to stop\n")
    
    if config is None:
        config = load_config()
    
    try:
        while True:
            account = mt5.account_info()
            if not account:
                print("❌ Cannot get account info")
                break
            
            positions = mt5.positions_get()
            bot_positions = [p for p in positions if p.magic == 234000] if positions else []
            
            # Calculate metrics
            starting_balance = config.get('current', {}).get('starting_balance', account.balance)
            daily_pl = account.balance - starting_balance
            floating_pl = sum(p.profit for p in bot_positions)
            total_pl = daily_pl + floating_pl
            
            drawdown = ((starting_balance - account.equity) / starting_balance) * 100 if starting_balance > 0 else 0
            
            # Limits
            max_loss_trade = config.get('current', {}).get('max_loss_per_trade', 1.0)
            global_max = config.get('current', {}).get('global_max_loss', 5.0)
            daily_limit = config.get('current', {}).get('daily_loss_limit', 10.0)
            max_dd = config.get('current', {}).get('max_drawdown_percent', 20.0)
            
            # Display
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            print(f"\r[{timestamp}] ", end='')
            print(f"Balance: ${account.balance:.2f} | ", end='')
            print(f"Equity: ${account.equity:.2f} | ", end='')
            print(f"Daily: ${daily_pl:+.2f}/{daily_limit:.2f} | ", end='')
            print(f"Floating: ${floating_pl:+.2f}/{global_max:.2f} | ", end='')
            print(f"DD: {drawdown:.1f}%/{max_dd:.1f}%  ", end='', flush=True)
            
            # Warnings
            if abs(daily_pl) >= daily_limit * 0.8:
                print("\n   ⚠️ Daily loss near limit!", end='')
            if abs(floating_pl) >= global_max * 0.8:
                print("\n   ⚠️ Floating loss near limit!", end='')
            if drawdown >= max_dd * 0.8:
                print("\n   ⚠️ Drawdown near limit!", end='')
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard closed")

def menu_34_balance_optimizer(config: dict, rate: float, currency_symbol: str) -> dict:
    """Auto-optimize settings based on current balance"""
    print("\n💰 BALANCE-BASED AUTO OPTIMIZER")
    print("="*70)
    
    account = mt5.account_info()
    if not account:
        print("❌ Cannot get account info")
        return config
    
    balance = account.balance # Ini sudah dalam mata uang akun (cth: 250000.00 IDR)
    
    balance_in_usd = balance / rate
    
    print(f"Current Balance: {currency_symbol}{balance:,.0f} (sekitar ${balance_in_usd:,.2f} USD)")
    
    # Deteksi kategori berdasarkan balance_in_usd
    if balance_in_usd < 20:
        category = "MICRO"
        print(f"Account Category: MICRO (<$20 USD)")
    elif balance_in_usd < 100:
        category = "MINI"
        print(f"Account Category: MINI ($20-$100 USD)")
    elif balance_in_usd < 500:
        category = "SMALL"
        print(f"Account Category: SMALL ($100-$500 USD)")
    elif balance < 1000:
        category = "MEDIUM"
        print(f"Account Category: MEDIUM ($500-$1000)")
    else:
        category = "LARGE"
        print(f"Account Category: LARGE (>$1000)")
    
    # Define optimal settings per category
    optimal_settings = {
        "MICRO": {
            "description": "Ultra-safe for $5-$20",
            "lot": 0.01,
            "max_positions_per_symbol": 1,
            "max_total_positions": 2,
            "max_daily_trades": 5,
            "auto_close_target": 0.2,
            "auto_close_total_target": 0.5,
            "max_loss_per_trade": 0.2,
            "global_max_loss": 0.4,
            "daily_loss_limit": 1.0,
            "max_drawdown_percent": 40.0,
            "risk_percent_per_trade": 0.5,
            "sl_multiplier": 0.8,
            "min_signal_strength": 0.3,
            "trade_mode": "CONSERVATIVE",
            "enable_scalping": False,
            "enable_multi_symbol": False,
            "auto_analyze_interval": 5,
            "bep_min_profit": 0.1,
            "step_lock_init": 0.15
        },
        "MINI": {
            "description": "Safe for $20-$100",
            "lot": 0.01,
            "max_positions_per_symbol": 2,
            "max_total_positions": 3,
            "max_daily_trades": 10,
            "auto_close_target": 0.3,
            "auto_close_total_target": 1.0,
            "max_loss_per_trade": 0.5,
            "global_max_loss": 1.0,
            "daily_loss_limit": 3.0,
            "max_drawdown_percent": 30.0,
            "risk_percent_per_trade": 1.0,
            "sl_multiplier": 1.0,
            "min_signal_strength": 0.25,
            "trade_mode": "CONSERVATIVE",
            "enable_scalping": False,
            "enable_multi_symbol": True,
            "auto_analyze_interval": 3,
            "bep_min_profit": 0.2,
            "step_lock_init": 0.3
        },
        "SMALL": {
            "description": "Moderate for $100-$500",
            "lot": 0.02,
            "max_positions_per_symbol": 3,
            "max_total_positions": 5,
            "max_daily_trades": 20,
            "auto_close_target": 0.5,
            "auto_close_total_target": 3.0,
            "max_loss_per_trade": 2.0,
            "global_max_loss": 5.0,
            "daily_loss_limit": 10.0,
            "max_drawdown_percent": 25.0,
            "risk_percent_per_trade": 1.5,
            "sl_multiplier": 1.5,
            "min_signal_strength": 0.2,
            "trade_mode": "MODERATE",
            "enable_scalping": True,
            "enable_multi_symbol": True,
            "auto_analyze_interval": 2,
            "bep_min_profit": 0.3,
            "step_lock_init": 0.5
        },
        "MEDIUM": {
            "description": "Aggressive for $500-$1000",
            "lot": 0.05,
            "max_positions_per_symbol": 5,
            "max_total_positions": 10,
            "max_daily_trades": 50,
            "auto_close_target": 1.0,
            "auto_close_total_target": 10.0,
            "max_loss_per_trade": 5.0,
            "global_max_loss": 15.0,
            "daily_loss_limit": 30.0,
            "max_drawdown_percent": 20.0,
            "risk_percent_per_trade": 2.0,
            "sl_multiplier": 2.0,
            "min_signal_strength": 0.15,
            "trade_mode": "AGGRESSIVE",
            "enable_scalping": True,
            "enable_multi_symbol": True,
            "auto_analyze_interval": 1,
            "bep_min_profit": 0.5,
            "step_lock_init": 1.0
        },
        "LARGE": {
            "description": "Rapid Fire for >$1000",
            "lot": 0.1,
            "max_positions_per_symbol": 10,
            "max_total_positions": 20,
            "max_daily_trades": 100,
            "auto_close_target": 5.0,
            "auto_close_total_target": 50.0,
            "max_loss_per_trade": 20.0,
            "global_max_loss": 50.0,
            "daily_loss_limit": 100.0,
            "max_drawdown_percent": 15.0,
            "risk_percent_per_trade": 2.5,
            "sl_multiplier": 2.0,
            "min_signal_strength": 0.1,
            "trade_mode": "RAPID_FIRE",
            "enable_scalping": True,
            "enable_multi_symbol": True,
            "auto_analyze_interval": 1,
            "bep_min_profit": 1.0,
            "step_lock_init": 2.0
        }
    }
    
    settings = optimal_settings.get(category, optimal_settings["SMALL"]) # Default ke SMALL jika ada error
    
    print(f"\n📋 RECOMMENDED SETTINGS ({settings['description']}):")
    print(f"="*70)
    print(f"\n🎯 Position Limits:")
    print(f"   Max positions/symbol: {settings['max_positions_per_symbol']}")
    print(f"   Max total positions: {settings['max_total_positions']}")
    print(f"   Max daily trades: {settings['max_daily_trades']}")
    
    print(f"\n💰 Profit Targets:")
    # Tampilkan USD dan konversinya
    print(f"   Per trade: ${settings['auto_close_target']:.2f} USD ({currency_symbol}{settings['auto_close_target'] * rate:,.0f})")
    print(f"   Total target: ${settings['auto_close_total_target']:.2f} USD ({currency_symbol}{settings['auto_close_total_target'] * rate:,.0f})")
    
    print(f"\n🛡️ Loss Protection:")
    # Tampilkan USD dan konversinya
    print(f"   Max loss/trade: ${settings['max_loss_per_trade']:.2f} USD ({currency_symbol}{settings['max_loss_per_trade'] * rate:,.0f})")
    print(f"   Global max loss: ${settings['global_max_loss']:.2f} USD ({currency_symbol}{settings['global_max_loss'] * rate:,.0f})")
    print(f"   Daily loss limit: ${settings['daily_loss_limit']:.2f} USD ({currency_symbol}{settings['daily_loss_limit'] * rate:,.0f})")
    print(f"   Max drawdown: {settings['max_drawdown_percent']:.1f}%")
    
    print(f"\n⚙️ Trading Settings:")
    print(f"   Lot size: {settings['lot']}")
    print(f"   Risk per trade: {settings['risk_percent_per_trade']:.1f}%")
    print(f"   SL multiplier: {settings['sl_multiplier']:.1f}x ATR")
    print(f"   Min signal strength: {settings['min_signal_strength']:.0%}")
    print(f"   Trade mode: {settings['trade_mode']}")
    
    print(f"\n{'='*70}")
    
    choice = input("\nApply these settings? (y/n): ").strip().lower()
    
    if choice == 'y':
        # Apply all settings (ini sudah benar, karena settings DALAM USD)
        for key, value in settings.items():
            if key != 'description':
                config['current'][key] = value
        
        # Also set starting balance
        config['current']['starting_balance'] = balance
        config['current']['micro_account_mode'] = (category == "MICRO")
        
        save_config(config)
        
        print(f"\n✅ Settings optimized for ${balance:.2f} balance!")
        print(f"   Category: {category}")
        print(f"   Safety Level: {settings['description']}")
        
        # Show risk summary
        print(f"\n📊 RISK SUMMARY:")
        print(f"   If you lose 1 trade: -${settings['max_loss_per_trade']:.2f} ({settings['max_loss_per_trade']/balance*100:.1f}%)")
        print(f"   If all positions hit SL: -${settings['global_max_loss']:.2f} ({settings['global_max_loss']/balance*100:.1f}%)")
        print(f"   Max daily loss allowed: -${settings['daily_loss_limit']:.2f} ({settings['daily_loss_limit']/balance*100:.1f}%)")
        print(f"   Equity can drop to: ${balance * (1 - settings['max_drawdown_percent']/100):.2f} before stop")
        
    else:
        print("\n❌ Settings not applied")
    
    return config

def menu_35_safety_calculator(config: dict, rate: float, currency_symbol: str) -> dict:
    """Calculate safe settings manually"""
    print("\n🧮 SAFETY SETTINGS CALCULATOR")
    print("="*70)
    
    account = mt5.account_info()
    if account:
        balance = account.balance # Ini IDR
        balance_in_usd = balance / rate
        print(f"Current Balance: {currency_symbol}{balance:,.0f} (${balance_in_usd:,.2f} USD)")
    else:
        try:
            balance = float(input("Enter your balance: $"))
        except:
            print("❌ Invalid balance")
            return config  # RETURN CONFIG
    
    print(f"\n📊 CALCULATING SAFE SETTINGS FOR {currency_symbol}{balance:,.0f}...")
    
    # Safe percentages
    risk_per_trade_percent = 3.0  # Max 3% risk per trade
    max_total_risk_percent = 10.0  # Max 10% total exposure
    daily_risk_percent = 15.0  # Max 15% daily loss
    drawdown_percent = 30.0  # Max 30% drawdown
    
    # Calculate USD values
    max_loss_per_trade_usd = balance_in_usd * (risk_per_trade_percent / 100)
    global_max_loss_usd = balance_in_usd * (max_total_risk_percent / 100)
    daily_loss_limit_usd = balance_in_usd * (daily_risk_percent / 100)
    
    # Calculate max positions
    max_positions = int(global_max_loss_usd / max_loss_per_trade_usd)
    max_positions = max(1, min(max_positions, 10))  # Between 1-10
    
    # Calculate profit targets (risk:reward 2:1)
    profit_per_trade_usd = max_loss_per_trade_usd * 2
    total_profit_target_usd = global_max_loss_usd * 2
    
    # Suggest lot size
    if balance < 20:
        lot_size = 0.01
    elif balance < 100:
        lot_size = 0.03
    elif balance < 500:
        lot_size = 0.05
    else:
        lot_size = 0.1
    
    print(f"\n✅ RECOMMENDED SAFE SETTINGS:")
    print(f"="*70)
    
    print(f"\n🛡️ LOSS PROTECTION (USD | {currency_symbol}):")
    print(f"   Max loss per trade: ${max_loss_per_trade_usd:.2f} | {currency_symbol}{max_loss_per_trade_usd * rate:,.0f}")
    print(f"   Global max loss: ${global_max_loss_usd:.2f} | {currency_symbol}{global_max_loss_usd * rate:,.0f}")
    print(f"   Daily loss limit: ${daily_loss_limit_usd:.2f} | {currency_symbol}{daily_loss_limit_usd * rate:,.0f}")
    print(f"   Max drawdown: {drawdown_percent:.1f}%")
    print(f"   → Equity stop level: {currency_symbol}{balance * (1 - drawdown_percent/100):,.0f}")
    
    print(f"\n💰 PROFIT TARGETS (USD | {currency_symbol}):")
    print(f"   Target per trade: ${profit_per_trade_usd:.2f} | {currency_symbol}{profit_per_trade_usd * rate:,.0f}")
    print(f"   Total target: ${total_profit_target_usd:.2f} | {currency_symbol}{total_profit_target_usd * rate:,.0f}")
    
    print(f"\n📊 POSITION LIMITS:")
    print(f"   Max positions: {max_positions}")
    print(f"   Suggested lot size: {lot_size}")
    
    print(f"\n📈 PROFIT POTENTIAL:")
    print(f"   If 1 trade hits TP: +${profit_per_trade_usd:.2f} ({profit_per_trade_usd/balance_in_usd*100:.1f}%)")
    print(f"   If all positions hit TP: +${total_profit_target_usd:.2f} ({total_profit_target_usd/balance_in_usd*100:.1f}%)")

    print(f"\n⚠️ RISK SCENARIOS:")
    print(f"   If 1 trade hits SL: -${max_loss_per_trade_usd:.2f} ({max_loss_per_trade_usd/balance_in_usd*100:.1f}%)")
    print(f"   If all positions hit SL: -${global_max_loss_usd:.2f} ({global_max_loss_usd/balance_in_usd*100:.1f}%)")
    print(f"   Max daily loss: -${daily_loss_limit_usd:.2f} ({daily_risk_percent:.1f}%)")

    print(f"\n💡 RECOMMENDATIONS:")
    if balance < 20:
        print("   ⚠️ MICRO ACCOUNT - Trade VERY carefully!")
        print("   • Use ONLY 0.01 lot")
        print("   • Max 1-2 positions")
        print("   • Take profit quickly ($0.20-$0.30)")
        print("   • Stop trading if down $1.5 (15%)")
    elif balance < 100:
        print("   📊 MINI ACCOUNT - Conservative approach recommended")
        print("   • Use 0.03 lot")
        print("   • Max 2-3 positions")
        print("   • Target $0.50-$1 per trade")
    else:
        print("   ✅ Good balance for active trading")
        print("   • Can use 0.02-0.05 lot")
        print("   • Can trade multiple symbols")
        print("   • Use proper risk management")
    
    print(f"="*70)
    
    # Ask if user wants to apply these settings
    print(f"\n💡 Want to apply these settings automatically?")
    print(f"   Use Menu 34 (Auto Balance Optimizer) for one-click setup!")
    
    return config  # RETURN CONFIG!

def menu_36_apply_calculator_settings(config: dict, rate: float, currency_symbol: str) -> dict:
    """Apply settings from calculator"""
    print("\n⚡ QUICK APPLY CALCULATED SETTINGS")
    print("="*70)
    
    account = mt5.account_info()
    if not account:
        print("❌ Cannot get account info")
        return config
    
    balance = account.balance # Ini IDR
    balance_in_usd = balance / rate
    
    # Auto-calculate
    risk_per_trade_percent = 3.0
    max_total_risk_percent = 10.0
    daily_risk_percent = 15.0
    drawdown_percent = 40.0  # Increased for micro accounts
    
    max_loss_per_trade_usd = balance_in_usd * (risk_per_trade_percent / 100)
    global_max_loss_usd = balance_in_usd * (max_total_risk_percent / 100)
    daily_loss_limit_usd = balance_in_usd * (daily_risk_percent / 100)
    
    max_positions = max(1, min(int(global_max_loss_usd / max_loss_per_trade_usd), 10))
    
    profit_per_trade_usd = max_loss_per_trade_usd * 2
    total_profit_target_usd = global_max_loss_usd * 2
    
    if balance < 20:
        lot_size = 0.01
    elif balance < 100:
        lot_size = 0.03
    else:
        lot_size = 0.05
    
    print(f"Balance: {currency_symbol}{balance:,.0f} (${balance_in_usd:,.2f} USD)")
    print(f"\nSettings to apply (USD | {currency_symbol}):")
    print(f"  Lot: {lot_size}")
    print(f"  Max positions: {max_positions}")
    print(f"  Max loss/trade: ${max_loss_per_trade_usd:.2f} | {currency_symbol}{max_loss_per_trade_usd * rate:,.0f}")
    print(f"  Global max loss: ${global_max_loss_usd:.2f} | {currency_symbol}{global_max_loss_usd * rate:,.0f}")
    print(f"  Daily limit: ${daily_loss_limit_usd:.2f} | {currency_symbol}{daily_loss_limit_usd * rate:,.0f}")
    print(f"  Max drawdown: {drawdown_percent:.0f}%")
    print(f"  TP/trade: ${profit_per_trade_usd:.2f} | {currency_symbol}{profit_per_trade_usd * rate:,.0f}")
    print(f"  Total TP: ${total_profit_target_usd:.2f} | {currency_symbol}{total_profit_target_usd * rate:,.0f}")
    
    choice = input("\nApply? (y/n): ").strip().lower()
    
    if choice == 'y':
        config['current']['lot'] = lot_size
        config['current']['max_total_positions'] = max_positions
        # Simpan nilai USD
        config['current']['max_loss_per_trade'] = round(max_loss_per_trade_usd, 2)
        config['current']['global_max_loss'] = round(global_max_loss_usd, 2)
        config['current']['daily_loss_limit'] = round(daily_loss_limit_usd, 2)
        config['current']['max_drawdown_percent'] = drawdown_percent
        config['current']['auto_close_target'] = round(profit_per_trade_usd, 2)
        config['current']['auto_close_total_target'] = round(total_profit_target_usd, 2)
        config['current']['starting_balance'] = balance # Starting balance tetap dalam mata uang akun
        
        save_config(config)
        
        print("\n✅ Settings applied!")
    else:
        print("\n❌ Cancelled")
    
    return config

def menu_37_profit_planner(config: dict, rate: float, currency_symbol: str) -> dict:
    """Plan daily profit target and calculate required trades"""
    print("\n💰 DAILY PROFIT PLANNER")
    print("="*70)
    
    account = mt5.account_info()
    if not account:
        print("❌ Cannot get account info")
        return config
    
    balance = account.balance # Ini IDR
    current_settings = config['current']
    
    # Ambil nilai USD dari config dan konversi
    tp_per_trade_usd = current_settings.get('auto_close_target', 0.5)
    tp_per_trade_display = tp_per_trade_usd * rate
    
    print(f"Current Balance: {currency_symbol}{balance:,.0f}")
    print(f"Current TP/trade: {currency_symbol}{tp_per_trade_display:,.0f} (${tp_per_trade_usd:.2f} USD)")
    print(f"Current Max Daily Trades: {current_settings.get('max_daily_trades', 5)}")
    
    print("\n📊 PROFIT PLANNER OPTIONS:")
    print(f"1) Set DAILY PROFIT TARGET ({currency_symbol})")
    print("2) Set MAX DAILY TRADES (auto-calculate max profit)")
    print("3) AGGRESSIVE mode (maximize trades with current balance)")
    print("4) CONSERVATIVE mode (safe trading)")
    print("0) ← Back")
    
    choice = input("\nSelect (0-4): ").strip()
    
    if choice == '0':
        return config
    
    if choice == '1':
        # Set daily target, calculate trades needed
        try:
            target_display = float(input(f"\nEnter daily profit target ({currency_symbol}, e.g., {5.0 * rate:,.0f}): {currency_symbol}"))
            
            if target_display <= 0 or target_display > balance * 0.5:
                print(f"❌ Target harus antara {currency_symbol}1 dan {currency_symbol}{balance * 0.5:,.0f} (50% of balance)")
                return config
            
            # Gunakan nilai yang sudah dikonversi
            avg_profit_per_trade_display = tp_per_trade_display
            
            # Calculate trades needed
            trades_needed = int(target_display / avg_profit_per_trade_display) + 1
            
            # Add buffer for losses (assume 70% win rate)
            trades_with_buffer = int(trades_needed / 0.7)
            
            print(f"\n📊 CALCULATION:")
            print(f"   Daily Target: {currency_symbol}{target_display:,.0f}")
            print(f"   Avg Profit/Trade: {currency_symbol}{avg_profit_per_trade_display:,.0f}")
            print(f"   Trades needed (100% win): {trades_needed}")
            print(f"   Trades needed (70% win rate): {trades_with_buffer}")
            
            # Calculate risk
            max_loss_per_trade_usd = current_settings.get('max_loss_per_trade', 0.3)
            max_loss_per_trade_display = max_loss_per_trade_usd * rate
            max_possible_loss_display = trades_with_buffer * max_loss_per_trade_display * 0.3
            
            print(f"\n⚠️ RISK ANALYSIS:")
            print(f"   Max loss if 30% of trades hit SL: {currency_symbol}{max_possible_loss_display:,.0f}")
            print(f"   Net profit (target - risk): {currency_symbol}{target_display - max_possible_loss_display:,.0f}")
            
            # Safety check
            if max_possible_loss_display > balance * 0.2:
                print(f"\n⚠️ WARNING: Risk too high ({max_possible_loss_display/balance*100:.1f}% of balance)")
                print(f"   Recommended: Reduce target or increase balance")
                
                confirm = input("\nContinue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    return config
            
            # Apply settings
            config['current']['max_daily_trades'] = trades_with_buffer
            save_config(config)
            
            print(f"\n✅ Settings updated!")
            print(f"   Max daily trades: {trades_with_buffer}")
            print(f"   Expected daily profit: {currency_symbol}{target_display:.2f}")
            
        except ValueError:
            print("❌ Invalid input")
    
    elif choice == '2':
        # Set max trades, calculate max profit
        try:
            max_trades = int(input("\nEnter max daily trades (e.g., 20): "))
            
            if max_trades < 1 or max_trades > 200:
                print("❌ Trades must be between 1 and 200")
                return config
            
            # Ambil nilai USD dan konversi
            avg_profit_per_trade_usd = current_settings.get('auto_close_target', 0.5)
            avg_profit_per_trade_display = avg_profit_per_trade_usd * rate
            max_loss_per_trade_usd = current_settings.get('max_loss_per_trade', 0.3)
            max_loss_per_trade_display = max_loss_per_trade_usd * rate
            
            # Calculate potential profit (70% win rate)
            winning_trades = int(max_trades * 0.7)
            losing_trades = max_trades - winning_trades
            
            expected_profit_display = (winning_trades * avg_profit_per_trade_display) - (losing_trades * max_loss_per_trade_display)
            max_profit_display = max_trades * avg_profit_per_trade_display
            
            print(f"\n📊 CALCULATION:")
            print(f"   Max trades: {max_trades}")
            print(f"   Avg profit/trade: {currency_symbol}{avg_profit_per_trade_display:,.0f}")
            print(f"   Max profit (100% win): {currency_symbol}{max_profit_display:,.0f} ({max_profit_display/balance*100:.1f}%)")
            print(f"   Expected profit (70% win): {currency_symbol}{expected_profit_display:,.0f} ({expected_profit_display/balance*100:.1f}%)")
            print(f"   Winning trades: {winning_trades}")
            print(f"   Losing trades: {losing_trades}")
            
            # Apply
            config['current']['max_daily_trades'] = max_trades
            save_config(config)
            
            print(f"\n✅ Max daily trades set to: {max_trades}")
            
        except ValueError:
            print("❌ Invalid input")
    
    elif choice == '3':
        # AGGRESSIVE mode
        print("\n🔥 AGGRESSIVE PROFIT MODE")
        print("="*70)
        
        balance_in_usd = balance / rate
        
        # Calculate aggressive settings (DALAM USD)
        risk_per_trade_percent = 0.02  # 2% per trade
        max_risk_usd = balance_in_usd * risk_per_trade_percent
        
        target_profit_per_trade_usd = 0.5
        
        # Calculate lot size for Gold
        # For XAUUSDm: risk = sl_distance * lot * 10
        # Assume SL distance ~$5
        sl_distance = 5.0
        lot_size = max_risk_usd / (sl_distance * 10)
        lot_size = max(0.02, round(lot_size, 2))
        
        # Calculate max trades (target $5/day)
        daily_target_usd = min(balance_in_usd * 0.3, 5.0)
        max_trades = int(daily_target_usd / target_profit_per_trade_usd / 0.7)
        max_trades = max(10, min(max_trades, 50))
        
        print(f"Aggressive Settings (USD | {currency_symbol}):")
        print(f"  Lot size: {lot_size}")
        print(f"  Max loss/trade: ${max_risk_usd:.2f} | {currency_symbol}{max_risk_usd * rate:,.0f}")
        print(f"  Target profit/trade: ${target_profit_per_trade_usd:.2f} | {currency_symbol}{target_profit_per_trade_usd * rate:,.0f}")
        print(f"  Max daily trades: {max_trades}")
        print(f"  Daily target: ${daily_target_usd:.2f} | {currency_symbol}{daily_target_usd * rate:,.0f}")
        print(f"  Max positions: 5")
        
        confirm = input("\nApply? (y/n): ").strip().lower()
        
        if confirm == 'y':
            # Simpan nilai USD
            config['current']['lot'] = lot_size
            config['current']['max_loss_per_trade'] = round(max_risk_usd, 2)
            config['current']['auto_close_target'] = target_profit_per_trade_usd
            config['current']['max_daily_trades'] = max_trades
            config['current']['max_total_positions'] = 5
            config['current']['global_max_loss'] = round(max_risk_usd * 3, 2)
            config['current']['trade_mode'] = 'AGGRESSIVE'
            
            save_config(config)
            print("\n✅ AGGRESSIVE mode activated!")
    
    elif choice == '4':
        # CONSERVATIVE mode (Simpan nilai USD)
        print("\n🛡️ CONSERVATIVE SAFE MODE")
        
        config['current']['lot'] = 0.02
        config['current']['max_loss_per_trade'] = 0.2
        config['current']['auto_close_target'] = 0.3
        config['current']['max_daily_trades'] = 10
        config['current']['max_total_positions'] = 2
        config['current']['trade_mode'] = 'CONSERVATIVE'
        
        save_config(config)
        print("\n✅ CONSERVATIVE mode activated!")
    
    return config

def menu_38_dynamic_trade_limit(config: dict) -> dict:
    """Enable dynamic trade limit based on profit"""
    print("\n📈 DYNAMIC TRADE LIMIT")
    print("="*70)
    
    current = config['current']
    enabled = current.get('dynamic_trade_limit', False)
    
    print(f"Status: {'ENABLED' if enabled else 'DISABLED'}")
    
    if enabled:
        print(f"Base trades: {current.get('base_daily_trades', 5)}")
        print(f"Bonus per $1 profit: {current.get('bonus_trades_per_dollar', 2)}")
        print(f"Max trades: {current.get('max_daily_trades_cap', 50)}")
    
    print("\nHow it works:")
    print("  • Start with BASE trades (e.g., 5)")
    print("  • For every $1 profit today, get BONUS trades (e.g., +2)")
    print("  • Max cap to prevent overtrading")
    print("\nExample:")
    print("  • Start: 5 trades allowed")
    print("  • After $1 profit: 5 + 2 = 7 trades")
    print("  • After $2 profit: 5 + 4 = 9 trades")
    print("  • After $5 profit: 5 + 10 = 15 trades")
    
    print("\n📋 OPTIONS:")
    print("1) ENABLE dynamic limit")
    print("2) DISABLE (use fixed limit)")
    print("3) Configure settings")
    print("0) ← Back")
    
    choice = input("\nSelect (0-3): ").strip()
    
    if choice == '1':
        config['current']['dynamic_trade_limit'] = True
        config['current']['base_daily_trades'] = 5
        config['current']['bonus_trades_per_dollar'] = 2
        config['current']['max_daily_trades_cap'] = 50
        save_config(config)
        print("\n✅ Dynamic trade limit ENABLED!")
        
    elif choice == '2':
        config['current']['dynamic_trade_limit'] = False
        save_config(config)
        print("\n✅ Dynamic trade limit DISABLED")
        
    elif choice == '3':
        try:
            base = int(input("Base daily trades: "))
            bonus = int(input("Bonus trades per $1 profit: "))
            cap = int(input("Max cap: "))
            
            config['current']['base_daily_trades'] = base
            config['current']['bonus_trades_per_dollar'] = bonus
            config['current']['max_daily_trades_cap'] = cap
            save_config(config)
            
            print("\n✅ Settings saved!")
        except:
            print("❌ Invalid input")
    
    return config

def menu_39_signal_strength_adjuster(config: dict) -> dict:
    """Adjust signal strength threshold untuk lebih/kurang sinyal"""
    print("\n🎯 SIGNAL STRENGTH ADJUSTER")
    print("="*70)
    
    current_strength = config['current'].get('min_signal_strength', 0.2)
    current_mode = config['current'].get('trade_mode', 'MODERATE')
    
    print(f"Current Settings:")
    print(f"  Trade Mode: {current_mode}")
    print(f"  Min Signal Strength: {current_strength:.0%}")
    
    # Calculate expected signals
    if current_strength >= 0.4:
        signal_freq = "Very Low (1-2 signals/hour)"
        quality = "Very High Quality"
    elif current_strength >= 0.3:
        signal_freq = "Low (2-4 signals/hour)"
        quality = "High Quality"
    elif current_strength >= 0.2:
        signal_freq = "Medium (4-8 signals/hour)"
        quality = "Good Quality"
    elif current_strength >= 0.15:
        signal_freq = "High (8-15 signals/hour)"
        quality = "Medium Quality"
    else:
        signal_freq = "Very High (15+ signals/hour)"
        quality = "Mixed Quality"
    
    print(f"  Expected Frequency: {signal_freq}")
    print(f"  Signal Quality: {quality}")
    
    print("\n📊 PRESETS:")
    print("1) ULTRA SELECTIVE (40%+) - Only strongest signals, 1-2/hour")
    print("2) SELECTIVE (30%) - High quality, 2-4/hour")
    print("3) BALANCED (20%) - Good mix, 4-8/hour ← RECOMMENDED")
    print("4) AGGRESSIVE (15%) - More trades, 8-15/hour")
    print("5) VERY AGGRESSIVE (10%) - Maximum trades, 15+/hour")
    print("6) CUSTOM - Set your own %")
    print("0) ← Back")
    
    choice = input("\nSelect preset (0-6): ").strip()
    
    presets = {
        '1': (0.4, "ULTRA_SELECTIVE"),
        '2': (0.3, "SELECTIVE"),
        '3': (0.2, "BALANCED"),
        '4': (0.15, "AGGRESSIVE"),
        '5': (0.1, "VERY_AGGRESSIVE")
    }
    
    if choice in presets:
        strength, mode = presets[choice]
        
        # Also enable/disable features based on mode
        if choice in ['1', '2']:  # Ultra/Selective
            config['current']['enable_scalping'] = False
            config['current']['enable_pattern_trading'] = True
            config['current']['enable_breakout_trading'] = False
            enabled_features = "Patterns only"
        elif choice == '3':  # Balanced
            config['current']['enable_scalping'] = True
            config['current']['enable_pattern_trading'] = True
            config['current']['enable_breakout_trading'] = True
            enabled_features = "All features"
        else:  # Aggressive/Very Aggressive
            config['current']['enable_scalping'] = True
            config['current']['enable_pattern_trading'] = True
            config['current']['enable_breakout_trading'] = True
            config['current']['enable_multi_timeframe'] = True
            enabled_features = "All features + Multi-TF"
        
        config['current']['min_signal_strength'] = strength
        save_config(config)
        
        print(f"\n✅ Signal strength set to {strength:.0%}")
        print(f"   Mode: {mode}")
        print(f"   Features: {enabled_features}")
        
    elif choice == '6':
        try:
            custom = float(input("Enter signal strength (0.05-0.50): "))
            if 0.05 <= custom <= 0.5:
                config['current']['min_signal_strength'] = custom
                save_config(config)
                print(f"\n✅ Custom strength: {custom:.0%}")
            else:
                print("❌ Must be between 5% and 50%")
        except:
            print("❌ Invalid input")
    
    return config

def menu_40_trade_statistics(config: dict) -> None:
    """Show comprehensive trade statistics"""
    print("\n📊 TRADE STATISTICS")
    print("="*70)
    
    # Get trades from today
    from datetime import datetime, timedelta
    from_date = datetime.now().replace(hour=0, minute=0, second=0)
    to_date = datetime.now()
    
    deals = mt5.history_deals_get(from_date, to_date)
    
    if not deals:
        print("No trades today")
        return
    
    # Filter bot trades
    bot_deals = [d for d in deals if d.magic == 234000 and d.entry == 1]  # Exit deals only
    
    if not bot_deals:
        print("No completed trades today")
        return
    
    # Calculate statistics
    total_trades = len(bot_deals)
    winning_trades = len([d for d in bot_deals if d.profit > 0])
    losing_trades = len([d for d in bot_deals if d.profit < 0])
    
    total_profit = sum(d.profit for d in bot_deals if d.profit > 0)
    total_loss = abs(sum(d.profit for d in bot_deals if d.profit < 0))
    net_profit = total_profit - total_loss
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
    
    avg_win = (total_profit / winning_trades) if winning_trades > 0 else 0
    avg_loss = (total_loss / losing_trades) if losing_trades > 0 else 0
    
    # Best and worst trades
    best_trade = max(bot_deals, key=lambda d: d.profit) if bot_deals else None
    worst_trade = min(bot_deals, key=lambda d: d.profit) if bot_deals else None
    
    print(f"\n📈 PERFORMANCE TODAY:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Winning: {winning_trades} ({win_rate:.1f}%)")
    print(f"   Losing: {losing_trades} ({100-win_rate:.1f}%)")
    
    print(f"\n💰 PROFIT & LOSS:")
    print(f"   Gross Profit: ${total_profit:.2f}")
    print(f"   Gross Loss: ${total_loss:.2f}")
    print(f"   Net Profit: ${net_profit:+.2f}")
    print(f"   Profit Factor: {profit_factor:.2f}")
    
    print(f"\n📊 AVERAGES:")
    print(f"   Avg Win: ${avg_win:.2f}")
    print(f"   Avg Loss: ${avg_loss:.2f}")
    print(f"   Risk:Reward: 1:{avg_win/avg_loss:.2f}" if avg_loss > 0 else "   Risk:Reward: N/A")
    
    if best_trade:
        print(f"\n🏆 BEST TRADE:")
        print(f"   Symbol: {best_trade.symbol}")
        print(f"   Profit: ${best_trade.profit:.2f}")
        print(f"   Time: {datetime.fromtimestamp(best_trade.time).strftime('%H:%M:%S')}")
    
    if worst_trade:
        print(f"\n💔 WORST TRADE:")
        print(f"   Symbol: {worst_trade.symbol}")
        print(f"   Loss: ${worst_trade.profit:.2f}")
        print(f"   Time: {datetime.fromtimestamp(worst_trade.time).strftime('%H:%M:%S')}")
    
    # Account info
    account = mt5.account_info()
    if account:
        starting_balance = config['current'].get('starting_balance', account.balance)
        roi_today = ((account.balance - starting_balance) / starting_balance * 100) if starting_balance > 0 else 0
        
        print(f"\n💼 ACCOUNT:")
        print(f"   Starting Balance: ${starting_balance:.2f}")
        print(f"   Current Balance: ${account.balance:.2f}")
        print(f"   ROI Today: {roi_today:+.2f}%")

def menu_41_market_condition_detector(config: dict) -> None:
    """Detect current market condition"""
    print("\n🌊 MARKET CONDITION DETECTOR")
    print("="*70)
    
    symbol = config['current']['symbol']
    
    # Get data for different timeframes
    h1_data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 50)
    h4_data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 50)
    
    if h1_data is None or h4_data is None:
        print("❌ Cannot get market data")
        return
    
    df_h1 = pd.DataFrame(h1_data)
    df_h4 = pd.DataFrame(h4_data)
    
    # Calculate indicators
    # ADX for trend strength
    def calculate_adx(df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = pd.concat([high - low, 
                       abs(high - close.shift()), 
                       abs(low - close.shift())], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (abs(minus_dm).rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1]
    
    # Bollinger Bands width for volatility
    def calculate_bb_width(df, period=20):
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        bb_width = (std * 4) / sma * 100
        return bb_width.iloc[-1]
    
    adx_h1 = calculate_adx(df_h1)
    adx_h4 = calculate_adx(df_h4)
    bb_width = calculate_bb_width(df_h1)
    
    # Determine trend
    sma_20_h1 = df_h1['close'].rolling(20).mean().iloc[-1]
    current_price = df_h1['close'].iloc[-1]
    
    if current_price > sma_20_h1:
        trend_h1 = "UPTREND"
    elif current_price < sma_20_h1:
        trend_h1 = "DOWNTREND"
    else:
        trend_h1 = "SIDEWAYS"
    
    # Market condition
    if adx_h1 > 25 and adx_h4 > 25:
        condition = "STRONG TRENDING"
        recommendation = "Use trend-following strategies, enable breakouts"
    elif adx_h1 > 25 or adx_h4 > 25:
        condition = "TRENDING"
        recommendation = "Good for breakouts and momentum trading"
    elif bb_width < 2:
        condition = "LOW VOLATILITY (Consolidation)"
        recommendation = "Wait for breakout or use scalping"
    else:
        condition = "RANGING"
        recommendation = "Use pattern trading, avoid breakouts"
    
    print(f"\n📊 {symbol} ANALYSIS:")
    print(f"   H1 Trend: {trend_h1}")
    print(f"   ADX H1: {adx_h1:.1f}")
    print(f"   ADX H4: {adx_h4:.1f}")
    print(f"   BB Width: {bb_width:.2f}%")
    
    print(f"\n🌊 MARKET CONDITION: {condition}")
    print(f"   💡 {recommendation}")
    
    # Auto-adjust settings
    print(f"\n⚙️ SUGGESTED SETTINGS:")
    if "TRENDING" in condition:
        print("   • Enable breakout trading")
        print("   • Disable scalping")
        print("   • Use wider stops")
    elif "RANGING" in condition:
        print("   • Enable pattern trading")
        print("   • Enable scalping")
        print("   • Use tighter stops")
    else:
        print("   • Wait for volatility")
        print("   • Reduce position size")

def menu_42_session_based_trading(config: dict) -> dict:
    """Enable trading during specific sessions"""
    print("\n🌍 SESSION-BASED TRADING")
    print("="*70)
    
    enabled = config['current'].get('session_trading_enabled', False)
    active_sessions = config['current'].get('active_sessions', ['ASIAN', 'LONDON', 'NEWYORK'])
    
    print(f"Status: {'ENABLED' if enabled else 'DISABLED'}")
    print(f"Active Sessions: {', '.join(active_sessions)}")
    
    print("\n📅 FOREX SESSIONS (Server Time):")
    print("   ASIAN    : 00:00 - 09:00 (Low volatility)")
    print("   LONDON   : 08:00 - 17:00 (High volatility)")
    print("   NEWYORK  : 13:00 - 22:00 (High volatility)")
    print("   OVERLAP  : 13:00 - 17:00 (Maximum volatility)")
    
    print("\n💡 Best for Gold (XAU):")
    print("   • LONDON session (most volatile)")
    print("   • NEWYORK session (strong trends)")
    print("   • Avoid late ASIAN (sideways)")
    
    print("\n📋 OPTIONS:")
    print("1) ENABLE session trading")
    print("2) DISABLE (trade 24/5)")
    print("3) Select active sessions")
    print("0) ← Back")
    
    choice = input("\nSelect (0-3): ").strip()
    
    if choice == '1':
        config['current']['session_trading_enabled'] = True
        config['current']['active_sessions'] = ['LONDON', 'NEWYORK']
        save_config(config)
        print("\n✅ Session trading enabled (London + NY)")
        
    elif choice == '2':
        config['current']['session_trading_enabled'] = False
        save_config(config)
        print("\n✅ Session trading disabled")
        
    elif choice == '3':
        print("\nSelect sessions to trade:")
        asian = input("Trade ASIAN session? (y/n): ").lower() == 'y'
        london = input("Trade LONDON session? (y/n): ").lower() == 'y'
        newyork = input("Trade NEWYORK session? (y/n): ").lower() == 'y'
        
        sessions = []
        if asian: sessions.append('ASIAN')
        if london: sessions.append('LONDON')
        if newyork: sessions.append('NEWYORK')
        
        config['current']['active_sessions'] = sessions
        config['current']['session_trading_enabled'] = True
        save_config(config)
        
        print(f"\n✅ Active sessions: {', '.join(sessions)}")
    
    return config

def menu_43_symbol_auto_setup(config: dict) -> dict:
    """Auto-detect and setup symbols for current broker"""
    print("\n🔍 SYMBOL AUTO-SETUP")
    print("="*70)
    
    if not symbol_detector:
        print("❌ Symbol detector not initialized")
        print("   Try reconnecting MT5")
        return config
    
    print("\n📊 Detecting available symbols...")
    
    # Get available Gold symbols
    gold_symbols = symbol_detector.get_available_gold_symbols()
    print(f"\n💰 GOLD Symbols Found: {len(gold_symbols)}")
    for i, sym in enumerate(gold_symbols[:5], 1):
        tick = mt5.symbol_info_tick(sym)
        if tick:
            print(f"   {i}. {sym} - Bid: {tick.bid:.2f}")
    
    # Get available Forex pairs
    forex_pairs = symbol_detector.get_available_forex_pairs()
    print(f"\n💱 FOREX Pairs Found: {len(forex_pairs)}")
    for pair, symbol in list(forex_pairs.items())[:5]:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            print(f"   {pair} → {symbol} - Bid: {tick.bid:.5f}")
    
    print("\n📋 SETUP OPTIONS:")
    print("1) Use GOLD only (recommended for micro accounts)")
    print("2) Use GOLD + EUR/USD")
    print("3) Use GOLD + 3 major pairs")
    print("4) Custom selection")
    print("0) ← Back")
    
    choice = input("\nSelect (0-4): ").strip()
    
    if choice == '0':
        return config
    
    selected_symbols = []
    
    if choice == '1':
        # Gold only
        if gold_symbols:
            selected_symbols = [gold_symbols[0]]
    
    elif choice == '2':
        # Gold + EURUSD
        if gold_symbols:
            selected_symbols.append(gold_symbols[0])
        
        eur_symbol = forex_pairs.get('EURUSD')
        if eur_symbol:
            selected_symbols.append(eur_symbol)
    
    elif choice == '3':
        # Gold + 3 majors
        if gold_symbols:
            selected_symbols.append(gold_symbols[0])
        
        major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
        for pair in major_pairs:
            symbol = forex_pairs.get(pair)
            if symbol:
                selected_symbols.append(symbol)
    
    elif choice == '4':
        # Custom selection
        print("\nAvailable symbols:")
        
        all_available = []
        
        # Add gold
        for g in gold_symbols[:3]:
            all_available.append(g)
            print(f"   {len(all_available)}. {g}")
        
        # Add forex
        for pair, symbol in list(forex_pairs.items())[:10]:
            all_available.append(symbol)
            print(f"   {len(all_available)}. {symbol} ({pair})")
        
        try:
            selections = input("\nEnter numbers separated by comma (e.g., 1,2,3): ").strip()
            indices = [int(x.strip()) - 1 for x in selections.split(',')]
            
            for idx in indices:
                if 0 <= idx < len(all_available):
                    selected_symbols.append(all_available[idx])
        except:
            print("❌ Invalid selection")
            return config
    
    if selected_symbols:
        # Update config
        config['current']['symbols_to_trade'] = selected_symbols
        
        # Set main symbol to first one
        config['current']['symbol'] = selected_symbols[0]
        
        # Enable multi-symbol if more than 1
        config['current']['enable_multi_symbol'] = len(selected_symbols) > 1
        
        save_config(config)
        
        print(f"\n✅ Symbols configured:")
        for sym in selected_symbols:
            print(f"   • {sym}")
        
        print(f"\n   Main symbol: {selected_symbols[0]}")
        print(f"   Multi-symbol: {'YES' if len(selected_symbols) > 1 else 'NO'}")
    
    return config

def menu_44_test_symbol_connection(config: dict) -> None:
    """Test if current symbols are accessible"""
    print("\n🧪 SYMBOL CONNECTION TEST")
    print("="*70)
    
    current_symbol = config['current']['symbol']
    symbols_to_trade = config['current'].get('symbols_to_trade', [current_symbol])
    
    print(f"\nTesting {len(symbols_to_trade)} symbol(s)...")
    
    for symbol in symbols_to_trade:
        print(f"\n📊 Testing: {symbol}")
        
        # Test 1: Symbol info
        info = mt5.symbol_info(symbol)
        if not info:
            print(f"   ❌ Cannot get symbol info")
            
            # Try to find alternative
            if symbol_detector:
                base = symbol.replace('m', '').replace('M', '').replace('.a', '').replace('-m', '')
                alternative = symbol_detector.find_symbol(base)
                if alternative and alternative != symbol:
                    print(f"   💡 Try using: {alternative}")
            continue
        
        print(f"   ✅ Symbol info OK")
        print(f"      Digits: {info.digits}")
        print(f"      Point: {info.point}")
        print(f"      Min lot: {info.volume_min}")
        print(f"      Max lot: {info.volume_max}")
        
        # Test 2: Get tick
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            print(f"   ❌ Cannot get tick data")
            continue
        
        print(f"   ✅ Tick data OK")
        print(f"      Bid: {tick.bid:.{info.digits}f}")
        print(f"      Ask: {tick.ask:.{info.digits}f}")
        print(f"      Spread: {(tick.ask - tick.bid):.{info.digits}f}")
        
        # Test 3: Get rates
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 10)
        if rates is None or len(rates) == 0:
            print(f"   ❌ Cannot get candle data")
            continue
        
        print(f"   ✅ Candle data OK")
        print(f"      Candles available: {len(rates)}")
        
        df = pd.DataFrame(rates)
        print(f"      Last close: {df['close'].iloc[-1]:.{info.digits}f}")
        print(f"      Last volume: {df['tick_volume'].iloc[-1]:.0f}")
        
        # Test 4: Check if trading allowed
        if info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
            print(f"   ✅ Trading allowed")
        else:
            print(f"   ⚠️ Trading restricted")
        
        print(f"\n   🎯 VERDICT: {'ALL TESTS PASSED ✅' if rates is not None else 'FAILED ❌'}")

def deepseek_quick_analysis(deepseek_config: Dict, symbol: str, technical_signal: str, 
                          bullish_count: int, bearish_count: int, patterns: list, current_price: float) -> Optional[str]:
    """Analisis cepat dengan DeepSeek"""
    try:
        if not deepseek_config:
            return None
            
        prompt = f"""
Analisis {symbol}:
- Sinyal Teknikal: {technical_signal} ({bullish_count} bullish, {bearish_count} bearish)
- Pola: {patterns}
- Harga: {current_price:.5f}

Berikan SATU kalimat: BUY/SELL/WAIT dan alasannya.
"""

        headers = {
            "Authorization": f"Bearer {deepseek_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system", 
                    "content": "Anda adalah analis trading profesional. Berikan analisis singkat dan jelas dalam satu kalimat."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 100,
            "temperature": 0.3,
            "stream": False
        }
        
        response = requests.post(
            deepseek_config["base_url"],
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"❌ DeepSeek API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error in DeepSeek quick analysis: {e}")
        return None


def menu_99_start_trading(config: dict, deepseek_config: Dict) -> None:
    """Start automated trading - SAFE with defaults"""
    
    # Ensure all required keys exist with defaults
    defaults = {
        'min_signal_strength': 0.1,
        'enable_scalping': True,
        'enable_pattern_trading': True,
        'enable_breakout_trading': True,
        'max_daily_trades': 50,
        'ignore_economic_calendar': False
    }
    
    for key, default_value in defaults.items():
        if key not in config['current']:
            config['current'][key] = default_value
    
    
    print("\n🚀 Starting Auto Trading...")
    print(f"Symbol: {config['current']['symbol']}")
    print(f"Timeframe: {config['current']['timeframe']}")
    print(f"Mode: {config['current']['trade_mode']}")
    print(f"Lot: {config['current']['lot']}")
    print(f"Min Signal: {config['current']['min_signal_strength']:.1%}")
    print(f"Max Daily Trades: {config['current']['max_daily_trades']}")
    print(f"Check Interval: {config['current']['auto_analyze_interval']} min")
    print(f"\nFeatures:")
    print(f"  Scalping: {'✅' if config['current']['enable_scalping'] else '❌'}")
    print(f"  Patterns: {'✅' if config['current']['enable_pattern_trading'] else '❌'}")
    print(f"  Breakouts: {'✅' if config['current']['enable_breakout_trading'] else '❌'}")
    
    main_symbol = config['current']['symbol']
    trading_symbols = config['current'].get('symbols_to_trade', [])
    
    if main_symbol not in trading_symbols:
        print(f"\n⚠️ Symbol Mismatch Detected!")
        print(f"   Main symbol: {main_symbol}")
        print(f"   Trading symbols: {', '.join(trading_symbols)}")
        
        fix = input("\nFix automatically? (y/n): ").lower()
        if fix == 'y':
            config['current']['symbols_to_trade'] = [main_symbol]
            save_config(config)
            print(f"✅ Fixed: Now using {main_symbol}")
        else:
            print("❌ Please fix symbol configuration first (Menu 2)")
            return
    
    account_info = mt5.account_info()
    if not account_info:
        print("❌ Tidak bisa mendapatkan info akun, trading dibatalkan.")
        print("   Pastikan MT5 terhubung.")
        return
    
    currency = account_info.currency
    print(f"\n✅ Mendeteksi mata uang akun: {currency}")

    try:
        analyzer = MarketAnalyzer(
            news_api_key=env.get('news_api_key'),
            te_key=env.get('trading_economics_key')
        )
        trader = TradeManager(config, currency)
        
        # Pastikan TradingBot menerima deepseek_config sebagai Dict, bukan genai.Client
        bot = TradingBot(config, analyzer, trader, deepseek_config, currency)
        
        print("\n✅ All systems ready!")
        print("🤖 DeepSeek AI: ACTIVE" if deepseek_config else "🤖 DeepSeek AI: INACTIVE")
        print("⚠️ Trading with AGGRESSIVE settings - expect MORE trades!")
        print("\nPress Ctrl+C to stop\n")
        
        bot.start()
        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

# --------------------------
# 5. LOOP UTAMA PROGRAM
# --------------------------
def main():
    config = load_config()
    if not config:
        print("❌ Failed to load config.json")
        print("Creating default config...")
        # Create minimal config
        config = {
            "current": {
                "symbol": "XAUUSDm",
                "timeframe": "M5",
                "candles": 100,
                "account": "DEMO",
                "auto_trade": False,
                "lot": 0.02
            },
            "options": {
                "timeframes": ["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
                "accounts": ["DEMO", "REAL"]
            }
        }
        save_config(config)
    
    if not init_mt5():
        print("❌ MT5 initialization failed")
        print("Please check your MT5 login credentials in .env file")
        return
    
    # INI YANG DIUBAH: init_deepseek() instead of init_gemini()
    global deepseek_config
    deepseek_config = init_deepseek()
    
    account_info = mt5.account_info()
    rate = 1.0
    currency_symbol = "$"
    if account_info:
        currency = account_info.currency
        if currency != "USD":
            # Panggil fungsi get_conversion_rate (yang sudah ditambah di atas)
            rate = get_conversion_rate(currency) 
            if currency == "IDR":
                currency_symbol = "Rp"
            else:
                currency_symbol = currency # Atau simbol lain
        print(f"✅ Mata Uang Akun: {currency} | Rate Konversi USD: {rate:,.2f}")
    else:
        print("⚠️ Gagal mendapat info akun, asumsi USD.")

    # Mapping menu - parameter deepseek_config sudah berupa Dict
    mapping_menu = {
        1: lambda c: menu_1_analyze_now(c, deepseek_config),
        2: menu_2_change_symbol,
        3: menu_3_change_timeframe,
        4: menu_4_change_candles,
        5: menu_5_switch_account,
        6: menu_6_change_trade_mode,
        7: menu_7_launch_trainer,
        8: menu_8_toggle_auto_trade,
        9: menu_9_set_auto_lot,
        10: menu_10_set_auto_slippage,
        11: menu_11_toggle_auto_close_profit,
        
        # --- UBAH LAMBDA UNTUK MENU YANG MEMBUTUHKAN RATE ---
        12: lambda c: menu_12_set_auto_close_settings(c, rate, currency_symbol),
        13: menu_13_toggle_auto_analyze,
        14: menu_14_set_auto_analyze_interval,
        15: menu_15_toggle_bep,
        16: lambda c: menu_16_set_bep_min_profit(c, rate, currency_symbol),
        17: menu_17_set_bep_spread_multiplier, # Tidak perlu rate
        18: menu_18_toggle_stpp_trailing,
        19: lambda c: menu_19_set_step_lock_init(c, rate, currency_symbol),
        20: lambda c: menu_20_set_step_step(c, rate, currency_symbol),
        0: menu_0_quit,
        21: menu_21_set_one_shot,
        22: menu_22_cancel_price_trigger,
        23: menu_23_set_entry_decimals,
        24: menu_24_backtest_custom,
        25: menu_25_backtest_7d,
        26: menu_26_backtest_14d,
        27: menu_27_backtest_30d,
        28: menu_28_backtest_60d,
        29: menu_29_toggle_trade_always_on,
        30: menu_30_change_mode_settings,
        31: menu_31_setup_multi_position,
        32: lambda c: menu_32_stop_loss_settings(c, rate, currency_symbol),
        33: lambda c: menu_33_safety_dashboard(), # Dashboard sudah ambil info sendiri
        34: lambda c: menu_34_balance_optimizer(c, rate, currency_symbol),
        35: lambda c: menu_35_safety_calculator(c, rate, currency_symbol),
        36: lambda c: menu_36_apply_calculator_settings(c, rate, currency_symbol),
        37: lambda c: menu_37_profit_planner(c, rate, currency_symbol),
        38: menu_38_dynamic_trade_limit,
        39: menu_39_signal_strength_adjuster,
        40: lambda c: menu_40_trade_statistics(c),
        41: lambda c: menu_41_market_condition_detector(c),
        42: menu_42_session_based_trading,
        43: menu_43_symbol_auto_setup,
        44: lambda c: menu_44_test_symbol_connection(c),
        99: lambda c: menu_99_start_trading(c, deepseek_config) # DeepSeek version
    }
    
    running = True
    while running:
        try:
            # Validate config before using
            if config is None or 'current' not in config:
                print("\n❌ Config error detected! Reloading...")
                config = load_config()
                if not config:
                    print("❌ Cannot recover config. Exiting...")
                    break
            
            cetak_menu(config, rate, currency_symbol)
            pilihan = pilih_menu()
            
            if pilihan == -1:
                continue
            
            if pilihan in mapping_menu:
                fungsi = mapping_menu[pilihan]
                
                # List menu yang return config
                menu_yang_return_config = [
                    2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,
                    23,29,30,31,32,34,35,36,37,38,39,40,41,42,43,44
                ]
                
                if pilihan in menu_yang_return_config:
                    # Panggil fungsi (lambda akan otomatis teruskan 'rate' dan 'currency_symbol')
                    result = fungsi(config) 
                    
                    if result is not None:
                        config = result
                    else:
                        print("⚠️ Menu returned None, keeping previous config")
                elif pilihan == 0:
                    running = mapping_menu[pilihan]()
                else:
                    # Panggil fungsi (lambda akan otomatis teruskan 'rate' dan 'currency_symbol')
                    fungsi(config)
            
            time.sleep(0.5)
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            mt5.shutdown()
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to recover
            print("\nAttempting to recover...")
            time.sleep(2)
            
            # Reload config
            config = load_config()
            if not config:
                print("❌ Cannot recover. Exiting...")
                break


if __name__ == "__main__":
    main()