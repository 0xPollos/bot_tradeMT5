from typing import Dict, Optional, List
import time
from datetime import datetime
import metatrader5 as mt5
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import yfinance as yf # TAMBAHAN

class TradingBot:
    def __init__(self, config: Dict, analyzer, trader, gemini_client=None, currency: str = 'USD'): # PERUBAHAN: Tambahkan 'currency'
        self.config = config
        self.analyzer = analyzer
        self.trader = trader
        self.gemini_client = gemini_client
        self.running = False
        self.last_analysis_time = 0

        # --- TAMBAHAN DETEKSI MATA UANG ---
        self.currency = currency
        self.rate = self.get_conversion_rate() # Panggil fungsi baru
        print(f"🤖 BotManager initialized for {self.currency} | Rate: {self.rate:.2f}")
        # --- BATAS TAMBAHAN ---

        # Initialize counters
        self.trades_today = 0
        self.last_trade_date = datetime.now().date()

        # Get account info for balance tracking
        account = mt5.account_info()
        if account:
            self.starting_balance = account.balance
            self.starting_equity = account.equity
            # Save to config if not set
            if config.get('current', {}).get('starting_balance', 0) == 0:
                config['current']['starting_balance'] = account.balance
                self._save_config()
        else:
            self.starting_balance = config.get('current', {}).get('starting_balance', 100)
            self.starting_equity = self.starting_balance

        # Loss tracking
        self.daily_loss = 0.0
        self.daily_profit = 0.0

        # Multi-position settings
        cfg = config.get('current', {}) # AMBIL DARI CONFIG
        self.max_daily_trades = cfg.get('max_daily_trades', 100)
        self.max_positions_per_symbol = cfg.get('max_positions_per_symbol', 3)
        self.max_total_positions = cfg.get('max_total_positions', 10)

        # Multi-symbol/timeframe
        self.enable_multi_symbol = cfg.get('enable_multi_symbol', False)
        self.enable_multi_timeframe = cfg.get('enable_multi_timeframe', False)
        self.symbols_to_trade = cfg.get('symbols_to_trade', [config['current']['symbol']])
        self.timeframes_to_check = cfg.get('timeframes_to_check', ['M5'])

        # Rapid fire mode
        self.rapid_fire_mode = cfg.get('rapid_fire_mode', False)

        # Grid trading
        self.grid_trading = cfg.get('grid_trading', False)
        self.grid_distance = cfg.get('grid_distance', 5.0)
        self.max_grid_levels = cfg.get('max_grid_levels', 3)

        # Dynamic lot sizing
        self.dynamic_lot_sizing = cfg.get('dynamic_lot_sizing', False)
        self.risk_percent = cfg.get('risk_percent_per_trade', 1.0)

        # Loss protection settings
        self.stop_loss_protection = cfg.get('stop_loss_protection', True)
        
        # --- KONVERSI NILAI LOSS PROTECTION ---
        # Ambil nilai USD
        usd_max_loss_per_trade = cfg.get('max_loss_per_trade', 1.0)
        usd_global_max_loss = cfg.get('global_max_loss', 5.0)
        usd_daily_loss_limit = cfg.get('daily_loss_limit', 10.0)
        
        # Konversi ke mata uang akun
        self.max_loss_per_trade = usd_max_loss_per_trade * self.rate
        self.global_max_loss = usd_global_max_loss * self.rate
        self.daily_loss_limit = usd_daily_loss_limit * self.rate
        
        self.max_drawdown_percent = cfg.get('max_drawdown_percent', 20.0)
        self.auto_close_on_loss = cfg.get('auto_close_on_loss', True)

        # Cetak untuk konfirmasi
        print(f"   Daily Loss Limit: ${usd_daily_loss_limit:.2f} USD -> {self.daily_loss_limit:.2f} {self.currency}")
        print(f"   Global Max Loss: ${usd_global_max_loss:.2f} USD -> {self.global_max_loss:.2f} {self.currency}")
        # --- BATAS KONVERSI ---

        # Dynamic trade limit
        self.dynamic_trade_limit = cfg.get('dynamic_trade_limit', False)
        self.base_daily_trades = cfg.get('base_daily_trades', 5)
        self.bonus_trades_per_dollar = cfg.get('bonus_trades_per_dollar', 2)
        self.max_daily_trades_cap = cfg.get('max_daily_trades_cap', 50)

        # Check interval
        self.check_interval = cfg.get('auto_analyze_interval', 1) * 60

        if self.rapid_fire_mode:
            self.check_interval = 5  # Check every 10 seconds in rapid fire mode

        # Display initialization info
        self._print_init_info()

        main_symbol = config['current']['symbol']

        # Get symbols_to_trade from config
        symbols_from_config = config.get('current', {}).get('symbols_to_trade', [])

        # If multi-symbol is enabled
        if config.get('current', {}).get('enable_multi_symbol', False):
            # Use symbols_to_trade if available and not empty
            if symbols_from_config and len(symbols_from_config) > 0:
                self.symbols_to_trade = symbols_from_config
            else:
                # Fallback: use main symbol only
                self.symbols_to_trade = [main_symbol]
                print(f"⚠️ No symbols_to_trade found, using main symbol: {main_symbol}")
        else:
            # Single symbol mode: always use main symbol
            self.symbols_to_trade = [main_symbol]
            print(f"📊 Single symbol mode: {main_symbol}")

        # Ensure main symbol is in the list
        if main_symbol not in self.symbols_to_trade:
            self.symbols_to_trade.insert(0, main_symbol)
            print(f"✅ Added main symbol {main_symbol} to trading list")

        # Validate all symbols
        validated_symbols = []
        from main import symbol_detector

        if symbol_detector:
            print(f"\n🔍 Validating symbols...")
            for sym in self.symbols_to_trade:
                # Check if symbol exists
                symbol_info = mt5.symbol_info(sym)
                if symbol_info:
                    validated_symbols.append(sym)
                    print(f"   ✅ {sym}")
                else:
                    # Try to find alternative
                    actual = symbol_detector.find_symbol(sym)
                    if actual:
                        validated_symbols.append(actual)
                        print(f"   🔧 {sym} → {actual}")
                    else:
                        print(f"   ❌ {sym} not found, skipping")

            if validated_symbols:
                self.symbols_to_trade = validated_symbols
            else:
                # No valid symbols, use main symbol
                print(f"⚠️ No valid symbols found, using main: {main_symbol}")
                self.symbols_to_trade = [main_symbol]

        # Update config to reflect actual symbols being used
        config['current']['symbols_to_trade'] = self.symbols_to_trade

        # Log final configuration
        print(f"\n📊 Final Symbol Configuration:")
        print(f"   Main: {main_symbol}")
        print(f"   Trading: {', '.join(self.symbols_to_trade)}")
        print(f"   Count: {len(self.symbols_to_trade)}")

        account = mt5.account_info()
        if account and account.balance < 500:
            print(f"\n⚠️ SAFETY MODE for balance ${account.balance:.2f}")
            print(f"   Disabling multi-timeframe")
            print(f"   Disabling multi-symbol")

            config['current']['enable_multi_timeframe'] = False
            config['current']['enable_multi_symbol'] = False
            config['current']['max_total_positions'] = 2
            config['current']['max_daily_trades'] = 5

            # Force save
            self._save_config()

        # Get timeframes
        self.enable_multi_timeframe = config.get('current', {}).get('enable_multi_timeframe', False)

        if self.enable_multi_timeframe:
            self.timeframes_to_check = config.get('current', {}).get('timeframes_to_check', ['M5'])
        else:
            # ALWAYS use configured timeframe when multi-TF is OFF
            main_tf = config['current']['timeframe']
            self.timeframes_to_check = [main_tf]
            print(f"📊 Single timeframe mode: {main_tf}")

    # --- TAMBAHAN FUNGSI BARU ---
    def get_conversion_rate(self) -> float:
        """Mendapatkan rate konversi USD ke mata uang akun."""
        if self.currency == "USD":
            return 1.0

        # Cth: USDIDR
        symbol_name = f"USD{self.currency}" 

        try:
            # 1. Coba ambil dari MT5 dulu
            if mt5.symbol_select(symbol_name, True):
                tick = mt5.symbol_info_tick(symbol_name)
                if tick and tick.ask > 0:
                    print(f"   Rate from MT5 ({symbol_name}): {tick.ask}")
                    return tick.ask
        except:
            pass # Lanjut ke yfinance

        # 2. Coba yfinance (butuh koneksi internet)
        pair = f"USD{self.currency}=X" # Cth: USDIDR=X
        try:
            data = yf.Ticker(pair).history(period="1d")
            if not data.empty:
                rate = data['Close'].iloc[-1]
                print(f"   Rate from yfinance ({pair}): {rate:.2f}")
                return rate
        except Exception as e:
            print(f"   yfinance error (perlu internet): {e}")

        # 3. Fallback jika semua gagal
        fallback_rate = 16000.0 if self.currency == "IDR" else 1.0
        print(f"⚠️ Gagal ambil rate, pakai fallback: {fallback_rate}")
        return fallback_rate
    # --- BATAS FUNGSI BARU ---

    def _print_init_info(self) -> None:
        """Print bot initialization information"""
        # Get TP mode info
        tp_mode = self.config.get('current', {}).get('auto_close_mode', 'PER_TRADE')
        per_trade = self.config.get('current', {}).get('auto_close_target', 0.4)
        total = self.config.get('current', {}).get('auto_close_total_target', 5.0)
        close_all = self.config.get('current', {}).get('close_all_on_target', False)

        print(f"\n{'='*70}")
        print(f"🤖 BOT INITIALIZED - RAPID FIRE MODE")
        print(f"{'='*70}")
        print(f"\n💰 ACCOUNT:")
        print(f"   Starting Balance: ${self.starting_balance:.2f}")
        print(f"   Starting Equity: ${self.starting_equity:.2f}")
        print(f"   Account Currency: {self.currency} (USD Rate: {self.rate:.2f})") # TAMBAHAN

        print(f"\n📊 POSITION LIMITS:")
        print(f"   Max positions per symbol: {self.max_positions_per_symbol}")
        print(f"   Max total positions: {self.max_total_positions}")
        print(f"   Max daily trades: {self.max_daily_trades}")

        print(f"\n🎯 AUTO TAKE PROFIT:")
        if tp_mode == 'PER_TRADE':
            print(f"   Mode: PER TRADE")
            print(f"   Target: ${per_trade:.2f} per position")
        elif tp_mode == 'TOTAL':
            print(f"   Mode: TOTAL")
            print(f"   Target: ${total:.2f} all positions combined")
        else:  # BOTH
            print(f"   Mode: BOTH")
            print(f"   Targets: ${per_trade:.2f}/trade OR ${total:.2f} total")

        if close_all:
            print(f"   Close All: ✅ (All positions close when ANY hits target)")

        print(f"\n🛡️ STOP LOSS PROTECTION:")
        print(f"   Protection: {'ON' if self.stop_loss_protection else 'OFF'}")
        if self.stop_loss_protection:
            print(f"   Max loss/trade: {self.max_loss_per_trade:.2f} {self.currency}") # PERUBAHAN
            print(f"   Global max loss: {self.global_max_loss:.2f} {self.currency}") # PERUBAHAN
            print(f"   Daily loss limit: {self.daily_loss_limit:.2f} {self.currency}") # PERUBAHAN
            print(f"   Max drawdown: {self.max_drawdown_percent:.1f}%")
            print(f"   Auto close on loss: {'YES' if self.auto_close_on_loss else 'NO'}")

        print(f"\n⚡ TRADING SETTINGS:")
        print(f"   Check interval: {self.check_interval}s")
        print(f"   Multi-symbol: {'✅' if self.enable_multi_symbol else '❌'} ({len(self.symbols_to_trade)} symbols)")
        print(f"   Multi-timeframe: {'✅' if self.enable_multi_timeframe else '❌'} ({len(self.timeframes_to_check)} TFs)")
        print(f"   Dynamic lot sizing: {'✅' if self.dynamic_lot_sizing else '❌'}")
        print(f"   Rapid fire: {'✅' if self.rapid_fire_mode else '❌'}")
        print(f"   Grid trading: {'✅' if self.grid_trading else '❌'}")
        print(f"{'='*70}\n")

    def start(self) -> None:
        """Start the trading bot"""
        self.running = True

        print(f"\n🚀 RAPID FIRE Bot Started!")
        print(f"⚡ Symbols: {', '.join(self.symbols_to_trade)}")
        print(f"⏱️ Checking every {self.check_interval}s")
        print(f"🛑 MAX {self.max_total_positions} POSITIONS - Will STOP when full!")
        print("Press Ctrl+C to stop\n")

        while self.running:
            # --- TAMBAHKAN TRY...EXCEPT DI LUAR SIKLUS UTAMA ---
            try: 
                # Print penanda awal siklus
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"\n[{timestamp}] --- Memulai Siklus ---") 

                # Reset daily counter if new day
                self._check_new_day()
                
                # Check emergency stop
                if self.config['current'].get('emergency_stop_active', False):
                    print("🚨 EMERGENCY STOP ACTIVE - Trading disabled!")
                    self.running = False
                    break
                
                # --- TAMBAHKAN TRY...EXCEPT DI SEKITAR MANAGE & RUN ---
                try:
                    # 1. Selalu manage posisi dulu
                    print(f"[{timestamp}] Memeriksa posisi terbuka...")
                    self.trader.manage_open_positions()
                    print(f"[{timestamp}] Selesai memeriksa posisi.")

                    # 2. Jalankan siklus trading (Rapid Fire atau Normal)
                    print(f"[{timestamp}] Menjalankan siklus trading...")
                    if self.rapid_fire_mode:
                        self._run_rapid_fire_cycle()
                    else:
                        self._run_cycle() # Jika kamu punya mode normal
                    print(f"[{timestamp}] Selesai siklus trading.")

                except mt5.MT5Exception as mt5_err:
                     print(f"❌ Error MT5 di dalam siklus: {mt5_err}")
                     print(f"   Kode Error: {mt5.last_error()}")
                     print("   Mencoba melanjutkan...")
                     time.sleep(5) # Beri jeda jika ada error MT5
                except Exception as cycle_err:
                    print(f"❌ Error tidak terduga di dalam siklus: {cycle_err}")
                    import traceback
                    traceback.print_exc()
                    print("   Mencoba melanjutkan...")
                    time.sleep(5) # Beri jeda
                # --- BATAS TRY...EXCEPT DALAM ---

                # Wait before next cycle
                print(f"[{timestamp}] --- Akhir Siklus, Jeda {self.check_interval}s ---")
                time.sleep(self.check_interval)
            
            # Tangkap KeyboardInterrupt untuk stop manual
            except KeyboardInterrupt:
                print("\n⏹️ Stopping bot...")
                self.stop()
                break
                
            # Tangkap SEMUA error lain di level terluar
            except Exception as outer_err:
                print(f"❌❌ ERROR KRITIS DI LUAR SIKLUS: {outer_err}")
                import traceback
                traceback.print_exc()
                print("   Bot mungkin berhenti. Silakan restart.")
                self.running = False # Hentikan bot jika error parah
                break
            # --- BATAS TRY...EXCEPT LUAR ---

    def stop(self) -> None:
        """Stop the trading bot"""
        self.running = False

        # Get final stats
        positions = self._get_current_positions()
        open_count = len(positions)
        total_profit = sum(p.profit for p in positions)

        account = mt5.account_info()
        if account:
            current_balance = account.balance
            profit_from_start = current_balance - self.starting_balance
        else:
            profit_from_start = 0

        print(f"\n{'='*70}")
        print(f"🛑 BOT STOPPED")
        print(f"{'='*70}")
        print(f"\n📊 SESSION SUMMARY:")
        print(f"   Starting balance: {self.starting_balance:.2f} {self.currency}") # PERUBAHAN
        if account:
            print(f"   Current balance: {account.balance:.2f} {self.currency}") # PERUBAHAN
        print(f"   Realized P/L: {profit_from_start:+.2f} {self.currency}") # PERUBAHAN
        print(f"   Trades executed: {self.trades_today}/{self.max_daily_trades}")
        print(f"   Open positions: {open_count}")
        print(f"   Floating P/L: {total_profit:+.2f} {self.currency}") # PERUBAHAN
        print(f"   Total P/L: {profit_from_start + total_profit:+.2f} {self.currency}") # PERUBAHAN
        print(f"{'='*70}\n")

    def _check_new_day(self) -> None:
        """Reset counter if new trading day"""
        current_date = datetime.now().date()

        if current_date > self.last_trade_date:
            print(f"\n{'='*70}")
            print(f"📅 NEW TRADING DAY - {current_date}")
            print(f"{'='*70}")
            print(f"   Yesterday's trades: {self.trades_today}")

            # Get yesterday's P/L
            account = mt5.account_info()
            if account:
                yesterday_pl = account.balance - self.starting_balance
                print(f"   Yesterday's P/L: {yesterday_pl:+.2f} {self.currency}") # PERUBAHAN

                # Reset starting balance for new day
                self.starting_balance = account.balance
                self.starting_equity = account.equity

                # Save to config
                self.config['current']['starting_balance'] = account.balance
                self._save_config()

            # Reset counters
            self.trades_today = 0
            self.last_trade_date = current_date
            print(f"{'='*70}\n")

    def _run_rapid_fire_cycle(self) -> None:
        """Rapid fire mode - with STRICT LIMITS"""
        try:
            # --- TAMBAHKAN INI ---
            # 0. CEK LIMIT KERUGIAN SEBELUM MELAKUKAN APAPUN
            if self.stop_loss_protection:
                if not self._check_loss_limits():
                    # Jika limit terlampaui, stop cycle ini
                    return 
            # --- BATAS TAMBAHAN ---

            # 1. Manage positions
            self.trader.manage_open_positions()
            
            # 2. Get current positions
            current_positions = self._get_current_positions()
            total_open = len(current_positions)

            # 3. CHECK DAILY TRADE LIMIT (STRICT)
            current_limit = self._get_current_trade_limit()

            if self.trades_today >= current_limit:
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp}] ⏸️ DAILY LIMIT REACHED ({self.trades_today}/{current_limit})")
                print(f"           No more new trades today")

                # Just manage existing positions
                return

            # 4. CHECK MAX POSITIONS
            if total_open >= self.max_total_positions:
                timestamp = datetime.now().strftime('%H:%M:%S')
                total_profit = sum(p.profit for p in current_positions)

                print(f"[{timestamp}] ⚠️ MAX POSITIONS ({total_open}/{self.max_total_positions}) | P/L: {total_profit:+.2f} {self.currency}") # PERUBAHAN

                return  # DON'T OPEN NEW

            # 5. Calculate remaining slots
            remaining_trades = current_limit - self.trades_today
            remaining_positions = self.max_total_positions - total_open

            remaining_slots = min(remaining_trades, remaining_positions)

            if remaining_slots <= 0:
                return

            # 6. Analyze
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"\n[{timestamp}] Analyzing {len(self.symbols_to_trade)} symbol(s) on {self.timeframes_to_check}")
            print(f"           Positions: {total_open}/{self.max_total_positions} | Trades: {self.trades_today}/{current_limit} | Slots: {remaining_slots}")

            signals = []

            # Use ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}

                for symbol in self.symbols_to_trade:
                    # Check per-symbol limit
                    symbol_positions = len([p for p in current_positions if p.symbol == symbol])
                    if symbol_positions >= self.max_positions_per_symbol:
                        continue

                    # Submit analysis for each timeframe
                    for tf in self.timeframes_to_check:
                        future = executor.submit(self._analyze_symbol_timeframe, symbol, tf)
                        futures[future] = (symbol, tf)

                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result and result['signal'] != 'WAIT':
                            signals.append(result)
                    except Exception as e:
                        pass

# 7. Execute signals (LIMIT to remaining_slots)
            if signals:
                print(f"           Found {len(signals)} initial signal(s)")

                # --- PERBAIKAN: PENCEGAHAN HEDGING ---
                filtered_signals = []
                symbols_processed = set()

                # 1. Urutkan berdasarkan sinyal TERKUAT
                signals.sort(key=lambda x: x['strength'], reverse=True)

                for s in signals:
                    symbol = s['symbol']
                    # 2. Cek apakah kita sudah memproses simbol ini
                    if symbol not in symbols_processed:
                        # Jika belum, ini adalah sinyal TERKUAT (karena sudah diurutkan)
                        
                        # 3. Cek apakah ada sinyal KONFLIK (lawan arah) untuk simbol ini
                        found_conflict = False # Flag untuk menandai konflik
                        for other_s in signals:
                            if other_s['symbol'] == symbol and other_s['signal'] != s['signal']:
                                # Jika ada konflik (BUY vs SELL)
                                print(f"           ⚠️ Konflik terdeteksi untuk {symbol}:")
                                print(f"               -> ✅ Diambil: {s['signal']} ({s['strength']:.0%}) dari {s['timeframe']}")
                                print(f"               -> ❌ Dibuang: {other_s['signal']} ({other_s['strength']:.0%}) dari {other_s['timeframe']}")
                                found_conflict = True # Tandai ada konflik
                                break # Cukup cek 1 konflik
                        
                        # 4. Tambahkan sinyal terkuat ke list final
                        filtered_signals.append(s)
                        # 5. Tandai simbol ini sudah selesai diproses
                        symbols_processed.add(symbol)
                
                # Hanya print jika ada sinyal yang dibuang
                if len(filtered_signals) < len(signals):
                    print(f"           Memfilter {len(signals) - len(filtered_signals)} sinyal konflik.")
                # --- BATAS PERBAIKAN ---

                
                # STRICT LIMIT (gunakan 'filtered_signals' yang sudah bersih)
                signals_to_execute = filtered_signals[:remaining_slots]
                
                # Cek lagi apakah ada sinyal setelah difilter
                if not signals_to_execute:
                     print(f"           Tidak ada sinyal valid setelah filter.")
                else:
                    executed = 0
                    for signal_data in signals_to_execute:
                        # Double check limit before each execution
                        if self.trades_today >= current_limit:
                            print(f"\n⏸️ Stopping execution - daily limit reached")
                            break
                        
                        current_pos_count = len(self._get_current_positions()) # Cek jumlah posisi saat ini
                        if current_pos_count >= self.max_total_positions:
                            print(f"\n⏸️ Stopping execution - max positions reached ({current_pos_count}/{self.max_total_positions})")
                            break
                        
                        if self._execute_signal(signal_data):
                            executed += 1
                            time.sleep(1)  # Delay between orders
                    
                    if executed > 0:
                        print(f"\n           ✅ Executed {executed}/{len(signals_to_execute)} signal(s)")
            else:
                print(f"           No signals found")


        except Exception as e:
            print(f"❌ Cycle error: {e}")
            import traceback
            traceback.print_exc()

    def _check_loss_limits(self) -> bool:
        """Check all loss limits - return False if any breached"""
        try:
            account = mt5.account_info()
            if not account:
                return True

            # 1. CHECK DAILY LOSS LIMIT
            daily_pl = account.balance - self.starting_balance

            # Gunakan self.daily_loss_limit (yang sudah dikonversi)
            if daily_pl < 0 and abs(daily_pl) >= self.daily_loss_limit:
                # Only print warning once
                if not hasattr(self, '_daily_limit_warned'):
                    self._daily_limit_warned = True

                    print(f"\n{'='*70}")
                    print(f"🚨 DAILY LOSS LIMIT REACHED!")
                    print(f"   Today's Loss: {daily_pl:.2f} {self.currency}") # PERUBAHAN
                    print(f"   Limit: {self.daily_loss_limit:.2f} {self.currency}") # PERUBAHAN
                    print(f"   Trading STOPPED for today!")
                    print(f"{'='*70}")

                    self.config['current']['auto_trade'] = False
                    self._save_config()

                    if self.auto_close_on_loss:
                        positions = self._get_current_positions()
                        if positions:
                            self._emergency_close_all(positions, "Daily loss limit")

                return False

            # 2. CHECK MAX DRAWDOWN
            drawdown = ((self.starting_equity - account.equity) / self.starting_equity) * 100 if self.starting_equity > 0 else 0

            if drawdown >= self.max_drawdown_percent:
                # Only print warning once
                if not hasattr(self, '_drawdown_warned'):
                    self._drawdown_warned = True

                    print(f"\n{'='*70}")
                    print(f"🚨 MAX DRAWDOWN REACHED!")
                    print(f"   Starting Equity: {self.starting_equity:.2f} {self.currency}") # PERUBAHAN
                    print(f"   Current Equity: {account.equity:.2f} {self.currency}") # PERUBAHAN
                    print(f"   Drawdown: {drawdown:.1f}% (Limit: {self.max_drawdown_percent:.1f}%)")
                    print(f"   Trading STOPPED!")
                    print(f"{'='*70}")
                    print(f"\n💡 RECOMMENDATIONS:")
                    print(f"   1. Use Menu 34 to optimize settings for your balance")
                    print(f"   2. Reduce position size")
                    print(f"   3. Increase max_drawdown_percent to {drawdown + 10:.0f}% (Menu 32)")
                    print(f"   4. Or deposit more funds")
                    print(f"{'='*70}")

                    self.config['current']['auto_trade'] = False
                    self._save_config()

                    if self.auto_close_on_loss:
                        positions = self._get_current_positions()
                        if positions:
                            self._emergency_close_all(positions, "Max drawdown")

                return False

            # Reset warning flags if recovered
            if daily_pl >= 0:
                self._daily_limit_warned = False
            if drawdown < self.max_drawdown_percent * 0.8:
                self._drawdown_warned = False

            # 3. CHECK INDIVIDUAL POSITION LOSSES (Jaring Pengaman "Software SL")
            positions = self._get_current_positions()
            timestamp = datetime.now().strftime('%H:%M:%S')

            for p in positions:
                # Cek jika profit negatif DAN kerugiannya melebihi batas
                if p.profit < 0 and abs(p.profit) >= self.max_loss_per_trade:

                    # --- TAMBAHKAN LOG DI SINI ---
                    print(f"\n{'='*70}")
                    print(f"[{timestamp}] 🛑 SOFTWARE STOP LOSS HIT! (Jaring Pengaman)")
                    print(f"   {p.symbol} Ticket #{p.ticket}")
                    print(f"   Loss: {p.profit:.2f} {self.currency} (Batas SL: {self.max_loss_per_trade:.2f} {self.currency})")

                    # Close position
                    if self.trader.close_position(p.ticket):
                        print(f"   ✅ Position closed successfully!")
                    else:
                        print(f"   ❌ Gagal menutup posisi SL")
                    print(f"{'='*70}")
                    # --- BATAS TAMBAHAN ---

                    time.sleep(0.5) # Beri jedax    

            return True

        except Exception as e:
            print(f"Loss limit check error: {e}")
            return True

    def _emergency_close_all(self, positions: List, reason: str) -> None:
        """Emergency close all positions"""
        print(f"\n{'='*70}")
        print(f"🚨 EMERGENCY CLOSE ALL")
        print(f"   Reason: {reason}")
        print(f"   Closing {len(positions)} position(s)...")
        print(f"{'='*70}")

        closed_count = 0
        total_pl = 0.0

        for p in positions:
            print(f"\n   Closing {p.symbol} #{p.ticket} ({p.profit:+.2f} {self.currency})") # PERUBAHAN

            if self.trader.close_position(p.ticket):
                closed_count += 1
                total_pl += p.profit
                time.sleep(0.5)

        print(f"\n✅ Closed {closed_count}/{len(positions)} positions")
        print(f"   Total P/L from closed positions: {total_pl:+.2f} {self.currency}") # PERUBAHAN
        print(f"{'='*70}")

        # Stop trading
        self.config['current']['auto_trade'] = False
        self._save_config()

    def _analyze_symbol_timeframe(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Analyze a single symbol/timeframe combination"""
        try:
            # VERIFY we're using correct timeframe
            print(f"   🔍 {symbol} {timeframe}", end='')

            # Get data
            df = self._get_market_data_for(symbol, timeframe)
            if df.empty:
                print(" ❌ No data")
                return None

            # Analyze
            analysis = self.analyzer.analyze_market(df, symbol, self.config)

            signal = analysis['overall']['signal']
            strength = analysis['overall']['strength']

            if signal != 'WAIT':
                print(f" → {signal} ({strength:.0%})")
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'signal': signal,
                    'strength': strength,
                    'analysis': analysis
                }
            else:
                print(f" → WAIT")

            return None

        except Exception as e:
            print(f" ❌ Error: {e}")
            return None

    def _execute_signal(self, signal_data: Dict) -> bool:
        """Execute a trading signal with VALIDATION"""
        try:
            symbol = signal_data['symbol']
            action = signal_data['signal']
            strength = signal_data['strength']
            tf = signal_data['timeframe']

            # VALIDATE before execution
            account = mt5.account_info()
            if not account:
                print(f"❌ Cannot get account info")
                return False

            # Check if we have enough margin
            if account.margin_free < 10:
                print(f"❌ Insufficient margin: {account.margin_free:.2f} {self.currency}") # PERUBAHAN
                return False

            print(f"\n{'='*60}")
            print(f"💰 EXECUTING {action} - {symbol} ({tf})")
            print(f"   Strength: {strength:.0%}")

            # Show key analysis
            analysis = signal_data.get('analysis', {})
            if analysis.get('patterns', {}).get('count', 0) > 0:
                patterns = analysis['patterns']['patterns'][:2]
                print(f"   🔥 {', '.join(patterns)}")

            # Calculate lot size (FIXED - use config)
            lot_size = self._calculate_lot_size(symbol)

            # VALIDATE lot size
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                if lot_size < symbol_info.volume_min or lot_size > symbol_info.volume_max:
                    print(f"❌ Invalid lot size: {lot_size} (min: {symbol_info.volume_min}, max: {symbol_info.volume_max})")
                    return False

            # Show what we're about to do
            print(f"\n   📋 Preparing order:")
            print(f"      Symbol: {symbol}")
            print(f"      Action: {action}")
            print(f"      Lot: {lot_size} (from config: {self.config['current']['lot']})")

            # Execute
            result = self.trader.place_order({
                'symbol': symbol,
                'action': action,
                'strength': strength,
                'lot_size': lot_size
            })

            if result['success']:
                self.trades_today += 1

                print(f"✅ SUCCESS! Ticket: #{result.get('ticket')}")
                print(f"   Entry: {result.get('price'):.5f}")
                print(f"   SL: {result.get('sl'):.5f} | TP: {result.get('tp'):.5f}")
                print(f"   Lot: {lot_size}")
                print(f"   Risk: {result.get('expected_risk', 0):.2f} {self.currency}") # PERUBAHAN
                print(f"   Trades: {self.trades_today}/{self._get_current_trade_limit()}")
                print(f"{'='*60}")

                return True
            else:
                print(f"❌ FAILED: {result.get('error', 'Unknown')}")
                print(f"{'='*60}")
                return False

        except Exception as e:
            print(f"❌ Execute error: {e}")
            return False

    def _calculate_lot_size(self, symbol: str) -> float:
        """Calculate lot size - ALWAYS USE CONFIG VALUE"""

        # ALWAYS use configured lot size (disable dynamic)
        configured_lot = self.config['current']['lot']

        # Safety check: ensure lot is not too large for balance
        account = mt5.account_info()
        if account:
            balance = account.balance

            # For micro accounts (<$500), cap at 0.01
            if balance < 500 / self.rate and configured_lot > 0.01: # PERUBAHAN: Konversi nilai ambang batas USD
                print(f"\n⚠️ Lot size reduced for safety:")
                print(f"   Configured: {configured_lot}")
                print(f"   Using: 0.01 (balance too low)")
                return 0.01

            # For small accounts (<$1000), cap at 0.02
            elif balance < 1000 / self.rate and configured_lot > 0.02: # PERUBAHAN: Konversi nilai ambang batas USD
                print(f"\n⚠️ Lot size reduced for safety:")
                print(f"   Configured: {configured_lot}")
                print(f"   Using: 0.02 (balance too low)")
                return 0.02

        # Ensure lot meets broker minimum
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max

            # Ensure within broker limits
            safe_lot = max(min_lot, min(configured_lot, max_lot))

            if safe_lot != configured_lot:
                print(f"\n⚠️ Lot adjusted to broker limits:")
                print(f"   Configured: {configured_lot}")
                print(f"   Broker min: {min_lot}, max: {max_lot}")
                print(f"   Using: {safe_lot}")

            return safe_lot

        return configured_lot

    def _check_grid_opportunities(self) -> None:
        """Check for grid trading opportunities"""
        if not self.grid_trading:
            return

        try:
            positions = self._get_current_positions()

            for symbol in self.symbols_to_trade:
                symbol_positions = [p for p in positions if p.symbol == symbol]

                if len(symbol_positions) == 0:
                    continue

                if len(symbol_positions) >= self.max_grid_levels:
                    continue

                # Get current price
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    continue

                current_price = tick.bid

                # Check if we should add grid level
                for pos in symbol_positions:
                    distance = abs(current_price - pos.price_open)

                    # If price moved enough, add grid level
                    if distance >= self.grid_distance:
                        # Same direction as original
                        action = 'BUY' if pos.type == 0 else 'SELL'

                        print(f"\n🎯 GRID LEVEL TRIGGERED - {symbol}")
                        print(f"   Distance: {distance:.2f} (threshold: {self.grid_distance})") # PERUBAHAN: Hapus simbol $

                        self._execute_signal({
                            'symbol': symbol,
                            'timeframe': self.config['current']['timeframe'],
                            'signal': action,
                            'strength': 0.5,
                            'analysis': {}
                        })

                        break  # Only one grid level per check

        except Exception as e:
            print(f"Grid check error: {e}")

    def _run_cycle(self) -> None:
        """Standard trading cycle (single symbol)"""
        try:
            # Check loss limits
            if self.stop_loss_protection:
                if not self._check_loss_limits():
                    return

            # Manage positions
            self.trader.manage_open_positions()

            # Check limits
            current_positions = self._get_current_positions()
            total_open = len(current_positions)

            if total_open >= self.max_total_positions:
                print(f"⚠️ Max total positions ({total_open}/{self.max_total_positions})")
                return

            symbol_positions = len([p for p in current_positions if p.symbol == self.config['current']['symbol']])

            if symbol_positions >= self.max_positions_per_symbol:
                print(f"⚠️ Max positions for {self.config['current']['symbol']} ({symbol_positions}/{self.max_positions_per_symbol})")
                return

            if self.trades_today >= self.max_daily_trades:
                return

            # Get data
            df = self._get_market_data()
            if df.empty:
                return

            # Analyze
            analysis = self.analyzer.analyze_market(
                df,
                self.config['current']['symbol'],
                self.config
            )

            signal = analysis['overall']['signal']
            strength = analysis['overall']['strength']

            # Log
            timestamp = datetime.now().strftime('%H:%M:%S')
            total_profit = sum(p.profit for p in current_positions)
            print(f"\n[{timestamp}] {signal} ({strength:.0%}) | Positions: {total_open}/{self.max_total_positions} | P/L: {total_profit:+.2f} {self.currency}") # PERUBAHAN

            # Execute
            if self._should_trade() and signal != 'WAIT':
                self._execute_signal({
                    'symbol': self.config['current']['symbol'],
                    'timeframe': self.config['current']['timeframe'],
                    'signal': signal,
                    'strength': strength,
                    'analysis': analysis
                })

        except Exception as e:
            print(f"❌ Cycle error: {e}")

    def _get_current_positions(self) -> List:
        """Get all current bot positions"""
        positions = mt5.positions_get()
        if not positions:
            return []

        # Filter only our bot's positions
        return [p for p in positions if p.magic == 234000]

    def _should_trade(self) -> bool:
        """Check if trading is allowed"""
        if not self.config.get('current', {}).get('auto_trade', False):
            return False

        account = mt5.account_info()
        if account and account.balance < 5:
            return False

        if self.config.get('current', {}).get('trade_always_on', True):
            return True

        now = datetime.now()
        return 1 <= now.hour < 23

    def _get_market_data(self) -> pd.DataFrame:
        """Get market data for current symbol"""
        return self._get_market_data_for(
            self.config['current']['symbol'],
            self.config['current']['timeframe']
        )

    def _get_market_data_for(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get market data for specific symbol/timeframe"""
        try:
            tf_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }

            mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
            candles = self.config['current']['candles']

            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, candles)

            if rates is None or len(rates) == 0:
                return pd.DataFrame()

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            return df

        except Exception as e:
            return pd.DataFrame()

    def _save_config(self) -> None:
        """Save config to file"""
        try:
            with open('config.json', 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Config save error: {e}")

    def get_statistics(self) -> Dict:
        """Get bot statistics"""
        positions = self._get_current_positions()

        # Group by symbol
        symbol_counts = {}
        for p in positions:
            symbol_counts[p.symbol] = symbol_counts.get(p.symbol, 0) + 1

        total_profit = sum(p.profit for p in positions)

        account = mt5.account_info()
        realized_pl = account.balance - self.starting_balance if account else 0

        return {
            'trades_today': self.trades_today,
            'max_daily_trades': self.max_daily_trades,
            'remaining_trades': self._get_current_trade_limit() - self.trades_today, # PERUBAHAN: Gunakan _get_current_trade_limit()
            'open_positions': len(positions),
            'max_total_positions': self.max_total_positions,
            'positions_by_symbol': symbol_counts,
            'floating_pl': total_profit,
            'realized_pl': realized_pl,
            'total_pl': realized_pl + total_profit,
            'is_running': self.running,
            'trade_date': self.last_trade_date.strftime('%Y-%m-%d'),
            'starting_balance': self.starting_balance,
            'currency': self.currency # TAMBAHAN
        }

    def _get_current_trade_limit(self) -> int:
        """Get current trade limit (dynamic if enabled)"""
        if not self.dynamic_trade_limit:
            return self.max_daily_trades

        # Calculate today's profit
        account = mt5.account_info()
        if not account:
            return self.base_daily_trades

        today_profit = account.balance - self.starting_balance

        if today_profit <= 0:
            return self.base_daily_trades

        # Calculate bonus trades (profit dikonversi ke USD untuk perhitungan bonus)
        # Asumsi bonus_trades_per_dollar adalah per USD
        today_profit_usd = today_profit / self.rate # PERUBAHAN: Konversi ke USD
        
        bonus_trades = int(today_profit_usd * self.bonus_trades_per_dollar) # PERUBAHAN: Gunakan profit USD

        # Total trades with cap
        total_trades = min(
            self.base_daily_trades + bonus_trades,
            self.max_daily_trades_cap
        )

        return total_trades
