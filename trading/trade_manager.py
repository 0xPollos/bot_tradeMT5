# trading/trade_manager.py - COMPLETE TRADE MANAGER WITH ALL PROTECTIONS
from typing import Dict, Optional
import metatrader5 as mt5
from datetime import datetime
import pandas as pd
import numpy as np
import time
import yfinance as yf

class TradeManager:
    def __init__(self, config: Dict, currency: str): # Tambahkan 'currency'
        self.config = config
        self.positions = {}
        
        self.currency = currency
        self.rate = self.get_conversion_rate() # Panggil fungsi baru
        print(f"📋 TradeManager initialized for {self.currency} | Rate: {self.rate:.2f}")

        cfg = config.get('current', {})
        usd_max_loss_per_trade = cfg.get('max_loss_per_trade', 1.0)
        
        self.stop_loss_protection = cfg.get('stop_loss_protection', True)
        self.max_loss_per_trade = usd_max_loss_per_trade * self.rate 
        self.auto_close_on_loss = cfg.get('auto_close_on_loss', True)
        
        print(f"   Max Loss per Trade: ${usd_max_loss_per_trade:.2f} USD -> {self.max_loss_per_trade:.2f} {self.currency}")

    def get_conversion_rate(self) -> float:
        """Mendapatkan rate konversi USD ke mata uang akun."""
        if self.currency == "USD":
            return 1.0
        
        pair = f"USD{self.currency}=X" # Cth: USDIDR=X
        
        try:
            # Coba ambil dari MT5 dulu (jika ada, cth: USDIDR)
            symbol_name = f"USD{self.currency}"
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
        fallback_rate = 16000.0 if self.currency == "IDR" else 1.0
        print(f"⚠️ Gagal ambil rate, pakai fallback: {fallback_rate}")
        return fallback_rate

    def place_order(self, signal: Dict) -> Dict:
        """Place order with comprehensive risk management"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            
            # Use custom lot size if provided, otherwise use config
            lot_size = signal.get('lot_size', self.config['current']['lot'])
            
            if action not in ['BUY', 'SELL']:
                return {'success': False, 'error': 'Invalid action'}
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return {'success': False, 'error': 'Symbol info not available'}
            
            # Check if trading is allowed for this symbol
            if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
                return {'success': False, 'error': 'Trading not allowed for this symbol'}
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return {'success': False, 'error': 'Could not get price'}
            
            # Calculate entry price
            price = tick.ask if action == 'BUY' else tick.bid
            
            # Get symbol parameters
            point = symbol_info.point
            digits = symbol_info.digits
            stops_level = symbol_info.trade_stops_level
            
            # Get SL/TP multipliers from config
            sl_multiplier = self.config['current'].get('sl_multiplier', 1.5)
            
            # Calculate ATR for dynamic SL/TP
            atr = self._calculate_atr(symbol)
            
            # Calculate SL/TP based on symbol type
            if 'XAU' in symbol or 'GOLD' in symbol:
                # GOLD - Dollar-based calculation
                sl_distance, tp_distance = self._calculate_gold_sl_tp(
                    atr, sl_multiplier, lot_size, price
                )
            elif 'BTC' in symbol or 'CRYPTO' in symbol:
                # CRYPTO - Larger movements
                sl_distance, tp_distance = self._calculate_crypto_sl_tp(
                    atr, sl_multiplier, lot_size
                )
            else:
                # FOREX - Pip-based calculation
                sl_distance, tp_distance = self._calculate_forex_sl_tp(
                    atr, sl_multiplier, point, lot_size
                )
            
            # Ensure minimum distance from broker requirements
            min_distance = stops_level * point
            if min_distance > 0:
                sl_distance = max(sl_distance, min_distance * 2)
                tp_distance = max(tp_distance, min_distance * 2)
            
            # Calculate actual SL and TP prices
            if action == 'BUY':
                sl = round(price - sl_distance, digits)
                tp = round(price + tp_distance, digits)
            else:  # SELL
                sl = round(price + sl_distance, digits)
                tp = round(price - tp_distance, digits)
            
            # Validate SL/TP positioning
            if action == 'BUY':
                if sl >= price or tp <= price:
                    return {'success': False, 'error': f'Invalid SL/TP: Price={price}, SL={sl}, TP={tp}'}
            else:  # SELL
                if sl <= price or tp >= price:
                    return {'success': False, 'error': f'Invalid SL/TP: Price={price}, SL={sl}, TP={tp}'}
            
            # Calculate expected risk in USD
            divisor_usd = self._get_divisor_usd(symbol, lot_size)
            divisor_currency = divisor_usd * self.rate
            expected_loss = sl_distance * divisor_currency

            # Cek lot_size (sudah di-handle di bot_manager, tapi cek lagi)
            if lot_size < symbol_info.volume_min or lot_size > symbol_info.volume_max:
                print(f"❌ Lot size {lot_size} di luar batas broker ({symbol_info.volume_min}-{symbol_info.volume_max}). Membatalkan.")
                return {'success': False, 'error': f'Lot size {lot_size} di luar batas broker'}
            
            # Determine filling type
            filling_type = self._get_filling_type(symbol_info)
            
            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": self.config['current']['slippage'],
                "magic": 234000,
                "comment": f"Bot {signal.get('strength', 0):.0%}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_type,
            }
            
            # Calculate risk-reward ratio
            risk = abs(price - sl)
            reward = abs(tp - price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Log the request for debugging
            print(f"\n📋 Order Request:")
            print(f"   Symbol: {symbol}")
            print(f"   Action: {action}")
            print(f"   Price: {price:.{digits}f}")
            print(f"   SL: {sl:.{digits}f} (distance: {abs(price-sl):.{digits}f})")
            print(f"   TP: {tp:.{digits}f} (distance: {abs(tp-price):.{digits}f})")
            print(f"   Lot: {lot_size}")
            print(f"   Risk:Reward: 1:{rr_ratio:.1f}")
            print(f"   Max Risk: {expected_loss:.2f} {self.currency}")
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                return {'success': False, 'error': 'order_send returned None'}
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order failed: {result.comment} (code: {result.retcode})"
                
                # Provide helpful error messages
                if result.retcode == mt5.TRADE_RETCODE_INVALID_STOPS:
                    error_msg += f"\n   Hint: SL/TP too close. Min distance: {min_distance:.{digits}f}"
                elif result.retcode == mt5.TRADE_RETCODE_NO_MONEY:
                    error_msg += f"\n   Hint: Insufficient funds. Check margin requirements."
                elif result.retcode == mt5.TRADE_RETCODE_INVALID_PRICE:
                    error_msg += f"\n   Hint: Price changed. Try again."
                elif result.retcode == 10027:
                    error_msg += f"\n   Hint: AutoTrading disabled in MT5. Press Ctrl+E"
                
                return {'success': False, 'error': error_msg}
            
            return {
                'success': True,
                'ticket': result.order,
                'price': price,
                'sl': sl,
                'tp': tp,
                'volume': lot_size,
                'expected_risk': expected_loss,
                'rr_ratio': rr_ratio
            }
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return {'success': False, 'error': f'{str(e)}\n{error_detail}'}
    
    def _calculate_gold_sl_tp(self, atr: float, multiplier: float, lot_size: float, price: float) -> tuple:
        """Calculate SL/TP for Gold with PROPER RISK (Updated to use helper)"""
        
        # 1. Ambil nilai kerugian & profit (sudah dalam mata uang akun, cth: IDR)
        max_loss = self.max_loss_per_trade 
        usd_tp_target = self.config['current'].get('auto_close_target', 0.5)
        tp_target_currency = usd_tp_target * self.rate 
        
        # 2. Hitung divisor (nilai per 1.0 poin) DALAM MATA UANG AKUN
        divisor_usd = self._get_divisor_usd('GOLD', lot_size) # Pakai 'GOLD' sbg petunjuk
        divisor_currency = divisor_usd * self.rate # Konversi ke IDR
        
        if divisor_currency == 0: 
            print("   ⚠️ Divisor Gold adalah nol! Pakai fallback ATR.")
            sl_distance = atr * multiplier
            tp_distance = sl_distance * self.config['current'].get('rr_ratio', 2.0)
            return sl_distance, tp_distance
        
        # 3. Hitung jarak harga (price distance)
        sl_distance = max_loss / divisor_currency
        tp_distance = tp_target_currency / divisor_currency

        # 4. (Opsional) Fallback jika perhitungan gagal
        if sl_distance <= 0 or tp_distance <= 0:
            print("   ⚠️ Perhitungan SL/TP gagal, pakai ATR sebagai fallback")
            sl_distance = atr * multiplier
            tp_distance = sl_distance * self.config['current'].get('rr_ratio', 2.0)
        
        # 5. Pastikan jarak minimum (misal $2 price distance untuk XAUUSD)
        min_distance_usd = 2.0 
        sl_distance = max(sl_distance, min_distance_usd)
        tp_distance = max(tp_distance, min_distance_usd) 
        
        print(f"   Gold SL Calculation (Fixed Risk & Profit):")
        print(f"   Max Risk: {max_loss:.2f} {self.currency}")
        print(f"   Target Profit: {tp_target_currency:.2f} {self.currency}")
        print(f"   Lot: {lot_size}")
        print(f"   SL Distance: {sl_distance:.2f} | TP Distance: {tp_distance:.2f}")
        
        return sl_distance, tp_distance
    
    def _calculate_forex_sl_tp(self, atr: float, multiplier: float, point: float, lot_size: float) -> tuple:
        """Calculate SL/TP for Forex pairs based on fixed risk"""
        
        # 1. Ambil nilai kerugian & profit (sudah dalam mata uang akun, cth: IDR)
        max_loss = self.max_loss_per_trade 
        usd_tp_target = self.config['current'].get('auto_close_target', 0.5)
        tp_target_currency = usd_tp_target * self.rate
        
        # 2. Hitung divisor (nilai per 1.0 poin) DALAM MATA UANG AKUN
        current_symbol = self.config['current']['symbol']
        divisor_usd = self._get_divisor_usd(current_symbol, lot_size)
        divisor_currency = divisor_usd * self.rate # Konversi ke IDR
        
        if divisor_currency == 0:
            print("   ⚠️ Divisor Forex adalah nol! Pakai fallback ATR.")
            sl_distance = atr * multiplier
            tp_distance = sl_distance * 2.0 # Fallback RR
            return sl_distance, tp_distance

        # 3. Hitung jarak harga (price distance)
        sl_distance = max_loss / divisor_currency
        tp_distance = tp_target_currency / divisor_currency

        # 4. Fallback jika perhitungan gagal
        if sl_distance <= 0 or tp_distance <= 0:
            print("   ⚠️ Perhitungan SL/TP Forex gagal, pakai ATR sebagai fallback")
            sl_distance = atr * multiplier
            tp_distance = sl_distance * self.config['current'].get('rr_ratio', 2.0)
        
        # 5. Pastikan jarak minimum (cth: 10 pips)
        # 1 pip = 10 * point (jika 5 digit), cth: 10 * 0.00001 = 0.00010
        min_pips = 10
        min_distance = (point * 10) * min_pips 
        
        sl_distance = max(sl_distance, min_distance)
        tp_distance = max(tp_distance, min_distance)
        
        print(f"   Forex SL Calculation (Fixed Risk & Profit):")
        print(f"   Max Risk: {max_loss:.2f} {self.currency}")
        print(f"   Target Profit: {tp_target_currency:.2f} {self.currency}")
        print(f"   Lot: {lot_size}")
        print(f"   SL Distance: {sl_distance:.5f} | TP Distance: {tp_distance:.5f}")
        
        return sl_distance, tp_distance
    
    def _calculate_crypto_sl_tp(self, atr: float, multiplier: float, lot_size: float) -> tuple:
        """Calculate SL/TP for Crypto (BTC, ETH, etc) based on fixed risk"""

        # 1. Ambil nilai kerugian & profit (sudah dalam mata uang akun, cth: IDR)
        max_loss = self.max_loss_per_trade 
        usd_tp_target = self.config['current'].get('auto_close_target', 0.5)
        tp_target_currency = usd_tp_target * self.rate
        
        # 2. Hitung divisor (nilai per 1.0 poin) DALAM MATA UANG AKUN
        current_symbol = self.config['current']['symbol']
        divisor_usd = self._get_divisor_usd(current_symbol, lot_size) # Asumsi BTC, dll pakai logika * 10
        divisor_currency = divisor_usd * self.rate # Konversi ke IDR
        
        if divisor_currency == 0:
            print("   ⚠️ Divisor Crypto adalah nol! Pakai fallback ATR.")
            sl_distance = atr * multiplier
            tp_distance = sl_distance * 2.0 # Fallback RR
            return sl_distance, tp_distance

        # 3. Hitung jarak harga (price distance)
        sl_distance = max_loss / divisor_currency
        tp_distance = tp_target_currency / divisor_currency

        # 4. Fallback jika perhitungan gagal
        if sl_distance <= 0 or tp_distance <= 0:
            print("   ⚠️ Perhitungan SL/TP Crypto gagal, pakai ATR sebagai fallback")
            sl_distance = atr * multiplier
            tp_distance = sl_distance * self.config['current'].get('rr_ratio', 2.0)
        
        # 5. Pastikan jarak minimum (cth: $100 price distance)
        min_distance_usd = 100.0 
        sl_distance = max(sl_distance, min_distance_usd)
        tp_distance = max(tp_distance, min_distance_usd)
        
        print(f"   Crypto SL Calculation (Fixed Risk & Profit):")
        print(f"   Max Risk: {max_loss:.2f} {self.currency}")
        print(f"   Target Profit: {tp_target_currency:.2f} {self.currency}")
        print(f"   Lot: {lot_size}")
        print(f"   SL Distance: {sl_distance:.2f} | TP Distance: {tp_distance:.2f}")
        
        return sl_distance, tp_distance
    
    def _get_divisor_usd(self, symbol: str, lot_size: float) -> float:
        """
        Menghitung 'divisor_usd': nilai profit (dalam USD) 
        untuk 1.0 poin pergerakan harga (cth: 4030 ke 4031) 
        pada lot_size yang diberikan.
        
        Ini berdasarkan logika dari fungsi _calculate_risk_usd yang lama.
        """
        try:
            if 'XAU' in symbol or 'GOLD' in symbol:
                # Logika lama: Risk = price_distance * lot_size * 10
                # Jadi, divisor_usd (nilai 1.0 poin) = lot_size * 10
                return lot_size * 10
            
            elif 'BTC' in symbol:
                # Logika lama: Risk = price_distance * lot_size * 10
                # Jadi, divisor_usd (nilai 1.0 poin) = lot_size * 10
                return lot_size * 10
                
            else:
                # Forex (Logika lama)
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    point = symbol_info.point
                    if point > 0:
                        # Logika lama: Risk = (price_distance / point) * (0.10 * (lot_size / 0.01))
                        # Jadi, divisor_usd (nilai 1.0 poin) adalah:
                        pip_value_usd = 0.10 * (lot_size / 0.01)
                        # 1 pip = 10 point (jika 5 digit), jadi 1 point = 0.1 pip
                        # Asumsi 1 pip = 10 * point (cth: 10 * 0.00001 = 0.0001)
                        # Ini adalah estimasi kasar dari kode lama
                        return pip_value_usd / (point * 10) 
            
            # Fallback jika semua gagal (asumsi Gold)
            print(f"⚠️ Peringatan: _get_divisor_usd gagal untuk {symbol}, pakai fallback Gold.")
            return lot_size * 10
            
        except Exception as e:
            print(f"❌ Error di _get_divisor_usd: {e}. Pakai fallback Gold.")
            return lot_size * 10
    
    def _get_filling_type(self, symbol_info) -> int:
        """Determine the correct filling type for the symbol"""
        filling_modes = symbol_info.filling_mode
        
        # Check available filling modes in order of preference
        if filling_modes & 2:  # FOK (Fill or Kill)
            return mt5.ORDER_FILLING_FOK
        elif filling_modes & 1:  # IOC (Immediate or Cancel)
            return mt5.ORDER_FILLING_IOC
        else:  # Return (Market execution)
            return mt5.ORDER_FILLING_RETURN
    
    def _calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate ATR for SL/TP sizing"""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, period + 1)
        
        if rates is None or len(rates) < period:
            # Default ATR based on symbol
            if 'XAU' in symbol or 'GOLD' in symbol:
                return 8.0  # Gold default ~$8 ATR
            elif 'BTC' in symbol:
                return 500.0  # Bitcoin
            else:
                return 0.0010  # Forex default
        
        df = pd.DataFrame(rates)
        
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        # Ensure minimum ATR
        if pd.isna(atr) or atr == 0:
            if 'XAU' in symbol or 'GOLD' in symbol:
                return 8.0
            elif 'BTC' in symbol:
                return 500.0
            else:
                return 0.0010
        
        return atr
    
    def modify_position(self, ticket: int, new_sl: Optional[float] = None, new_tp: Optional[float] = None) -> bool:
        """Modify existing position's SL/TP"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            pos = position[0]
            symbol_info = mt5.symbol_info(pos.symbol)
            
            # Use existing values if not provided
            sl = new_sl if new_sl is not None else pos.sl
            tp = new_tp if new_tp is not None else pos.tp
            
            # Round to symbol digits
            if sl:
                sl = round(sl, symbol_info.digits)
            if tp:
                tp = round(tp, symbol_info.digits)
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": pos.symbol,
                "sl": sl,
                "tp": tp
            }
            
            result = mt5.order_send(request)
            return result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            print(f"Error modifying position: {e}")
            return False
    
    def close_position(self, ticket: int) -> bool:
        """Close specific position"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            pos = position[0]
            symbol_info = mt5.symbol_info(pos.symbol)
            tick = mt5.symbol_info_tick(pos.symbol)
            
            if not tick:
                return False
            
            # Determine close price
            close_price = tick.bid if pos.type == 0 else tick.ask
            
            # Determine filling type
            filling_type = self._get_filling_type(symbol_info)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                "price": close_price,
                "deviation": self.config['current']['slippage'],
                "magic": 234000,
                "comment": "Bot Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_type,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return True
            else:
                print(f"❌ Close failed: {result.comment}")
                return False
            
        except Exception as e:
            print(f"Error closing position: {e}")
            return False
    
    def manage_open_positions(self) -> None:
        """Manage all open positions - AUTO CLOSE, BEP, TRAILING"""
        # --- TAMBAHKAN TRY DI SINI ---
        try:
            positions = mt5.positions_get()
            if positions is None: # Cek jika gagal ambil posisi
                 last_err = mt5.last_error()
                 print(f"⚠️ Peringatan: Gagal mengambil posisi terbuka (Error: {last_err}). Lewati siklus manage.")
                 return # Keluar jika gagal

            if not positions:
                return # Tidak ada posisi, keluar
            
            # Filter our bot positions
            bot_positions = [p for p in positions if p.magic == 234000]
            
            if not bot_positions:
                return
            
            # Check if auto close is enabled
            cfg_current = self.config['current'] # Ambil config sekali
            if not cfg_current.get('auto_close_profit', False):
                # Still do BEP and trailing
                for position in bot_positions:
                    if cfg_current.get('bep', False):
                        usd_bep_min_profit = cfg_current.get('bep_min_profit', 0.2)
                        self._check_breakeven(position, usd_bep_min_profit * self.rate) # Kirim nilai yg sdh dikonversi
                    
                    if cfg_current.get('stpp_trailing', False):
                        usd_step_init = cfg_current.get('step_lock_init', 0.3)
                        usd_step_size = cfg_current.get('step_step', 0.1)
                        self._update_trailing_stop(position, usd_step_init * self.rate, usd_step_size * self.rate)
                return
            
            # Get auto close settings
            mode = cfg_current.get('auto_close_mode', 'PER_TRADE')
            usd_per_trade_target = cfg_current.get('auto_close_target', 0.4)
            usd_total_target = cfg_current.get('auto_close_total_target', 5.0)
            close_all_on_target = cfg_current.get('close_all_on_target', False)
            
            # --- KONVERSI NILAI ---
            per_trade_target = usd_per_trade_target * self.rate
            total_target = usd_total_target * self.rate
            # Calculate total profit
            total_profit = sum(p.profit for p in bot_positions)
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # === MODE: TOTAL ===
            if mode == 'TOTAL' or mode == 'BOTH':
                if total_profit >= total_target: # Gunakan nilai yang sudah dikonversi
                    print(f"\n{'='*70}")
                    print(f"[{timestamp}] 🎯 TOTAL PROFIT TARGET HIT!")
                    print(f"   Total Profit: {total_profit:.2f} {self.currency} (Target: {total_target:.2f} {self.currency})")
                    print(f"   Closing ALL {len(bot_positions)} position(s)...")
                    print(f"{'='*70}")
                    
                    # Close all positions
                    closed_count = 0
                    for position in bot_positions:
                        print(f"\n   Closing {position.symbol} #{position.ticket} (Profit: ${position.profit:.2f})")
                        if self.close_position(position.ticket):
                            closed_count += 1
                            time.sleep(0.5)  # Small delay between closes
                    
                    print(f"\n✅ Closed {closed_count}/{len(bot_positions)} positions")
                    print(f"{'='*70}")
                    return  # Don't process individual positions
            
            # === MODE: PER_TRADE or BOTH ===
            if mode == 'PER_TRADE' or mode == 'BOTH':
                for position in bot_positions:
                    # Check per-trade profit target
                    if position.profit >= per_trade_target: # Gunakan nilai yang sudah dikonversi
                        print(f"\n{'='*70}")
                        print(f"[{timestamp}] 💰 PER-TRADE PROFIT TARGET HIT!")
                        print(f"   {position.symbol} Ticket #{position.ticket}")
                        print(f"   Profit: {position.profit:.2f} {self.currency} (Target: {per_trade_target:.2f} {self.currency})")
                        
                        # If close_all_on_target is enabled, close all positions
                        if close_all_on_target:
                            print(f"   🔴 CLOSE ALL mode - Closing ALL positions!")
                            print(f"{'='*70}")
                            
                            closed_count = 0
                            for p in bot_positions:
                                print(f"\n   Closing {p.symbol} #{p.ticket} (Profit: ${p.profit:.2f})")
                                if self.close_position(p.ticket):
                                    closed_count += 1
                                    time.sleep(0.5)
                            
                            print(f"\n✅ Closed {closed_count}/{len(bot_positions)} positions")
                            print(f"{'='*70}")
                            return  # Stop processing
                        
                        # Otherwise, just close this position
                        else:
                            if self.close_position(position.ticket):
                                print(f"   ✅ Position closed successfully!")
                            else:
                                print(f"   ❌ Failed to close position")
                            print(f"{'='*70}")
                            
                            continue  # Skip BEP/trailing for closed position
                    
                    # If position not closed, apply BEP and trailing
                    if cfg_current.get('bep', False):
                        usd_bep_min_profit = cfg_current.get('bep_min_profit', 0.2)
                        self._check_breakeven(position, usd_bep_min_profit * self.rate) # Kirim nilai yg sdh dikonversi
                    
                    if cfg_current.get('stpp_trailing', False):
                        usd_step_init = cfg_current.get('step_lock_init', 0.3)
                        usd_step_size = cfg_current.get('step_step', 0.1)
                        self._update_trailing_stop(position, usd_step_init * self.rate, usd_step_size * self.rate) # Kirim nilai yg sdh dikonversi
                        
        except Exception as e:
            print(f"Error managing positions: {e}")
            import traceback
            traceback.print_exc()
        except mt5.MT5Exception as mt5_err:
            print(f"❌ Error MT5 di manage_open_positions: {mt5_err}")
            print(f"   Kode Error: {mt5.last_error()}")
        except AttributeError as attr_err:
             # Ini bisa terjadi jika objek 'position' tiba-tiba None
             print(f"❌ Error Akses Atribut di manage_open_positions: {attr_err}")
             print(f"   Mungkin posisi sudah ditutup?")
        except Exception as e:
            print(f"❌ Error tidak terduga di manage_open_positions: {e}")
            import traceback
            traceback.print_exc()
    
    def _check_breakeven(self, position, min_profit: float) -> None:
        """Move stop loss to breakeven when profit threshold reached"""
        try:
            spread_mult = self.config['current'].get('bep_spread_multiplier', 1.0)
            
            # Check if profit meets threshold
            if position.profit < min_profit: # Gunakan min_profit dari parameter
                return
            
            symbol_info = mt5.symbol_info(position.symbol)
            spread = (symbol_info.ask - symbol_info.bid)
            
            # Calculate breakeven level (entry + spread)
            if position.type == 0:  # BUY
                bep_level = position.price_open + (spread * spread_mult)
                # Only move SL up
                if position.sl < bep_level:
                    if self.modify_position(position.ticket, new_sl=bep_level):
                        print(f"🔒 BEP: {position.symbol} #{position.ticket} @ {bep_level:.{symbol_info.digits}f}")
            else:  # SELL
                bep_level = position.price_open - (spread * spread_mult)
                # Only move SL down
                if position.sl > bep_level or position.sl == 0:
                    if self.modify_position(position.ticket, new_sl=bep_level):
                        print(f"🔒 BEP: {position.symbol} #{position.ticket} @ {bep_level:.{symbol_info.digits}f}")
                    
        except Exception as e:
            pass  # Silently skip BEP errors
    
    def _update_trailing_stop(self, position, step_init: float, step_size: float) -> None: # Tambahkan parameter
        """Update trailing stop based on profit"""
        try:
            step_init = self.config['current'].get('step_lock_init', 0.3)
            step_size = self.config['current'].get('step_step', 0.1)
            
            # Check if profit meets initial threshold
            if position.profit < step_init: # Gunakan parameter
                return
            
            symbol_info = mt5.symbol_info(position.symbol)
            
            # Calculate how many steps passed
            steps_passed = int((position.profit - step_init) / step_size)
            
            if steps_passed < 1:
                return
            
            # Calculate new SL level in USD
            new_sl_profit = step_init + (steps_passed * step_size)
            
            # Convert USD profit to price distance
            # This is approximate and works best for Gold
            if 'XAU' in position.symbol or 'GOLD' in position.symbol:
                # For gold: $1 profit ≈ $1 price movement for 0.01 lot
                price_distance = new_sl_profit / (position.volume * 100)
            else:
                # For forex: approximate
                price_distance = new_sl_profit * 0.0001 / position.volume
            
            # Calculate new SL price
            if position.type == 0:  # BUY
                new_sl = position.price_open + price_distance
                # Only move SL up
                if new_sl > position.sl:
                    if self.modify_position(position.ticket, new_sl=new_sl):
                        print(f"📈 Trailing: {position.symbol} #{position.ticket} SL→{new_sl:.{symbol_info.digits}f}")
            else:  # SELL
                new_sl = position.price_open - price_distance
                # Only move SL down
                if new_sl < position.sl or position.sl == 0:
                    if self.modify_position(position.ticket, new_sl=new_sl):
                        print(f"📉 Trailing: {position.symbol} #{position.ticket} SL→{new_sl:.{symbol_info.digits}f}")
                    
        except Exception as e:
            pass  # Silently skip trailing errors
