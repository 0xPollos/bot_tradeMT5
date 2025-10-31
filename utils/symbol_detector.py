# utils/symbol_detector.py
import metatrader5 as mt5
from typing import Dict, List, Optional

class SymbolDetector:
    def __init__(self):
        self.symbol_cache = {}
        self.base_symbols = {
            'GOLD': ['XAU', 'GOLD'],
            'EUR': ['EUR'],
            'GBP': ['GBP'],
            'JPY': ['JPY', 'USDJPY'],
            'AUD': ['AUD'],
            'BTC': ['BTC', 'BITCOIN']
        }
        
    def find_symbol(self, base_symbol: str) -> Optional[str]:
        """Find the actual symbol name available in broker"""
        
        # Check cache first
        if base_symbol in self.symbol_cache:
            return self.symbol_cache[base_symbol]
        
        # Get all available symbols
        all_symbols = mt5.symbols_get()
        if not all_symbols:
            return None
        
        # Create list of possible variations
        variations = self._generate_variations(base_symbol)
        
        # Try to find exact or close match
        for symbol_info in all_symbols:
            name = symbol_info.name
            name_upper = name.upper()
            
            # Check exact match (case insensitive)
            if name_upper == base_symbol.upper():
                if self._test_symbol(name):
                    self.symbol_cache[base_symbol] = name
                    return name
            
            # Check variations
            for variant in variations:
                if name_upper == variant.upper():
                    if self._test_symbol(name):
                        self.symbol_cache[base_symbol] = name
                        return name
        
        # If no exact match, try contains
        for symbol_info in all_symbols:
            name = symbol_info.name
            name_upper = name.upper()
            
            # For Gold/XAU
            if 'XAU' in base_symbol.upper() or 'GOLD' in base_symbol.upper():
                if ('XAU' in name_upper or 'GOLD' in name_upper) and 'USD' in name_upper:
                    if self._test_symbol(name):
                        print(f"   Found Gold symbol: {name}")
                        self.symbol_cache[base_symbol] = name
                        return name
            
            # For forex pairs
            elif len(base_symbol) == 6:  # Likely a forex pair like EURUSD
                base_clean = base_symbol.replace('USD', '').replace('JPY', '').replace('GBP', '').replace('EUR', '')
                if base_clean in name_upper and 'USD' in name_upper:
                    if self._test_symbol(name):
                        print(f"   Found forex pair: {name}")
                        self.symbol_cache[base_symbol] = name
                        return name
        
        print(f"❌ Symbol {base_symbol} not found in broker")
        return None
    
    def _generate_variations(self, base: str) -> List[str]:
        """Generate possible symbol variations"""
        base_upper = base.upper().replace('M', '').replace('-M', '').replace('.A', '').replace('_M', '')
        
        variations = [
            base,
            base_upper,
            f"{base_upper}m",
            f"{base_upper}M",
            f"{base_upper}-m",
            f"{base_upper}-M",
            f"{base_upper}.a",
            f"{base_upper}.A",
            f"{base_upper}_m",
            f"{base_upper}_M",
            f"{base_upper}.",
            f"{base_upper}#",
            f".{base_upper}",
        ]
        
        return variations
    
    def _test_symbol(self, symbol: str) -> bool:
        """Test if symbol can be selected and has data"""
        try:
            # Try to select
            if not mt5.symbol_select(symbol, True):
                return False
            
            # Try to get tick
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return False
            
            # Try to get some rates
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 10)
            if rates is None or len(rates) == 0:
                return False
            
            return True
        except:
            return False
    
    def get_available_gold_symbols(self) -> List[str]:
        """Get all available Gold/XAU symbols"""
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
    
    def get_available_forex_pairs(self) -> Dict[str, str]:
        """Get available major forex pairs"""
        all_symbols = mt5.symbols_get()
        if not all_symbols:
            return {}
        
        pairs = {}
        major_bases = ['EUR', 'GBP', 'AUD', 'NZD', 'USD', 'CAD']
        major_quotes = ['USD', 'JPY', 'CHF', 'GBP', 'EUR']
        
        for symbol_info in all_symbols:
            name = symbol_info.name
            name_upper = name.upper()
            
            # Check if it's a major pair
            for base in major_bases:
                for quote in major_quotes:
                    if base != quote:
                        pair_name = f"{base}{quote}"
                        if pair_name in name_upper:
                            if self._test_symbol(name):
                                pairs[pair_name] = name
                                break
        
        return pairs
    
    def auto_detect_symbols(self, preferred: List[str]) -> List[str]:
        """Auto-detect actual symbol names from preferred list"""
        detected = []
        
        print("\n🔍 Auto-detecting symbols...")
        
        for pref in preferred:
            actual = self.find_symbol(pref)
            if actual:
                detected.append(actual)
                print(f"   ✅ {pref} → {actual}")
            else:
                print(f"   ❌ {pref} not available")
        
        return detected
