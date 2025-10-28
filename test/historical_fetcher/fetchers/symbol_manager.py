"""
Symbol Manager for Historical Data Fetcher

Handles symbol retrieval, filtering, and categorization from OpenAlgo's symtoken table.
"""

import sys
import os
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from database.symbol import SymToken, db_session
from config.settings import Settings, InstrumentType, TimeFrame

@dataclass
class SymbolInfo:
    """Enhanced symbol information with instrument classification"""
    symbol: str
    brsymbol: str
    name: str
    exchange: str
    brexchange: str
    instrument_token: str
    instrument_type: InstrumentType
    
    # F&O specific fields
    expiry: Optional[str] = None
    strike: Optional[float] = None
    underlying_symbol: Optional[str] = None
    
    # Additional metadata
    lot_size: int = 1
    tick_size: float = 0.05
    
    def get_table_name(self) -> str:
        """Get appropriate QuestDB table name based on instrument type"""
        table_mapping = {
            InstrumentType.EQUITY: "equity_historical_data",
            InstrumentType.FUTURES: "futures_historical_data",
            InstrumentType.CALL_OPTION: "options_historical_data",
            InstrumentType.PUT_OPTION: "options_historical_data",
            InstrumentType.INDEX: "index_historical_data"
        }
        return table_mapping.get(self.instrument_type, "historical_data_master")
    
    def extract_underlying_symbol(self) -> str:
        """Extract underlying symbol for F&O instruments"""
        if self.instrument_type in [InstrumentType.FUTURES, InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION]:
            if self.underlying_symbol:
                return self.underlying_symbol
            # Extract from symbol name (e.g., RELIANCE24JANFUT -> RELIANCE)
            match = re.match(r'^([A-Z]+)', self.symbol)
            return match.group(1) if match else self.symbol
        return self.symbol
    
    def is_weekly_option(self) -> bool:
        """Check if option is weekly (simplified logic)"""
        if self.instrument_type not in [InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION]:
            return False
        
        if not self.expiry:
            return False
        
        try:
            expiry_date = datetime.strptime(self.expiry, "%d-%b-%y").date()
            # Weekly options typically expire on Thursdays, monthly on last Thursday
            return expiry_date.weekday() == 3 and expiry_date.day <= 7
        except:
            return False

class SymbolManager:
    """Manages symbol retrieval and filtering from OpenAlgo database"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = db_session
    
    async def get_symbols_by_instrument_type(
        self, 
        instrument_type: InstrumentType,
        exchanges: List[str] = None
    ) -> List[SymbolInfo]:
        """Get symbols filtered by instrument type"""
        
        query = self.session.query(SymToken).filter(
            SymToken.instrumenttype == instrument_type.value,
            SymToken.token.isnot(None),
            SymToken.token != ""
        )
        
        if exchanges:
            query = query.filter(SymToken.exchange.in_(exchanges))
        
        symbols = query.all()
        
        return [self._convert_to_symbol_info(sym) for sym in symbols]
    
    async def get_all_active_symbols(self) -> Dict[InstrumentType, List[SymbolInfo]]:
        """Get all active symbols categorized by instrument type"""
        
        result = {}
        
        for instrument_type in self.settings.get_instrument_type_objects():
            symbols = await self.get_symbols_by_instrument_type(
                instrument_type, 
                self.settings.enabled_exchanges
            )
            result[instrument_type] = symbols
        
        return result
    
    async def get_equity_symbols(self) -> List[SymbolInfo]:
        """Get equity symbols from NSE and BSE"""
        return await self.get_symbols_by_instrument_type(
            InstrumentType.EQUITY,
            ["NSE", "BSE"]
        )
    
    async def get_futures_symbols(self) -> List[SymbolInfo]:
        """Get futures symbols"""
        return await self.get_symbols_by_instrument_type(
            InstrumentType.FUTURES,
            ["NFO", "BFO", "MCX"]
        )
    
    async def get_options_symbols(self) -> List[SymbolInfo]:
        """Get options symbols (both CE and PE)"""
        ce_symbols = await self.get_symbols_by_instrument_type(
            InstrumentType.CALL_OPTION,
            ["NFO", "BFO"]
        )
        pe_symbols = await self.get_symbols_by_instrument_type(
            InstrumentType.PUT_OPTION,
            ["NFO", "BFO"]
        )
        return ce_symbols + pe_symbols
    
    async def get_index_symbols(self) -> List[SymbolInfo]:
        """Get index symbols"""
        return await self.get_symbols_by_instrument_type(
            InstrumentType.INDEX,
            ["NSE_INDEX", "BSE_INDEX"]
        )
    
    async def filter_symbols_by_criteria(
        self,
        symbols: List[SymbolInfo],
        min_volume: Optional[int] = None,
        active_contracts_only: bool = True,
        exclude_weekly_options: bool = False
    ) -> List[SymbolInfo]:
        """Apply additional filtering criteria"""
        
        filtered_symbols = symbols
        
        # Filter active F&O contracts (not expired)
        if active_contracts_only:
            current_date = datetime.now().date()
            filtered_symbols = [
                sym for sym in filtered_symbols
                if not sym.expiry or self._parse_expiry_date(sym.expiry) >= current_date
            ]
        
        # Filter weekly options if requested
        if exclude_weekly_options:
            filtered_symbols = [
                sym for sym in filtered_symbols
                if not sym.is_weekly_option()
            ]
        
        return filtered_symbols
    
    async def get_symbols_for_batch(
        self, 
        symbols: List[SymbolInfo],
        batch_size: int, 
        offset: int = 0
    ) -> List[SymbolInfo]:
        """Get symbols in batches for processing"""
        
        start_idx = offset * batch_size
        end_idx = start_idx + batch_size
        
        return symbols[start_idx:end_idx]
    
    def _convert_to_symbol_info(self, sym: SymToken) -> SymbolInfo:
        """Convert SymToken to SymbolInfo"""
        
        # Extract instrument token (before ::::)
        instrument_token = sym.token.split('::::')[0] if '::::' in sym.token else sym.token
        
        # Determine instrument type
        try:
            instrument_type = InstrumentType(sym.instrumenttype) if sym.instrumenttype else InstrumentType.EQUITY
        except ValueError:
            # Handle unknown instrument types
            instrument_type = InstrumentType.EQUITY
        
        # Extract underlying symbol for F&O
        underlying_symbol = None
        if instrument_type in [InstrumentType.FUTURES, InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION]:
            underlying_symbol = self._extract_underlying_from_symbol(sym.symbol)
        
        return SymbolInfo(
            symbol=sym.symbol,
            brsymbol=sym.brsymbol,
            name=sym.name or sym.symbol,
            exchange=sym.exchange,
            brexchange=sym.brexchange,
            instrument_token=instrument_token,
            instrument_type=instrument_type,
            expiry=sym.expiry,
            strike=sym.strike,
            underlying_symbol=underlying_symbol,
            lot_size=sym.lotsize or 1,
            tick_size=sym.tick_size or 0.05
        )
    
    def _parse_expiry_date(self, expiry_str: str) -> datetime.date:
        """Parse expiry date from DD-MMM-YY format"""
        try:
            return datetime.strptime(expiry_str, "%d-%b-%y").date()
        except:
            return datetime.now().date()
    
    def _extract_underlying_from_symbol(self, symbol: str) -> str:
        """Extract underlying symbol from F&O symbol"""
        # Pattern to match underlying symbol (letters at the beginning)
        match = re.match(r'^([A-Z]+)', symbol)
        return match.group(1) if match else symbol
    
    async def get_symbol_statistics(self) -> Dict[str, int]:
        """Get statistics about symbols in database"""
        
        stats = {}
        
        # Total symbols
        total_query = self.session.query(SymToken).filter(
            SymToken.token.isnot(None),
            SymToken.token != ""
        )
        stats['total_symbols'] = total_query.count()
        
        # By instrument type
        for instrument_type in InstrumentType:
            count = self.session.query(SymToken).filter(
                SymToken.instrumenttype == instrument_type.value,
                SymToken.token.isnot(None),
                SymToken.token != ""
            ).count()
            stats[f'{instrument_type.value}_symbols'] = count
        
        # By exchange
        for exchange in self.settings.enabled_exchanges:
            count = self.session.query(SymToken).filter(
                SymToken.exchange == exchange,
                SymToken.token.isnot(None),
                SymToken.token != ""
            ).count()
            stats[f'{exchange}_symbols'] = count
        
        return stats
