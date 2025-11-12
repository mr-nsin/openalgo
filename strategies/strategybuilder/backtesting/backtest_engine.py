"""
Backtesting Engine for Strategy Builder

Provides a framework for backtesting trading strategies
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class Trade:
    """Represents a single trade"""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position_type: str  # 'Long' or 'Short'
    pnl: float
    stop_loss: Optional[float] = None
    exit_reason: str = ""  # 'Signal', 'Stop Loss', etc.


@dataclass
class BacktestResult:
    """Results from a backtest"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy display"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': f"{self.win_rate:.2f}%",
            'total_pnl': f"{self.total_pnl:.2f}",
            'initial_capital': f"{self.initial_capital:.2f}",
            'final_capital': f"{self.final_capital:.2f}",
            'total_return': f"{self.total_return:.2f}%",
            'max_drawdown': f"{self.max_drawdown:.2f}%",
            'avg_win': f"{self.avg_win:.2f}",
            'avg_loss': f"{self.avg_loss:.2f}",
            'profit_factor': f"{self.profit_factor:.2f}",
        }


class BacktestEngine:
    """
    Backtesting engine for trading strategies
    
    Handles position management, stop loss tracking, and trade execution
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # 0=No position, 1=Long, -1=Short
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.entry_index = 0
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        strategy,
        initial_capital: Optional[float] = None
    ) -> BacktestResult:
        """
        Run backtest on a strategy
        
        Args:
            df: DataFrame with price data and signals (must have 'Signal' column)
            strategy: Strategy object with get_stop_loss method
            initial_capital: Starting capital (overrides engine default)
        
        Returns:
            BacktestResult object with statistics
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital
            self.capital = initial_capital
        
        self.position = 0
        self.trades = []
        self.equity_curve = [self.initial_capital]
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'Signal']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Get minimum period needed (usually long_sma_period)
        min_period = getattr(strategy.params, 'long_sma_period', 200)
        
        # Process each bar
        for i in range(min_period, len(df)):
            current_price = df['close'].iloc[i]
            signal = df['Signal'].iloc[i]
            current_date = df.index[i] if hasattr(df.index, '__getitem__') else i
            
            # Update stop loss for existing positions
            if self.position != 0:
                self._update_stop_loss(df, i, strategy)
            
            # Check for stop loss hit
            if self.position == 1:  # Long position
                if current_price < self.stop_loss:
                    self._exit_position(df, i, 'Stop Loss', current_price)
                    continue
            elif self.position == -1:  # Short position
                if current_price > self.stop_loss:
                    self._exit_position(df, i, 'Stop Loss', current_price)
                    continue
            
            # Check for signal-based exit
            if self.position == 1 and signal == -1:
                self._exit_position(df, i, 'Signal', current_price)
            elif self.position == -1 and signal == 1:
                self._exit_position(df, i, 'Signal', current_price)
            
            # Check for new entry signals
            if self.position == 0:
                if signal == 1:  # Buy signal
                    self._enter_position(df, i, 1, current_price, strategy)
                elif signal == -1:  # Sell signal
                    self._enter_position(df, i, -1, current_price, strategy)
            
            # Update equity curve
            self._update_equity(current_price)
        
        # Close any open position at the end
        if self.position != 0:
            final_price = df['close'].iloc[-1]
            self._exit_position(df, len(df) - 1, 'End of Data', final_price)
        
        # Calculate results
        return self._calculate_results()
    
    def _enter_position(
        self,
        df: pd.DataFrame,
        index: int,
        position_type: int,
        entry_price: float,
        strategy
    ):
        """Enter a new position"""
        self.position = position_type
        self.entry_price = entry_price
        self.entry_index = index
        self.stop_loss = strategy.get_stop_loss(df, index, position_type)
        
        logger.debug(
            f"Entered {'Long' if position_type == 1 else 'Short'} position at {entry_price:.2f}, "
            f"Stop Loss: {self.stop_loss:.2f}"
        )
    
    def _exit_position(
        self,
        df: pd.DataFrame,
        index: int,
        exit_reason: str,
        exit_price: float
    ):
        """Exit current position"""
        if self.position == 0:
            return
        
        # Calculate P&L
        if self.position == 1:  # Long
            pnl = exit_price - self.entry_price
        else:  # Short
            pnl = self.entry_price - exit_price
        
        # Update capital
        self.capital += pnl
        
        # Create trade record
        entry_date = df.index[self.entry_index] if hasattr(df.index, '__getitem__') else self.entry_index
        exit_date = df.index[index] if hasattr(df.index, '__getitem__') else index
        
        trade = Trade(
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=self.entry_price,
            exit_price=exit_price,
            position_type='Long' if self.position == 1 else 'Short',
            pnl=pnl,
            stop_loss=self.stop_loss,
            exit_reason=exit_reason
        )
        
        self.trades.append(trade)
        
        logger.debug(
            f"Exited {trade.position_type} position: Entry={self.entry_price:.2f}, "
            f"Exit={exit_price:.2f}, P&L={pnl:.2f}, Reason={exit_reason}"
        )
        
        # Reset position
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
    
    def _update_stop_loss(self, df: pd.DataFrame, index: int, strategy):
        """Update trailing stop loss"""
        new_stop = strategy.get_stop_loss(df, index, self.position)
        
        if self.position == 1:  # Long - only move stop loss up
            self.stop_loss = max(self.stop_loss, new_stop)
        else:  # Short - only move stop loss down
            self.stop_loss = min(self.stop_loss, new_stop)
    
    def _update_equity(self, current_price: float):
        """Update equity curve"""
        if self.position == 0:
            equity = self.capital
        elif self.position == 1:
            unrealized_pnl = current_price - self.entry_price
            equity = self.capital + unrealized_pnl
        else:  # Short
            unrealized_pnl = self.entry_price - current_price
            equity = self.capital + unrealized_pnl
        
        self.equity_curve.append(equity)
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest statistics"""
        if not self.trades:
            return BacktestResult(
                initial_capital=self.initial_capital,
                final_capital=self.capital
            )
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        win_rate = (len(winning_trades) / len(self.trades)) * 100 if self.trades else 0
        
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        return BacktestResult(
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_pnl=total_pnl,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            total_return=total_return,
            max_drawdown=max_drawdown,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=self.trades
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        if not self.equity_curve:
            return 0.0
        
        peak = self.equity_curve[0]
        max_dd = 0.0
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = ((peak - equity) / peak) * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd

