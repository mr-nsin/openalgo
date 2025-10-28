import logging
import time
from typing import Dict, List, Any, Optional
from database.symbol import get_option_symbols_by_expiry
from extensions import socketio
from utils.plugin_loader import get_broker_auth_token, get_active_broker
from services.websocket_service import subscribe_to_symbols, get_websocket_connection
from services.market_data_service import MarketDataService
import json

logger = logging.getLogger(__name__)

def get_option_chain_data(symbol: str, expiry: str, auth_token: str) -> Dict[str, Any]:
    """
    Get option chain data for a given symbol and expiry
    
    Args:
        symbol: The underlying symbol (e.g., 'NIFTY', 'BANKNIFTY', 'RELIANCE')
        expiry: The expiry date
        auth_token: User's authentication token
    
    Returns:
        Dictionary containing option chain data
    """
    try:
        # Get option symbols for the given underlying and expiry
        option_symbols = get_option_symbols_by_expiry(symbol, expiry)
        
        if not option_symbols:
            return {
                'underlying': symbol,
                'expiry': expiry,
                'options': [],
                'message': 'No option data found for the given symbol and expiry'
            }
        
        # Group options by strike price
        strikes = {}
        
        for option in option_symbols:
            strike = option.strike
            if strike not in strikes:
                strikes[strike] = {
                    'strike': strike,
                    'call': None,
                    'put': None
                }
            
            option_data = {
                'symbol': option.symbol,
                'token': option.token,
                'lotsize': option.lotsize,
                'tick_size': option.tick_size,
                'ltp': 0.0,  # Will be updated via WebSocket
                'oi': 0,     # Will be updated via WebSocket
                'change': 0.0,  # Will be updated via WebSocket
                'change_percent': 0.0  # Will be updated via WebSocket
            }
            
            if option.instrumenttype == 'CE':
                strikes[strike]['call'] = option_data
            elif option.instrumenttype == 'PE':
                strikes[strike]['put'] = option_data
        
        # Convert to sorted list
        sorted_strikes = sorted(strikes.keys())
        option_chain = []
        
        for strike in sorted_strikes:
            option_chain.append(strikes[strike])
        
        return {
            'underlying': symbol,
            'expiry': expiry,
            'options': option_chain,
            'total_strikes': len(option_chain)
        }
    
    except Exception as e:
        logger.error(f"Error getting option chain data: {str(e)}")
        raise

def subscribe_to_option_chain(symbol: str, expiry: str, auth_token: str, user_id: str) -> Dict[str, Any]:
    """
    Subscribe to real-time option chain updates
    
    Args:
        symbol: The underlying symbol
        expiry: The expiry date
        auth_token: User's authentication token
        user_id: User ID for session management
    
    Returns:
        Dictionary containing subscription result and tokens to subscribe
    """
    try:
        # Get option symbols for the given underlying and expiry
        option_symbols = get_option_symbols_by_expiry(symbol, expiry)
        
        if not option_symbols:
            return {
                'status': 'error',
                'message': 'No option symbols found for subscription'
            }
        
        # Prepare symbols for WebSocket subscription
        symbols_to_subscribe = []
        symbol_token_map = {}
        
        for option in option_symbols:
            if option.token and option.exchange:
                symbol_data = {
                    'symbol': option.symbol,
                    'exchange': option.exchange,
                    'token': option.token
                }
                symbols_to_subscribe.append(symbol_data)
                
                symbol_token_map[option.token] = {
                    'symbol': option.symbol,
                    'strike': option.strike,
                    'instrument_type': option.instrumenttype,
                    'exchange': option.exchange
                }
        
        if not symbols_to_subscribe:
            return {
                'status': 'error',
                'message': 'No valid symbols found for subscription'
            }
        
        # Get active broker
        active_broker = get_active_broker()
        if not active_broker:
            return {
                'status': 'error',
                'message': 'No active broker found'
            }
        
        # Subscribe to symbols via WebSocket service
        success, response_data, status_code = subscribe_to_symbols(
            username=user_id,
            broker=active_broker,
            symbols=symbols_to_subscribe,
            mode="Quote"  # Use Quote mode for LTP, OI, and change data
        )
        
        if success:
            # Register callback for market data updates
            market_data_service = MarketDataService()
            
            def option_chain_callback(market_data):
                """Callback to handle market data updates for option chain"""
                process_option_chain_update(market_data, user_id)
            
            # Subscribe to market data updates
            subscriber_id = market_data_service.subscribe_to_updates(
                event_type='quote',
                callback=option_chain_callback,
                filter_symbols={f"{opt['exchange']}:{opt['symbol']}" for opt in symbols_to_subscribe}
            )
            
            # Store subscriber ID for cleanup (could be stored in session or database)
            # For now, we'll emit it to the frontend
            
            # Emit SocketIO event to inform frontend about successful subscription
            socketio.emit('option_chain_subscription', {
                'action': 'subscribe',
                'underlying': symbol,
                'expiry': expiry,
                'symbol_map': symbol_token_map,
                'total_symbols': len(symbols_to_subscribe),
                'subscriber_id': subscriber_id
            }, room=f'user_{user_id}')
            
            logger.info(f"Option chain subscription successful for {symbol} {expiry} with {len(symbols_to_subscribe)} symbols")
            
            return {
                'status': 'success',
                'underlying': symbol,
                'expiry': expiry,
                'total_symbols': len(symbols_to_subscribe),
                'subscriber_id': subscriber_id,
                'message': f'Successfully subscribed to {len(symbols_to_subscribe)} option symbols'
            }
        else:
            logger.error(f"WebSocket subscription failed: {response_data}")
            return {
                'status': 'error',
                'message': response_data.get('message', 'Failed to subscribe to WebSocket')
            }
    
    except Exception as e:
        logger.error(f"Error subscribing to option chain: {str(e)}")
        raise

def process_option_chain_update(market_data: Dict[str, Any], user_id: str) -> None:
    """
    Process real-time market data updates for option chain
    
    Args:
        market_data: Market data received from WebSocket
        user_id: User ID to emit updates to
    """
    try:
        # Extract relevant data from market data
        # Market data format may vary by broker, handle common fields
        symbol = market_data.get('symbol')
        exchange = market_data.get('exchange')
        token = market_data.get('token')
        ltp = market_data.get('ltp', market_data.get('last_price', 0.0))
        oi = market_data.get('oi', market_data.get('open_interest', 0))
        change = market_data.get('change', market_data.get('net_change', 0.0))
        change_percent = market_data.get('change_percent', market_data.get('percentage_change', 0.0))
        
        # Use token if available, otherwise create a key from symbol and exchange
        identifier = token or f"{exchange}:{symbol}" if symbol and exchange else None
        
        if not identifier:
            return
        
        # Emit real-time update to frontend
        socketio.emit('option_chain_update', {
            'token': identifier,
            'symbol': symbol,
            'exchange': exchange,
            'ltp': float(ltp) if ltp else 0.0,
            'oi': int(oi) if oi else 0,
            'change': float(change) if change else 0.0,
            'change_percent': float(change_percent) if change_percent else 0.0,
            'timestamp': market_data.get('timestamp', int(time.time() * 1000))
        }, room=f'user_{user_id}')
        
    except Exception as e:
        logger.error(f"Error processing option chain update: {str(e)}")

def unsubscribe_option_chain(user_id: str, subscriber_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Unsubscribe from option chain updates
    
    Args:
        user_id: User ID for session management
        subscriber_id: Optional subscriber ID to unsubscribe from market data service
    
    Returns:
        Dictionary containing unsubscription result
    """
    try:
        # Unsubscribe from market data service if subscriber_id is provided
        if subscriber_id:
            market_data_service = MarketDataService()
            success = market_data_service.unsubscribe_from_updates(subscriber_id)
            if success:
                logger.info(f"Unsubscribed from market data service with ID {subscriber_id}")
        
        # Emit SocketIO event to inform frontend about unsubscription
        socketio.emit('option_chain_subscription', {
            'action': 'unsubscribe'
        }, room=f'user_{user_id}')
        
        logger.info(f"Option chain unsubscription initiated for user {user_id}")
        
        return {
            'status': 'success',
            'message': 'Unsubscription completed'
        }
    
    except Exception as e:
        logger.error(f"Error unsubscribing from option chain: {str(e)}")
        raise
