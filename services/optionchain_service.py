import logging
from typing import Dict, List, Any, Optional
from database.symbol import get_option_symbols_by_expiry
from extensions import socketio
from utils.plugin_loader import get_broker_auth_token, get_active_broker
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
        
        # Extract tokens for WebSocket subscription
        tokens = []
        symbol_token_map = {}
        
        for option in option_symbols:
            if option.token:
                tokens.append(option.token)
                symbol_token_map[option.token] = {
                    'symbol': option.symbol,
                    'strike': option.strike,
                    'instrument_type': option.instrumenttype,
                    'exchange': option.exchange
                }
        
        # Emit SocketIO event to inform frontend about subscription
        socketio.emit('option_chain_subscription', {
            'action': 'subscribe',
            'underlying': symbol,
            'expiry': expiry,
            'tokens': tokens,
            'symbol_map': symbol_token_map,
            'total_symbols': len(tokens)
        }, room=user_id)
        
        logger.info(f"Option chain subscription initiated for {symbol} {expiry} with {len(tokens)} symbols")
        
        return {
            'status': 'success',
            'underlying': symbol,
            'expiry': expiry,
            'tokens': tokens,
            'total_symbols': len(tokens),
            'message': f'Subscription initiated for {len(tokens)} option symbols'
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
        token = market_data.get('token')
        ltp = market_data.get('ltp', 0.0)
        oi = market_data.get('oi', 0)
        change = market_data.get('change', 0.0)
        change_percent = market_data.get('change_percent', 0.0)
        
        if not token:
            return
        
        # Emit real-time update to frontend
        socketio.emit('option_chain_update', {
            'token': token,
            'ltp': ltp,
            'oi': oi,
            'change': change,
            'change_percent': change_percent,
            'timestamp': market_data.get('timestamp')
        }, room=user_id)
        
    except Exception as e:
        logger.error(f"Error processing option chain update: {str(e)}")

def unsubscribe_option_chain(user_id: str) -> Dict[str, Any]:
    """
    Unsubscribe from option chain updates
    
    Args:
        user_id: User ID for session management
    
    Returns:
        Dictionary containing unsubscription result
    """
    try:
        # Emit SocketIO event to inform frontend about unsubscription
        socketio.emit('option_chain_subscription', {
            'action': 'unsubscribe'
        }, room=user_id)
        
        logger.info(f"Option chain unsubscription initiated for user {user_id}")
        
        return {
            'status': 'success',
            'message': 'Unsubscription initiated'
        }
    
    except Exception as e:
        logger.error(f"Error unsubscribing from option chain: {str(e)}")
        raise
