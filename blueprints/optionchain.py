from flask import Blueprint, render_template, request, jsonify, session
from flask_wtf.csrf import validate_csrf
from flask_socketio import join_room, leave_room
from database.auth_db import get_auth_token
from database.symbol import get_symbols_with_options, get_expiry_dates
from services.optionchain_service import get_option_chain_data, subscribe_to_option_chain
from utils.session import check_session_validity
from extensions import socketio
import logging

# Create the optionchain blueprint
optionchain_bp = Blueprint('optionchain_bp', __name__, url_prefix='/optionchain')

@optionchain_bp.route('/')
@check_session_validity
def optionchain():
    """Render the option chain page"""
    return render_template('optionchain.html')

@optionchain_bp.route('/api/symbols', methods=['GET'])
@check_session_validity
def get_symbols():
    """Get symbols that have option chains available"""
    try:
        symbols = get_symbols_with_options()
        return jsonify({
            'status': 'success',
            'data': symbols
        })
    
    except Exception as e:
        logging.error(f"Error fetching symbols: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch symbols'
        }), 500

@optionchain_bp.route('/api/expiry', methods=['GET'])
@check_session_validity
def get_expiry():
    """Get expiry dates for a given symbol"""
    try:
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({
                'status': 'error',
                'message': 'Symbol parameter is required'
            }), 400
        
        expiry_dates = get_expiry_dates(symbol)
        return jsonify({
            'status': 'success',
            'data': expiry_dates
        })
    
    except Exception as e:
        logging.error(f"Error fetching expiry dates: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch expiry dates'
        }), 500

@optionchain_bp.route('/api/data', methods=['POST'])
@check_session_validity
def get_option_data():
    """Get option chain data for a symbol and expiry"""
    try:
        # Validate CSRF token
        validate_csrf(request.headers.get('X-CSRFToken'))
        
        data = request.get_json()
        symbol = data.get('symbol')
        expiry = data.get('expiry')
        
        if not symbol or not expiry:
            return jsonify({
                'status': 'error',
                'message': 'Symbol and expiry are required'
            }), 400
        
        # Get auth token for the user
        username = session.get('user')
        auth_token = get_auth_token(username)
        if not auth_token:
            return jsonify({
                'status': 'error',
                'message': 'Authentication token not found'
            }), 401
        
        # Get option chain data
        broker = session.get('broker')
        option_data = get_option_chain_data(symbol, expiry, auth_token, broker)
        
        return jsonify({
            'status': 'success',
            'data': option_data
        })
    
    except Exception as e:
        logging.error(f"Error fetching option chain data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch option chain data'
        }), 500

@optionchain_bp.route('/api/subscribe', methods=['POST'])
@check_session_validity
def subscribe_option_chain():
    """Subscribe to real-time option chain updates"""
    try:
        # Validate CSRF token
        validate_csrf(request.headers.get('X-CSRFToken'))
        
        data = request.get_json()
        symbol = data.get('symbol')
        expiry = data.get('expiry')
        
        if not symbol or not expiry:
            return jsonify({
                'status': 'error',
                'message': 'Symbol and expiry are required'
            }), 400
        
        # Get auth token for the user
        username = session.get('user')
        auth_token = get_auth_token(username)
        if not auth_token:
            return jsonify({
                'status': 'error',
                'message': 'Authentication token not found'
            }), 401
        
        # Subscribe to option chain updates
        broker = session.get('broker')
        subscription_result = subscribe_to_option_chain(symbol, expiry, auth_token, username, broker)
        
        return jsonify({
            'status': 'success',
            'data': subscription_result
        })
    
    except Exception as e:
        logging.error(f"Error subscribing to option chain: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to subscribe to option chain updates'
        }), 500

# SocketIO event handlers for option chain
@socketio.on('connect')
def handle_optionchain_connect():
    """Handle SocketIO connection and join user room"""
    username = session.get('user')
    if username:
        join_room(f'user_{username}')
        logging.info(f"User {username} joined SocketIO room for option chain")

@socketio.on('disconnect')
def handle_optionchain_disconnect():
    """Handle SocketIO disconnection and leave user room"""
    username = session.get('user')
    if username:
        leave_room(f'user_{username}')
        logging.info(f"User {username} left SocketIO room for option chain")
