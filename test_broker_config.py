#!/usr/bin/env python3
"""
Test script to validate the broker configuration management system.
This script demonstrates that the BrokerConfigManager works correctly.
"""

import sys
import os
from flask import Flask

def test_broker_config_system():
    """Test the broker configuration system."""
    
    # Set up minimal Flask app context
    app = Flask(__name__)
    app.config['TESTING'] = True
    
    with app.app_context():
        try:
            from utils.broker_config import BrokerConfigManager
            
            print("🧪 Testing Broker Configuration Management System\n")
            
            # Initialize manager
            manager = BrokerConfigManager()
            print("✓ BrokerConfigManager initialized successfully")
            
            # Test Angel configuration loading
            angel_config = manager.get_broker_config('angel')
            if angel_config:
                print("✓ Angel configuration loaded successfully")
                
                # Test API configuration
                base_url = manager.get_base_url('angel')
                headers = manager.get_headers('angel')
                print(f"✓ Base URL: {base_url}")
                print(f"✓ Headers loaded: {len(headers)} headers")
                
                # Test WebSocket configuration
                ws_config = manager.get_websocket_config('angel')
                print(f"✓ WebSocket URL: {ws_config['url']}")
                
                # Test features
                features = manager.get_features('angel')
                print(f"✓ Features: {len(features)} features supported")
                
                # Test exchanges and timeframes
                exchanges = manager.get_exchanges('angel')
                timeframes = manager.get_timeframes('angel')
                print(f"✓ Exchanges: {len(exchanges)} exchanges")
                print(f"✓ Timeframes: {len(timeframes)} timeframes")
                
                # Test convenience functions
                from utils.broker_config import get_broker_config_manager
                manager2 = get_broker_config_manager()
                assert manager is manager2, "Singleton pattern working"
                print("✓ Singleton pattern verified")
                
                print("\n🎉 All tests passed! Broker configuration system is working correctly.")
                return True
                
            else:
                print("❌ Angel configuration not found")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    success = test_broker_config_system()
    sys.exit(0 if success else 1)
