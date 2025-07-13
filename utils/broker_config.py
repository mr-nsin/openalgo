"""
Centralized broker configuration management system for OpenAlgo.

This module provides dynamic loading and management of broker-specific configurations,
similar to how auth functions are loaded dynamically. Each broker can have its own
configuration file that defines endpoints, timeframes, limits, and other settings.
"""

import os
import json
import importlib
import inspect
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from flask import current_app
from utils.logging import get_logger
import threading

logger = get_logger(__name__)

class BrokerConfigurationError(Exception):
    """Exception raised when broker configuration is invalid or missing."""
    pass

class BrokerConfigManager:
    """
    Manages broker configurations dynamically, similar to how auth functions are loaded.
    Provides a centralized way to access broker-specific settings, endpoints, and capabilities.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the broker configuration manager."""
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.configs = {}
            self.loaded_brokers = set()
            self._cache = {}
            self._cache_lock = threading.Lock()
            self.logger = get_logger(__name__)
            
    def get_broker_name_from_caller(self) -> Optional[str]:
        """
        Extract broker name from the calling file path.
        
        Looks for pattern: broker/{broker_name}/api/*.py
        
        Returns:
            Broker name if found, None otherwise
        """
        try:
            # Get the caller's frame
            frame = inspect.currentframe()
            if frame is None:
                return None
                
            # Go up the stack to find the caller from broker directory
            while frame:
                frame = frame.f_back
                if frame is None:
                    break
                    
                filename = frame.f_code.co_filename
                if '/broker/' in filename and '/api/' in filename:
                    # Extract broker name from path pattern: ...broker/{broker_name}/api/...
                    parts = filename.split('/broker/')
                    if len(parts) > 1:
                        broker_path = parts[1]
                        broker_name = broker_path.split('/')[0]
                        return broker_name
                        
        except Exception as e:
            self.logger.debug(f"Could not extract broker name from caller: {e}")
        
        return None
    
    def load_broker_config(self, broker_name: str) -> Optional[Dict[str, Any]]:
        """
        Load configuration for a specific broker.
        
        Args:
            broker_name: Name of the broker to load configuration for
            
        Returns:
            Dictionary containing broker configuration, None if not found
        """
        if broker_name in self.configs:
            return self.configs[broker_name]
        
        try:
            # Try to load from broker-specific config file first
            config_path = self._get_broker_config_path(broker_name)
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.configs[broker_name] = config
                    self.loaded_brokers.add(broker_name)
                    self.logger.info(f"Loaded configuration for broker: {broker_name}")
                    return config
            
            # Try to load from broker module's config directory
            module_config_path = self._get_module_config_path(broker_name)
            if module_config_path and os.path.exists(module_config_path):
                with open(module_config_path, 'r') as f:
                    config = json.load(f)
                    self.configs[broker_name] = config
                    self.loaded_brokers.add(broker_name)
                    self.logger.info(f"Loaded configuration for broker: {broker_name} from module")
                    return config
            
            # If no config file found, try to create a default one
            default_config = self._create_default_config(broker_name)
            if default_config:
                self.configs[broker_name] = default_config
                self.loaded_brokers.add(broker_name)
                self.logger.warning(f"Created default configuration for broker: {broker_name}")
                return default_config
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration for broker {broker_name}: {e}")
            
        return None
    
    def _get_broker_config_path(self, broker_name: str) -> Optional[str]:
        """Get path to broker configuration file."""
        if hasattr(current_app, 'root_path'):
            return os.path.join(current_app.root_path, 'broker', broker_name, 'config', f'{broker_name}_config.json')
        return None
    
    def _get_module_config_path(self, broker_name: str) -> Optional[str]:
        """Get path to broker module configuration file."""
        if hasattr(current_app, 'root_path'):
            return os.path.join(current_app.root_path, 'broker', broker_name, f'{broker_name}_config.json')
        return None
    
    def _create_default_config(self, broker_name: str) -> Optional[Dict[str, Any]]:
        """Create a default configuration for a broker."""
        # This is a fallback - ideally all brokers should have config files
        return {
            "broker_name": broker_name,
            "api_config": {
                "base_url": f"https://api.{broker_name}.com",
                "headers": {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                "endpoints": {
                    "auth": "/auth",
                    "quotes": "/quotes",
                    "orders": "/orders",
                    "positions": "/positions",
                    "holdings": "/holdings"
                },
                "timeout": 30,
                "retry_count": 3
            },
            "websocket_config": {
                "url": f"wss://ws.{broker_name}.com",
                "heartbeat_interval": 30,
                "reconnect_delay": 5,
                "max_reconnect_attempts": 10
            },
            "features": {
                "supports_market_data": True,
                "supports_websocket": True,
                "supports_options": True,
                "supports_futures": True
            },
            "timeframes": {
                "supported": ["1m", "5m", "15m", "30m", "1h", "1D"]
            },
            "exchanges": {
                "supported": ["NSE", "BSE", "NFO", "BFO", "CDS", "MCX"]
            }
        }
    
    def get_broker_config(self, broker_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a broker.
        
        Args:
            broker_name: Name of the broker. If None, tries to detect from caller.
            
        Returns:
            Dictionary containing broker configuration, None if not found
        """
        if broker_name is None:
            broker_name = self.get_broker_name_from_caller()
            
        if broker_name is None:
            self.logger.warning("Could not determine broker name")
            return None
            
        return self.load_broker_config(broker_name)
    
    def get_api_config(self, broker_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get API configuration for a broker."""
        config = self.get_broker_config(broker_name)
        return config.get('api_config') if config else None
    
    def get_websocket_config(self, broker_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get WebSocket configuration for a broker."""
        config = self.get_broker_config(broker_name)
        return config.get('websocket_config') if config else None
    
    def get_base_url(self, broker_name: Optional[str] = None) -> Optional[str]:
        """Get base URL for a broker."""
        api_config = self.get_api_config(broker_name)
        return api_config.get('base_url') if api_config else None
    
    def get_headers(self, broker_name: Optional[str] = None) -> Dict[str, str]:
        """Get default headers for a broker."""
        api_config = self.get_api_config(broker_name)
        return api_config.get('headers', {}) if api_config else {}
    
    def get_endpoint(self, endpoint_name: str, broker_name: Optional[str] = None) -> Optional[str]:
        """Get a specific endpoint for a broker."""
        api_config = self.get_api_config(broker_name)
        if api_config:
            endpoints = api_config.get('endpoints', {})
            return endpoints.get(endpoint_name)
        return None
    
    def get_full_url(self, endpoint_name: str, broker_name: Optional[str] = None) -> Optional[str]:
        """Get full URL for a specific endpoint."""
        base_url = self.get_base_url(broker_name)
        endpoint = self.get_endpoint(endpoint_name, broker_name)
        
        if base_url and endpoint:
            return f"{base_url.rstrip('/')}{endpoint}"
        return None
    
    def get_timeframes(self, broker_name: Optional[str] = None) -> List[str]:
        """Get supported timeframes for a broker."""
        config = self.get_broker_config(broker_name)
        if config:
            return config.get('timeframes', {}).get('supported', [])
        return []
    
    def get_exchanges(self, broker_name: Optional[str] = None) -> List[str]:
        """Get supported exchanges for a broker."""
        config = self.get_broker_config(broker_name)
        if config:
            return config.get('exchanges', {}).get('supported', [])
        return []
    
    def get_features(self, broker_name: Optional[str] = None) -> Dict[str, Any]:
        """Get feature capabilities for a broker."""
        config = self.get_broker_config(broker_name)
        return config.get('features', {}) if config else {}
    
    def supports_feature(self, feature_name: str, broker_name: Optional[str] = None) -> bool:
        """Check if a broker supports a specific feature."""
        features = self.get_features(broker_name)
        return features.get(feature_name, False)
    
    def get_retry_config(self, broker_name: Optional[str] = None) -> Dict[str, Any]:
        """Get retry configuration for a broker."""
        api_config = self.get_api_config(broker_name)
        if api_config:
            return {
                'retry_count': api_config.get('retry_count', 3),
                'timeout': api_config.get('timeout', 30),
                'backoff_factor': api_config.get('backoff_factor', 0.3)
            }
        return {'retry_count': 3, 'timeout': 30, 'backoff_factor': 0.3}
    
    def clear_cache(self, broker_name: Optional[str] = None):
        """Clear cached configurations."""
        with self._cache_lock:
            if broker_name:
                self._cache.pop(broker_name, None)
            else:
                self._cache.clear()
    
    def reload_config(self, broker_name: str) -> bool:
        """Reload configuration for a specific broker."""
        try:
            if broker_name in self.configs:
                del self.configs[broker_name]
            if broker_name in self.loaded_brokers:
                self.loaded_brokers.remove(broker_name)
            
            config = self.load_broker_config(broker_name)
            return config is not None
        except Exception as e:
            self.logger.error(f"Failed to reload configuration for broker {broker_name}: {e}")
            return False
    
    def list_loaded_brokers(self) -> List[str]:
        """Get list of brokers with loaded configurations."""
        return list(self.loaded_brokers)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate a broker configuration."""
        required_keys = ['broker_name', 'api_config']
        for key in required_keys:
            if key not in config:
                return False
        
        api_config = config.get('api_config', {})
        if 'base_url' not in api_config:
            return False
            
        return True

# Global instance for easy access
_config_manager = None

def get_broker_config_manager() -> BrokerConfigManager:
    """Get the global broker configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = BrokerConfigManager()
    return _config_manager

# Convenience functions for common operations
def get_current_broker_config() -> Optional[Dict[str, Any]]:
    """Get configuration for the current broker (detected from caller)."""
    return get_broker_config_manager().get_broker_config()

def get_current_broker_headers() -> Dict[str, str]:
    """Get headers for the current broker (detected from caller)."""
    return get_broker_config_manager().get_headers()

def get_current_broker_base_url() -> Optional[str]:
    """Get base URL for the current broker (detected from caller)."""
    return get_broker_config_manager().get_base_url()

def get_current_broker_endpoint(endpoint_name: str) -> Optional[str]:
    """Get endpoint for the current broker (detected from caller)."""
    return get_broker_config_manager().get_endpoint(endpoint_name)

def get_current_broker_full_url(endpoint_name: str) -> Optional[str]:
    """Get full URL for the current broker (detected from caller)."""
    return get_broker_config_manager().get_full_url(endpoint_name)

def get_current_broker_features() -> Dict[str, Any]:
    """Get features for the current broker (detected from caller)."""
    return get_broker_config_manager().get_features()

def broker_supports_feature(feature_name: str) -> bool:
    """Check if current broker supports a specific feature."""
    return get_broker_config_manager().supports_feature(feature_name)
