# utils/plugin_loader.py

import os
import importlib
from flask import current_app
from utils.logging import get_logger
from utils.broker_config import get_broker_config_manager

logger = get_logger(__name__)

def load_broker_auth_functions(broker_directory='broker'):
    """Load authentication functions for all available brokers."""
    auth_functions = {}
    broker_path = os.path.join(current_app.root_path, broker_directory)
    # List all items in broker directory and filter out __pycache__ and non-directories
    broker_names = [d for d in os.listdir(broker_path)
                    if os.path.isdir(os.path.join(broker_path, d)) and d != '__pycache__']

    for broker_name in broker_names:
        try:
            # Construct module name and import the module
            module_name = f"{broker_directory}.{broker_name}.api.auth_api"
            auth_module = importlib.import_module(module_name)
            # Retrieve the authenticate_broker function
            auth_function = getattr(auth_module, 'authenticate_broker', None)
            if auth_function:
                auth_functions[f"{broker_name}_auth"] = auth_function
        except ImportError as e:
            logger.error(f"Failed to import broker plugin {broker_name}: {e}")
        except AttributeError as e:
            logger.error(f"Authentication function not found in broker plugin {broker_name}: {e}")

    return auth_functions

def load_broker_configurations(broker_directory='broker'):
    """Load configurations for all available brokers."""
    config_manager = get_broker_config_manager()
    loaded_configs = {}
    
    broker_path = os.path.join(current_app.root_path, broker_directory)
    # List all items in broker directory and filter out __pycache__ and non-directories
    broker_names = [d for d in os.listdir(broker_path)
                    if os.path.isdir(os.path.join(broker_path, d)) and d != '__pycache__']

    for broker_name in broker_names:
        try:
            config = config_manager.load_broker_config(broker_name)
            if config:
                loaded_configs[broker_name] = config
                logger.info(f"Loaded configuration for broker: {broker_name}")
            else:
                logger.warning(f"No configuration found for broker: {broker_name}")
        except Exception as e:
            logger.error(f"Failed to load configuration for broker {broker_name}: {e}")

    return loaded_configs

def load_broker_data_classes(broker_directory='broker'):
    """Load data classes for all available brokers."""
    data_classes = {}
    broker_path = os.path.join(current_app.root_path, broker_directory)
    broker_names = [d for d in os.listdir(broker_path)
                    if os.path.isdir(os.path.join(broker_path, d)) and d != '__pycache__']

    for broker_name in broker_names:
        try:
            # Construct module name and import the module
            module_name = f"{broker_directory}.{broker_name}.api.data"
            data_module = importlib.import_module(module_name)
            # Retrieve the BrokerData class
            data_class = getattr(data_module, 'BrokerData', None)
            if data_class:
                data_classes[broker_name] = data_class
        except ImportError as e:
            logger.error(f"Failed to import data module for broker {broker_name}: {e}")
        except AttributeError as e:
            logger.error(f"BrokerData class not found in broker {broker_name}: {e}")

    return data_classes

def load_broker_websocket_adapters(broker_directory='broker'):
    """Load WebSocket adapters for all available brokers."""
    adapters = {}
    broker_path = os.path.join(current_app.root_path, broker_directory)
    broker_names = [d for d in os.listdir(broker_path)
                    if os.path.isdir(os.path.join(broker_path, d)) and d != '__pycache__']

    for broker_name in broker_names:
        try:
            # Check if streaming module exists
            streaming_path = os.path.join(broker_path, broker_name, 'streaming')
            if os.path.exists(streaming_path):
                # Try to import the adapter
                module_name = f"{broker_directory}.{broker_name}.streaming.{broker_name}_adapter"
                adapter_module = importlib.import_module(module_name)
                # Look for WebSocket adapter class
                adapter_class_name = f"{broker_name.title()}WebSocketAdapter"
                adapter_class = getattr(adapter_module, adapter_class_name, None)
                if adapter_class:
                    adapters[broker_name] = adapter_class
                    logger.info(f"Loaded WebSocket adapter for broker: {broker_name}")
        except ImportError as e:
            logger.debug(f"No WebSocket adapter found for broker {broker_name}: {e}")
        except AttributeError as e:
            logger.debug(f"WebSocket adapter class not found for broker {broker_name}: {e}")

    return adapters
