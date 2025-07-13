# api/funds.py

import os
import httpx
import json
from utils.httpx_client import get_httpx_client
from utils.logging import get_logger
from utils.broker_config import get_broker_config_manager

logger = get_logger(__name__)


def get_margin_data(auth_token):
    """Fetch margin data from the broker's API using the provided auth token."""
    api_key = os.getenv('BROKER_API_KEY')
    
    # Get broker configuration
    config_manager = get_broker_config_manager()
    broker_config = config_manager.get_broker_config('angel')
    
    if not broker_config:
        logger.error("Angel broker configuration not found")
        return None
    
    api_config = broker_config.get('api_config', {})
    
    # Get the shared httpx client with connection pooling
    client = get_httpx_client()
    
    # Get headers from configuration
    headers = api_config.get('headers', {}).copy()
    headers.update({
        'Authorization': f'Bearer {auth_token}',
        'X-PrivateKey': api_key
    })
    
    # Get funds endpoint URL from configuration
    base_url = api_config.get('base_url', 'https://apiconnect.angelbroking.com')
    funds_endpoint = api_config.get('endpoints', {}).get('funds', '/rest/secure/angelbroking/user/v1/getRMS')
    funds_url = f"{base_url}{funds_endpoint}"
    
    response = client.get(
        funds_url,
        headers=headers
    )
    
    # Add status attribute for compatibility with the existing codebase
    response.status = response.status_code
    
    margin_data = json.loads(response.text)

    logger.info(f"Margin Data: {margin_data}")

    if margin_data.get('data'):
        required_keys = [
            "availablecash", 
            "collateral", 
            "m2mrealized", 
            "m2munrealized", 
            "utiliseddebits"
        ]
        filtered_data = {}
        for key in required_keys:
            value = margin_data['data'].get(key, 0)
            try:
                formatted_value = "{:.2f}".format(float(value))
            except (ValueError, TypeError):
                formatted_value = value
            filtered_data[key] = formatted_value
        return filtered_data
    else:
        return {}
