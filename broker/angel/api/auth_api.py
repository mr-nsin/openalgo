import httpx
import json
import os
from utils.httpx_client import get_httpx_client
from utils.broker_config import get_broker_config_manager

def authenticate_broker(clientcode, broker_pin, totp_code):
    """
    Authenticate with the broker and return the auth token.
    """
    # Get broker configuration
    config_manager = get_broker_config_manager()
    broker_config = config_manager.get_broker_config('angel')
    
    if not broker_config:
        return None, None, "Broker configuration not found"
    
    api_key = os.getenv('BROKER_API_KEY')
    api_config = broker_config.get('api_config', {})
    
    try:
        # Get the shared httpx client
        client = get_httpx_client()
        
        payload = json.dumps({
            "clientcode": clientcode,
            "password": broker_pin,
            "totp": totp_code
        })
        
        # Get headers from configuration
        headers = api_config.get('headers', {}).copy()
        headers['X-PrivateKey'] = api_key
        
        # Get auth endpoint URL from configuration
        base_url = api_config.get('base_url', 'https://apiconnect.angelbroking.com')
        auth_endpoint = api_config.get('endpoints', {}).get('auth', '/rest/auth/angelbroking/user/v1/loginByPassword')
        auth_url = f"{base_url}{auth_endpoint}"

        response = client.post(
            auth_url,
            headers=headers,
            content=payload
        )
        
        # Add status attribute for compatibility with the existing codebase
        response.status = response.status_code
        
        data = response.text
        data_dict = json.loads(data)

        if 'data' in data_dict and 'jwtToken' in data_dict['data']:
            # Return both JWT token and feed token if available (None if not)
            auth_token = data_dict['data']['jwtToken']
            feed_token = data_dict['data'].get('feedToken', None)
            return auth_token, feed_token, None
        else:
            return None, None, data_dict.get('message', 'Authentication failed. Please try again.')
    except Exception as e:
        return None, None, str(e)
