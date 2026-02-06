
import os
import requests
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class ProxyManager:
    """
    Manages dynamic proxy configuration.
    
    Tries to connect via proxy defined in environment variables.
    If that fails, tries direct connection.
    Sets os.environ accordingly.
    """
    
    CHECK_URL = "https://huggingface.co"
    TIMEOUT = 5 # seconds
    
    @staticmethod
    def get_proxy_settings() -> Dict[str, str]:
        """Get proxy settings from environment variables."""
        proxies = {}
        # Check common proxy environment variables
        for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            if os.environ.get(key):
                proxies[key] = os.environ[key]
        return proxies

    @staticmethod
    def test_connection(proxies: Optional[Dict[str, str]] = None) -> bool:
        """
        Test connectivity to target URL.
        
        Args:
            proxies: Dictionary of proxy settings to use (or None for direct)
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # We use a simple lightweight request
            requests.head(
                ProxyManager.CHECK_URL, 
                proxies=proxies, 
                timeout=ProxyManager.TIMEOUT,
                allow_redirects=True
            )
            return True
        except Exception as e:
            # logger.debug(f"Connection test failed (proxies={proxies is not None}): {str(e)}")
            return False

    @classmethod
    def configure(cls, verbose: bool = True):
        """
        Auto-configure proxy settings.
        
        Logic:
        1. Capture current proxy env vars
        2. If proxies exist: Test WITH proxies -> If fail, Test WITHOUT -> If good, Unset env
        3. If no proxies: Test direct -> If fail, Warn
        """
        # 1. Get current settings
        initial_proxies = cls.get_proxy_settings()
        
        # Helper to clear all proxy vars
        def clear_proxy_env():
            keys_to_clear = [
                'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
                'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy'
            ]
            for key in keys_to_clear:
                if key in os.environ:
                    del os.environ[key]
        
        if not initial_proxies:
            # Already direct
            if verbose: print("No proxy settings found. Using direct connection.")
            return

        # 2. Proxies are set, test them
        if verbose: print(f"Checking configured proxy...")
        if cls.test_connection(initial_proxies):
            if verbose: print(f"✓ Proxy connection successful.")
            return
        
        # 3. Proxy failed, try direct
        if verbose: print(f"✗ Proxy connection failed. Attempting direct connection...")
        if cls.test_connection(None):
            if verbose: print(f"✓ Direct connection successful. Clearing proxy settings.")
            clear_proxy_env()
        else:
            print(f"⚠️ Warning: Connection failed both with proxy and direct.")
            # We default to clearing them if they are broken, or keeping them?
            # Safest to keep them if both fail (maybe network down), but user asked to fix "messy proxy".
            # If direct failed too, network might be down.
            pass
