
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
        2. Test connectivity WITH proxies
        3. If fail -> Test connectivity WITHOUT proxies
        4. If fail -> Warn user
        5. Set os.environ to the working configuration
        """
        # 1. Get current settings
        initial_proxies = cls.get_proxy_settings()
        
        # If no proxies set, just test direct connection
        if not initial_proxies:
            if verbose:
                print("No proxy settings found in environment. Testing direct connection...")
            if cls.test_connection(None):
                if verbose: print(f"✓ Direct connection to {cls.CHECK_URL} successful.")
                return
            else:
                print(f"⚠️ Warning: Direct connection to {cls.CHECK_URL} failed.")
                return

        # 2. Test WITH proxies
        if verbose:
            print(f"Testing connectivity with configured proxy...")
        
        if cls.test_connection(initial_proxies):
            if verbose:
                print(f"✓ Proxy connection successful.")
            # Ensure they remain set (they are already in os.environ)
            return
        
        # 3. If proxy failed, test WITHOUT proxies
        if verbose:
            print(f"✗ Proxy connection failed. Attempting direct connection...")
        
        if cls.test_connection(None):
            if verbose:
                print(f"✓ Direct connection successful. Unsetting proxy environment variables.")
            
            # Unset variables in os.environ
            for key in initial_proxies:
                if key in os.environ:
                    del os.environ[key]
            
            # Also clear from python's request session if needed, but os.environ is main one
        else:
            print(f"⚠️ Warning: Both proxy and direct connections to {cls.CHECK_URL} failed.")
            print("  Retaining original proxy settings.")
