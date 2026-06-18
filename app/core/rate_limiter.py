# app/core/rate_limiter.py
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def check_rate_limit(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """
        Check if request is within rate limit
        Returns True if allowed, False if rate limited
        """
        async with self.lock:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=window_seconds)
            
            # Clean old requests
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if req_time > window_start
            ]
            
            # Check if within limit
            if len(self.requests[key]) >= max_requests:
                return False
            
            # Add current request
            self.requests[key].append(now)
            return True
    
    def clear(self, key: str = None):
        """Clear rate limit data for a specific key or all keys"""
        if key:
            self.requests.pop(key, None)
        else:
            self.requests.clear()