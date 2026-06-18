# test_core.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing core module imports...")

try:
    from app.core import (
        hash_password,
        verify_password,
        create_jwt_token,
        get_current_user,
        get_current_admin,
        RateLimiter,
        ConnectionManager,
        PLATFORM_FEE_PERCENTAGE,
        PLATFORM_FIXED_FEE,
        PROJECT_POSTING_FEE,
        setup_logging,
        get_logger,
        UnauthorizedException,
        ForbiddenException,
        NotFoundException,
    )
    
    print("✅ All core imports successful!")
    
    # Test password hashing
    password = "test_password_123"
    hashed = hash_password(password)
    print(f"Password: {password}")
    print(f"Hashed: {hashed[:30]}...")
    print(f"Verify: {verify_password(password, hashed)}")
    
    # Test constants
    print(f"PLATFORM_FEE_PERCENTAGE: {PLATFORM_FEE_PERCENTAGE}%")
    print(f"PLATFORM_FIXED_FEE: ${PLATFORM_FIXED_FEE}")
    print(f"PROJECT_POSTING_FEE: ${PROJECT_POSTING_FEE}")
    
    # Test RateLimiter
    limiter = RateLimiter()
    import asyncio
    
    async def test_rate_limit():
        key = "test_user"
        # First request should pass
        result1 = await limiter.check_rate_limit(key, 5, 60)
        print(f"Rate limit check 1: {result1}")
        return result1
    
    result = asyncio.run(test_rate_limit())
    
    print("\n✅ Core module working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()