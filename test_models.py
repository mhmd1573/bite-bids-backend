# test_models.py
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Try importing from the new structure
    from app.models import (
        Base, User, Project, Bid, ChatRoom, ChatMessage,
        CheckoutSession, DeveloperPayout, ProjectDisputeSimple,
        Notification, ProjectGithubRepo, ProjectUpload, ContactFormRecord
    )
    
    print("✅ All models imported successfully!")
    print(f"Base: {Base}")
    print(f"User: {User}")
    print(f"Project: {Project}")
    print(f"Number of models: {len([m for m in dir() if not m.startswith('_')])}")
    
    # Check if we can create instances
    print("\n✅ Testing model instantiation...")
    user = User(name="Test User", email="test@example.com", role="developer")
    print(f"User instance: {user}")
    
    print("\n✅ All models are working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")