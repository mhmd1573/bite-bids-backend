# app/utils/converters.py
import uuid
from decimal import Decimal
from datetime import datetime
from typing import Any, Dict, List, Optional


def model_to_dict(model_instance) -> Optional[Dict[str, Any]]:
    """Convert SQLAlchemy model instance to dictionary"""
    if model_instance is None:
        return None
    
    result = {}
    for column in model_instance.__table__.columns:
        value = getattr(model_instance, column.name)
        
        # Handle UUID
        if isinstance(value, uuid.UUID):
            result[column.name] = str(value)
        # Handle asyncpg UUID
        elif hasattr(value, '__class__') and 'pgproto.UUID' in str(value.__class__):
            result[column.name] = str(value)
        # Handle datetime
        elif isinstance(value, datetime):
            result[column.name] = value.isoformat() + 'Z'
        # Handle Decimal
        elif isinstance(value, Decimal):
            result[column.name] = float(value)
        else:
            result[column.name] = value
    
    # Map 'id' to '_id' for backward compatibility with frontend
    if 'id' in result:
        result['_id'] = result['id']
    
    return result


def models_to_list(model_instances) -> List[Dict[str, Any]]:
    """Convert list of SQLAlchemy model instances to list of dictionaries"""
    return [model_to_dict(instance) for instance in model_instances]