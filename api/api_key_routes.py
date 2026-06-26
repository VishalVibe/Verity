import secrets
import hashlib
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime

from api.database import get_db
from api import models
from api.auth import get_current_user
from api.schemas import ApiKeyResponse, ApiKeyCreateResponse

router = APIRouter(prefix="/auth/keys", tags=["api_keys"])


@router.get("", response_model=list[ApiKeyResponse])
def list_keys(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return db.query(models.UserApiKey).filter(
        models.UserApiKey.user_id == current_user.id,
        models.UserApiKey.is_active == True
    ).order_by(models.UserApiKey.created_at.desc()).all()


@router.post("", response_model=ApiKeyCreateResponse)
def create_key(
    name_dict: dict,  # Receive body {"name": "..."}
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    name = name_dict.get("name", "Default API Key").strip()
    if not name:
        name = "Default API Key"

    # Generate cryptographically secure API key
    raw_key = f"vt_live_{secrets.token_hex(24)}"
    prefix = raw_key[:12]  # "vt_live_abcd"
    hashed = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    new_key = models.UserApiKey(
        user_id=current_user.id,
        key_hash=hashed,
        name=name,
        prefix=prefix,
        is_active=True
    )
    db.add(new_key)
    db.commit()
    db.refresh(new_key)

    return ApiKeyCreateResponse(
        api_key=raw_key,
        id=new_key.id,
        name=new_key.name,
        prefix=new_key.prefix,
        created_at=new_key.created_at
    )


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_key(
    key_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    key_record = db.query(models.UserApiKey).filter(
        models.UserApiKey.id == key_id,
        models.UserApiKey.user_id == current_user.id
    ).first()

    if not key_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API Key not found or does not belong to you"
        )

    # Soft delete / deactivate
    key_record.is_active = False
    db.commit()
    return None
