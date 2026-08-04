# app/api/v1/bids.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import uuid
from datetime import datetime

from app.database import get_db
from app.models.project import Project
from app.models.bid import Bid
from app.models.user import User
from app.schemas.bid import BidCreate
from app.core.dependencies import get_current_user
from app.core.exceptions import ForbiddenException, NotFoundException
from app.utils.converters import model_to_dict
from app.services.notification_service import NotificationService

router = APIRouter(prefix="/bids", tags=["Bids"])


@router.post("/projects/{project_id}/bids")
async def create_bid(
    project_id: str,
    bid_data: BidCreate,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Place a bid on a project"""
    
    if current_user["role"] != "investor":
        raise ForbiddenException("Only investors can place bids")
    
    project = await db.scalar(select(Project).where(Project.id == uuid.UUID(project_id)))
    if not project:
        raise NotFoundException("Project not found")
    
    # Check if user already placed a bid
    existing_bid = await db.scalar(
        select(Bid).where(
            Bid.project_id == project.id,
            Bid.investor_id == current_user["id"]
        )
    )
    if existing_bid:
        raise HTTPException(status_code=400, detail="You already placed a bid")
    
    new_bid = Bid(
        project_id=project.id,
        investor_id=current_user["id"],
        amount=bid_data.amount,
        status="pending"
    )
    
    db.add(new_bid)
    await db.commit()
    await db.refresh(new_bid)
    
    # Send notification to project owner
    formatted_amount = f"${bid_data.amount:,.2f}"
    
    await NotificationService.send_notification_to_user(
        str(project.developer_id),
        {
            "type": "bid_received",
            "title": f"💰 New Bid: {formatted_amount}",
            "message": f"Investor bid {formatted_amount} on '{project.title[:50]}'",
            "link": f"/project/{project_id}/bids",
            "details": {
                "bid_id": str(new_bid.id),
                "project_id": project_id,
                "amount": float(bid_data.amount)
            }
        },
        db
    )
    
    return model_to_dict(new_bid)


@router.put("/{bid_id}/accept")
async def accept_bid(
    bid_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Accept a bid"""
    
    bid = await db.scalar(select(Bid).where(Bid.id == uuid.UUID(bid_id)))
    if not bid:
        raise NotFoundException("Bid not found")
    
    project = await db.scalar(select(Project).where(Project.id == bid.project_id))
    
    if str(project.developer_id) != str(current_user["id"]):
        raise ForbiddenException("Only the project owner can accept bids")
    
    bid.status = "accepted"
    bid.accepted_at = datetime.utcnow()
    
    # Update project stats
    amount = float(bid.amount)
    project.highest_bid = max(amount, project.highest_bid or amount)
    project.lowest_bid = min(amount, project.lowest_bid or amount)
    project.bids_count = (project.bids_count or 0) + 1
    
    await db.commit()
    
    # Send notification to investor
    formatted_amount = f"${float(bid.amount):,.2f}"
    
    await NotificationService.send_notification_to_user(
        str(bid.investor_id),
        {
            "type": "bid_accepted",
            "title": f"🎉 Bid Accepted!",
            "message": f"Your {formatted_amount} bid on '{project.title[:50]}' was accepted by developer.",
            "link": f"/project/{project.id}",
            "details": {
                "project_id": str(project.id),
                "bid_id": bid_id,
                "amount": float(bid.amount)
            }
        },
        db
    )
    
    return {"message": "Bid accepted"}


@router.put("/{bid_id}/reject")
async def reject_bid(
    bid_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Reject a bid"""
    
    bid = await db.scalar(select(Bid).where(Bid.id == uuid.UUID(bid_id)))
    if not bid:
        raise NotFoundException("Bid not found")
    
    project = await db.scalar(select(Project).where(Project.id == bid.project_id))
    
    if str(project.developer_id) != str(current_user["id"]):
        raise ForbiddenException("Unauthorized")
    
    bid.status = "rejected"
    await db.commit()
    
    # Send notification to investor
    await NotificationService.send_notification_to_user(
        str(bid.investor_id),
        {
            "type": "bid_rejected",
            "title": "❌ Bid Not Selected",
            "message": f"Your bid on '{project.title[:50]}' wasn't selected this time.",
            "details": {
                "project_id": str(project.id),
                "bid_id": bid_id
            }
        },
        db
    )
    
    return {
        "message": "Bid rejected successfully",
        "bid_id": str(bid.id),
        "status": "rejected"
    }

@router.put("/projects/{project_id}/close-bidding")
async def close_bidding(
    project_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Close bidding on a project and select the winner.
    Only the project owner (developer) can close bidding.
    The highest accepted bid wins.
    """
    # Get project
    project = await db.scalar(select(Project).where(Project.id == uuid.UUID(project_id)))
    if not project:
        raise NotFoundException("Project not found")
    
    # Only project owner can close bidding
    if str(project.developer_id) != str(current_user["id"]):
        raise ForbiddenException("Only the project owner can close bidding")
    
    # Check if project is open for bidding
    if project.status not in ["open", "bidding"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot close bidding. Project status is '{project.status}'"
        )
    
    # Get accepted bids ordered highest first
    accepted_bids = await db.scalars(
        select(Bid).where(
            Bid.project_id == project.id,
            Bid.status == "accepted"
        ).order_by(Bid.amount.desc())
    )
    bids = list(accepted_bids)
    
    if not bids:
        raise HTTPException(
            status_code=400, 
            detail="No accepted bids to finalize"
        )
    
    # The highest accepted bid wins
    winner = bids[0]
    
    # Update project
    project.status = "winner_selected"
    project.assigned_to = winner.investor_id
    project.highest_bid = float(winner.amount)
    project.updated_at = datetime.utcnow()
    
    # Reject all other bids
    await db.execute(
        update(Bid)
        .where(
            Bid.project_id == project.id,
            Bid.id != winner.id,
            Bid.status == "accepted"
        )
        .values(status="rejected")
    )
    await db.commit()
    
    # Fetch winner's user profile for notification
    winner_user = await db.scalar(
        select(User).where(User.id == winner.investor_id)
    )
    
    # Send notification to winner
    formatted_amount = f"${float(winner.amount):,.2f}"
    
    await NotificationService.send_notification_to_user(
        str(winner.investor_id),
        {
            "type": "payment_required",
            "title": "🎉 Congratulations! You Won the Project!",
            "message": (
                f"Your bid of {formatted_amount} on '{project.title}' has been selected! "
                f"Please complete the payment to start the project."
            ),
            "link": f"/project/{project_id}/payment",
            "details": {
                "bid_id": str(winner.id),
                "project_id": str(project.id),
                "project_title": project.title,
                "amount": float(winner.amount),
                "bid_amount": float(winner.amount),
                "investor_id": str(winner.investor_id),
                "developer_id": str(project.developer_id),
                "payment_type": "project_winner",
                "action_required": "payment"
            }
        },
        db
    )
    
    # Send notification to all rejected bidders
    rejected_bids = [b for b in bids if b.id != winner.id]
    for bid in rejected_bids:
        await NotificationService.send_notification_to_user(
            str(bid.investor_id),
            {
                "type": "bid_rejected",
                "title": "📢 Project Bidding Closed",
                "message": f"Bidding on '{project.title}' has closed. Unfortunately, your bid wasn't selected this time.",
                "link": f"/project/{project_id}",
                "details": {
                    "project_id": str(project.id),
                    "bid_id": str(bid.id)
                }
            },
            db
        )
    
    return {
        "message": "Bidding closed successfully",
        "winner": str(winner.investor_id),
        "winner_name": winner_user.name if winner_user else None,
        "winner_amount": float(winner.amount),
        "total_bids": len(bids),
        "rejected_count": len(rejected_bids),
        "notification_sent": True
    }


@router.get("/projects/{project_id}/bids")
async def get_project_bids(
    project_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get all bids for a project"""
    
    bids = await db.scalars(
        select(Bid)
        .where(Bid.project_id == uuid.UUID(project_id))
        .order_by(Bid.created_at.desc())
    )
    return models_to_list(bids)


@router.get("/investor/{investor_id}")
async def get_investor_bids(
    investor_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get all bids by an investor"""
    
    bids = await db.scalars(
        select(Bid)
        .where(Bid.investor_id == uuid.UUID(investor_id))
        .order_by(Bid.created_at.desc())
    )
    return models_to_list(bids)