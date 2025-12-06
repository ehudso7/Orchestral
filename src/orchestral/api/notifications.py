"""
Notifications API endpoints for Orchestral.

Provides a comprehensive notification system with real-time updates.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from orchestral.api.auth import get_current_user
from orchestral.api.db import db

logger = structlog.get_logger()

router = APIRouter(prefix="/notifications", tags=["notifications"])


# ==================== DATA MODELS ====================

class NotificationType:
    """Notification type constants."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SYSTEM = "system"
    BILLING = "billing"
    API = "api"
    SECURITY = "security"


class NotificationPriority:
    """Notification priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Notification(BaseModel):
    """Notification model."""
    id: str
    user_id: str
    type: str = Field(default=NotificationType.INFO)
    priority: str = Field(default=NotificationPriority.MEDIUM)
    title: str
    message: str
    link: str | None = None
    icon: str = Field(default="fas fa-info-circle")
    read: bool = Field(default=False)
    created_at: str
    expires_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NotificationResponse(BaseModel):
    """Response for notification queries."""
    notifications: list[Notification]
    unread_count: int
    total_count: int


class CreateNotification(BaseModel):
    """Request to create a notification."""
    type: str = Field(default=NotificationType.INFO)
    priority: str = Field(default=NotificationPriority.MEDIUM)
    title: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=500)
    link: str | None = None
    icon: str | None = None
    expires_in_hours: int | None = None


# ==================== HELPER FUNCTIONS ====================

def get_user_notifications(user_id: str) -> list[Notification]:
    """Get all notifications for a user."""
    notifications_data = db._load_data().get("notifications", {})
    user_notifications = []

    current_time = datetime.now(timezone.utc)

    for notif_id, notif in notifications_data.items():
        if notif.get("user_id") == user_id:
            # Check if notification has expired
            if notif.get("expires_at"):
                expires_at = datetime.fromisoformat(notif["expires_at"])
                if expires_at < current_time:
                    continue  # Skip expired notifications

            user_notifications.append(Notification(**notif))

    # Sort by created_at, newest first
    user_notifications.sort(key=lambda x: x.created_at, reverse=True)

    return user_notifications


def save_notification(notification: Notification) -> None:
    """Save a notification to the database."""
    data = db._load_data()
    if "notifications" not in data:
        data["notifications"] = {}

    data["notifications"][notification.id] = notification.model_dump()
    db._save_data(data)


def delete_notification(notif_id: str, user_id: str) -> bool:
    """Delete a notification."""
    data = db._load_data()
    if "notifications" in data and notif_id in data["notifications"]:
        if data["notifications"][notif_id].get("user_id") == user_id:
            del data["notifications"][notif_id]
            db._save_data(data)
            return True
    return False


def mark_notification_read(notif_id: str, user_id: str) -> bool:
    """Mark a notification as read."""
    data = db._load_data()
    if "notifications" in data and notif_id in data["notifications"]:
        if data["notifications"][notif_id].get("user_id") == user_id:
            data["notifications"][notif_id]["read"] = True
            db._save_data(data)
            return True
    return False


def create_system_notification(
    user_id: str,
    title: str,
    message: str,
    type: str = NotificationType.SYSTEM,
    priority: str = NotificationPriority.MEDIUM,
    **kwargs
) -> Notification:
    """Create a system-generated notification."""
    notification = Notification(
        id=f"notif_{secrets.token_urlsafe(16)}",
        user_id=user_id,
        type=type,
        priority=priority,
        title=title,
        message=message,
        created_at=datetime.now(timezone.utc).isoformat(),
        **kwargs
    )
    save_notification(notification)
    return notification


# ==================== API ENDPOINTS ====================

@router.get("", response_model=NotificationResponse)
async def get_notifications(
    unread_only: bool = Query(False, description="Return only unread notifications"),
    notification_type: str | None = Query(None, description="Filter by notification type"),
    priority: str | None = Query(None, description="Filter by priority"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of notifications to return"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Get user notifications.

    Returns a list of notifications with filtering options.
    """
    try:
        notifications = get_user_notifications(current_user["id"])

        # Apply filters
        if unread_only:
            notifications = [n for n in notifications if not n.read]

        if notification_type:
            notifications = [n for n in notifications if n.type == notification_type]

        if priority:
            notifications = [n for n in notifications if n.priority == priority]

        # Calculate counts before limiting
        total_count = len(notifications)
        unread_count = len([n for n in notifications if not n.read])

        # Apply limit
        notifications = notifications[:limit]

        return NotificationResponse(
            notifications=notifications,
            unread_count=unread_count,
            total_count=total_count,
        )

    except Exception as e:
        logger.error(f"Failed to get notifications: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve notifications",
        )


@router.get("/unread-count")
async def get_unread_count(
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get count of unread notifications."""
    try:
        notifications = get_user_notifications(current_user["id"])
        unread_count = len([n for n in notifications if not n.read])

        return {"unread_count": unread_count}

    except Exception as e:
        logger.error(f"Failed to get unread count: {e}")
        return {"unread_count": 0}


@router.post("/{notification_id}/read")
async def mark_as_read(
    notification_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Mark a notification as read."""
    if mark_notification_read(notification_id, current_user["id"]):
        return {"message": "Notification marked as read"}

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Notification not found",
    )


@router.post("/mark-all-read")
async def mark_all_as_read(
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Mark all notifications as read."""
    try:
        data = db._load_data()
        if "notifications" in data:
            count = 0
            for notif_id, notif in data["notifications"].items():
                if notif.get("user_id") == current_user["id"] and not notif.get("read"):
                    data["notifications"][notif_id]["read"] = True
                    count += 1

            if count > 0:
                db._save_data(data)

            return {"message": f"Marked {count} notifications as read"}

        return {"message": "No notifications to mark as read"}

    except Exception as e:
        logger.error(f"Failed to mark all as read: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark notifications as read",
        )


@router.delete("/{notification_id}")
async def delete_notification_endpoint(
    notification_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Delete a notification."""
    if delete_notification(notification_id, current_user["id"]):
        return {"message": "Notification deleted"}

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Notification not found",
    )


@router.delete("/clear-all")
async def clear_all_notifications(
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Clear all notifications for the current user."""
    try:
        data = db._load_data()
        if "notifications" in data:
            # Find and remove all user's notifications
            to_delete = []
            for notif_id, notif in data["notifications"].items():
                if notif.get("user_id") == current_user["id"]:
                    to_delete.append(notif_id)

            for notif_id in to_delete:
                del data["notifications"][notif_id]

            if to_delete:
                db._save_data(data)

            return {"message": f"Cleared {len(to_delete)} notifications"}

        return {"message": "No notifications to clear"}

    except Exception as e:
        logger.error(f"Failed to clear notifications: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear notifications",
        )


@router.post("/test", include_in_schema=False)
async def create_test_notification(
    request: CreateNotification,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Create a test notification (for development)."""
    expires_at = None
    if request.expires_in_hours:
        from datetime import timedelta
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=request.expires_in_hours)).isoformat()

    # Determine icon based on type if not provided
    if not request.icon:
        icon_map = {
            NotificationType.INFO: "fas fa-info-circle",
            NotificationType.SUCCESS: "fas fa-check-circle",
            NotificationType.WARNING: "fas fa-exclamation-triangle",
            NotificationType.ERROR: "fas fa-times-circle",
            NotificationType.SYSTEM: "fas fa-cog",
            NotificationType.BILLING: "fas fa-credit-card",
            NotificationType.API: "fas fa-key",
            NotificationType.SECURITY: "fas fa-shield-alt",
        }
        icon = icon_map.get(request.type, "fas fa-bell")
    else:
        icon = request.icon

    notification = Notification(
        id=f"notif_{secrets.token_urlsafe(16)}",
        user_id=current_user["id"],
        type=request.type,
        priority=request.priority,
        title=request.title,
        message=request.message,
        link=request.link,
        icon=icon,
        read=False,
        created_at=datetime.now(timezone.utc).isoformat(),
        expires_at=expires_at,
    )

    save_notification(notification)

    return {"message": "Test notification created", "notification": notification}


# ==================== SYSTEM NOTIFICATION FUNCTIONS ====================

def notify_api_key_created(user_id: str, key_name: str):
    """Send notification when API key is created."""
    create_system_notification(
        user_id=user_id,
        title="API Key Created",
        message=f"New API key '{key_name}' has been created for your account.",
        type=NotificationType.API,
        priority=NotificationPriority.MEDIUM,
        icon="fas fa-key",
        link="/dashboard#api-keys",
    )


def notify_api_key_revoked(user_id: str, key_name: str):
    """Send notification when API key is revoked."""
    create_system_notification(
        user_id=user_id,
        title="API Key Revoked",
        message=f"API key '{key_name}' has been revoked and is no longer valid.",
        type=NotificationType.API,
        priority=NotificationPriority.HIGH,
        icon="fas fa-key",
        link="/dashboard#api-keys",
    )


def notify_subscription_changed(user_id: str, new_plan: str):
    """Send notification when subscription changes."""
    create_system_notification(
        user_id=user_id,
        title="Subscription Updated",
        message=f"Your subscription has been updated to the {new_plan} plan.",
        type=NotificationType.BILLING,
        priority=NotificationPriority.HIGH,
        icon="fas fa-credit-card",
        link="/dashboard#billing",
    )


def notify_usage_limit_warning(user_id: str, percent_used: int):
    """Send notification when approaching usage limits."""
    create_system_notification(
        user_id=user_id,
        title="Usage Limit Warning",
        message=f"You've used {percent_used}% of your monthly API quota. Consider upgrading for more capacity.",
        type=NotificationType.WARNING,
        priority=NotificationPriority.MEDIUM if percent_used < 90 else NotificationPriority.HIGH,
        icon="fas fa-exclamation-triangle",
        link="/dashboard#usage",
    )


def notify_security_event(user_id: str, event: str):
    """Send notification for security events."""
    create_system_notification(
        user_id=user_id,
        title="Security Alert",
        message=event,
        type=NotificationType.SECURITY,
        priority=NotificationPriority.URGENT,
        icon="fas fa-shield-alt",
        link="/dashboard#settings",
    )


def notify_welcome(user_id: str, user_name: str):
    """Send welcome notification to new users."""
    create_system_notification(
        user_id=user_id,
        title="Welcome to Orchestral!",
        message=f"Hello {user_name}! Your account has been successfully created. Explore the dashboard to get started with our AI orchestration platform.",
        type=NotificationType.SUCCESS,
        priority=NotificationPriority.LOW,
        icon="fas fa-rocket",
        link="/dashboard",
    )