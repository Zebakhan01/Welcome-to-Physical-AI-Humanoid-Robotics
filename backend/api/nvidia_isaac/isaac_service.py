"""
Main NVIDIA Isaac service orchestrator
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
import logging

from backend.api.nvidia_isaac.models import IsaacAppConfig, IsaacAppStatus
from backend.api.nvidia_isaac.isaac_sim_service import sim_router, IsaacSimService
from backend.api.nvidia_isaac.isaac_ros_bridge_service import ros_bridge_router, IsaacROSBridgeService
from backend.api.nvidia_isaac.isaac_ai_service import ai_router, IsaacAIService
from backend.utils.logger import get_logger

router = APIRouter(prefix="", tags=["nvidia-isaac"])

# Include sub-routers
router.include_router(sim_router)
router.include_router(ros_bridge_router)
router.include_router(ai_router)

# Initialize services
sim_service = IsaacSimService()
ros_bridge_service = IsaacROSBridgeService()
ai_service = IsaacAIService()
logger = get_logger(__name__)

# App lifecycle management
active_apps: Dict[str, Any] = {}

@router.post("/apps/launch", response_model=IsaacAppStatus)
async def launch_isaac_app(config: IsaacAppConfig):
    """Launch a new Isaac application"""
    try:
        # Create and store the application instance
        app_instance = {
            'config': config,
            'sim_service': sim_service,
            'ros_bridge_service': ros_bridge_service,
            'ai_service': ai_service,
            'status': 'running',
            'start_time': __import__('time').time()
        }

        active_apps[config.app_name] = app_instance

        logger.info(f"Launched Isaac app: {config.app_name}")

        return IsaacAppStatus(
            app_name=config.app_name,
            is_running=True,
            status_message="Application launched successfully",
            start_time=app_instance['start_time']
        )
    except Exception as e:
        logger.error(f"Error launching Isaac app {config.app_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to launch app: {str(e)}")


@router.post("/apps/stop")
async def stop_isaac_app(app_name: str):
    """Stop a running Isaac application"""
    try:
        if app_name in active_apps:
            # Perform cleanup operations
            app_instance = active_apps[app_name]
            # In a real implementation, we would properly shut down the Isaac app
            del active_apps[app_name]

            logger.info(f"Stopped Isaac app: {app_name}")
            return {"message": f"Application {app_name} stopped successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Application {app_name} not found")
    except Exception as e:
        logger.error(f"Error stopping Isaac app {app_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop app: {str(e)}")


@router.get("/apps/status", response_model=Dict[str, IsaacAppStatus])
async def get_all_app_status():
    """Get status of all Isaac applications"""
    try:
        status_dict = {}
        for app_name, app_instance in active_apps.items():
            status_dict[app_name] = IsaacAppStatus(
                app_name=app_name,
                is_running=app_instance['status'] == 'running',
                pid=None,  # In real implementation, this would be the actual process ID
                status_message=app_instance['status'],
                start_time=app_instance.get('start_time')
            )

        return status_dict
    except Exception as e:
        logger.error(f"Error getting app status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get app status: {str(e)}")


@router.get("/health")
async def isaac_health_check():
    """Health check for Isaac services"""
    try:
        # Check if services are available (in a real implementation, this would check actual Isaac connection)
        health_status = {
            "sim_service": "available" if sim_service.is_available() else "unavailable",
            "ros_bridge_service": "available" if ros_bridge_service.is_available() else "unavailable",
            "ai_service": "available" if ai_service.is_available() else "unavailable",
            "overall_status": "healthy" if all([
                sim_service.is_available(),
                ros_bridge_service.is_available(),
                ai_service.is_available()
            ]) else "degraded"
        }
        return health_status
    except Exception as e:
        logger.error(f"Error in Isaac health check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")