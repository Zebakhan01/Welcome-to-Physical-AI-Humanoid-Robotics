from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import API routers
from backend.api.chat.chat_routes import router as chat_router
from backend.api.chat.message_processing import router as message_router
from backend.api.rag.rag_endpoints import router as rag_router
from backend.api.auth.auth_routes import router as auth_router
from backend.api.content.content_parser import router as content_router
from backend.api.content.content_loader import router as content_loader_router
from backend.api.content.metadata_service import router as metadata_router
from backend.api.personalization.user_service import router as personalization_router
from backend.api.robotics.kinematics_service import router as kinematics_router
from backend.api.robotics.dynamics_service import router as dynamics_router
from backend.api.sensors.sensor_processing import router as sensor_router
from backend.api.sensors.computer_vision import router as vision_router
from backend.api.motion_control.motion_control_service import router as motion_router
from backend.api.locomotion.locomotion_service import router as locomotion_router
from backend.api.manipulation.manipulation_service import router as manipulation_router
from backend.api.learning.learning_service import router as learning_router
from backend.api.vla.vla_service import router as vla_router
from backend.api.ros.ros_service import router as ros_router
from backend.api.simulation.simulation_service import router as simulation_router
from backend.api.hardware_integration.hardware_integration_service import router as hardware_integration_router
from backend.api.unity_integration.unity_integration_service import router as unity_integration_router
from backend.api.humanoid_architecture.humanoid_architecture_service import router as humanoid_architecture_router
from backend.api.project_integration.project_integration_service import router as project_integration_router
from backend.api.ros2.ros2_service import router as ros2_router
from backend.api.nvidia_isaac.isaac_service import router as isaac_router
from backend.config.rag_settings import settings

# Create FastAPI app
app = FastAPI(
    title="Physical AI & Humanoid Robotics Textbook API",
    description="Backend API for the Physical AI & Humanoid Robotics textbook with RAG chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(message_router, prefix="/api/chat", tags=["message-processing"])
app.include_router(rag_router, prefix="/api", tags=["rag"])
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(content_router, prefix="/api/content", tags=["content"])
app.include_router(content_loader_router, prefix="/api/content", tags=["content-loader"])
app.include_router(metadata_router, prefix="/api/content", tags=["metadata"])
app.include_router(kinematics_router, prefix="/api/robotics", tags=["kinematics"])
app.include_router(dynamics_router, prefix="/api/robotics", tags=["dynamics"])
app.include_router(sensor_router, prefix="/api/sensors", tags=["sensor-processing"])
app.include_router(vision_router, prefix="/api/sensors", tags=["computer-vision"])
app.include_router(motion_router, prefix="/api/motion", tags=["motion-control"])
app.include_router(locomotion_router, prefix="/api/locomotion", tags=["locomotion"])
app.include_router(manipulation_router, prefix="/api/manipulation", tags=["manipulation"])
app.include_router(learning_router, prefix="/api/learning", tags=["learning"])
app.include_router(vla_router, prefix="/api/vla", tags=["vla"])
app.include_router(ros_router, prefix="/api/ros", tags=["ros"])
app.include_router(simulation_router, prefix="/api/simulation", tags=["simulation"])
app.include_router(hardware_integration_router, prefix="/api/hardware", tags=["hardware-integration"])
app.include_router(humanoid_architecture_router, prefix="/api/humanoid", tags=["humanoid-architecture"])
app.include_router(project_integration_router, prefix="/api/integration", tags=["project-integration"])
app.include_router(ros2_router, prefix="/api/ros2", tags=["ros2"])
app.include_router(unity_integration_router, prefix="/api/unity", tags=["unity-integration"])
app.include_router(isaac_router, prefix="/api/isaac", tags=["nvidia-isaac"])

@app.get("/")
async def root():
    return {"message": "Physical AI & Humanoid Robotics Textbook API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
