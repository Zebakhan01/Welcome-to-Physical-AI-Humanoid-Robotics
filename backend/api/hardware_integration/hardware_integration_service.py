from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
from backend.utils.hardware_integration_utils import (
    hardware_manager, CommunicationProtocol, HardwareComponentType,
    SafetyLevel, HardwareStatus, SensorData, ActuatorCommand,
    SerialInterface, NetworkInterface, SensorInterface, ActuatorInterface
)
from backend.utils.logger import logger


router = APIRouter()


class RegisterComponentRequest(BaseModel):
    component_id: str
    component_type: str  # sensor, actuator, controller
    protocol: str  # serial, usb, ethernet, can, etc.
    connection_params: Dict[str, Any]


class RegisterComponentResponse(BaseModel):
    success: bool
    message: str


class ConnectRequest(BaseModel):
    component_id: str


class ConnectResponse(BaseModel):
    success: bool
    message: str


class SendCommandRequest(BaseModel):
    component_id: str
    command: Dict[str, Any]


class SendCommandResponse(BaseModel):
    success: bool
    response: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class ReadDataResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class SafetyStatusResponse(BaseModel):
    safe: bool
    violations: List[Dict[str, str]]
    emergency_stop_active: bool
    success: bool


class ComponentStatusResponse(BaseModel):
    component_id: str
    status: str
    temperature: float
    voltage: float
    current: float
    last_update: str
    error_count: int
    safety_level: str
    success: bool


class CalibrationRequest(BaseModel):
    component_id: str
    calibration_data: Dict[str, Any]


class CalibrationResponse(BaseModel):
    success: bool
    message: str


class EmergencyStopRequest(BaseModel):
    activate: bool


class EmergencyStopResponse(BaseModel):
    success: bool
    message: str


@router.post("/register-component", response_model=RegisterComponentResponse)
async def register_component(request: RegisterComponentRequest):
    """
    Register a new hardware component
    """
    try:
        component_id = request.component_id
        component_type = request.component_type
        protocol = request.protocol
        params = request.connection_params

        # Create appropriate interface based on protocol
        interface = None
        if protocol.lower() == "serial":
            port = params.get("port", "/dev/ttyUSB0")
            baudrate = params.get("baudrate", 115200)
            interface = SerialInterface(component_id, port, baudrate)
        elif protocol.lower() in ["ethernet", "wifi"]:
            host = params.get("host", "localhost")
            port = params.get("port", 8080)
            interface = NetworkInterface(component_id, host, port)
        else:
            raise HTTPException(status_code=400, detail=f"Protocol {protocol} not supported")

        # Wrap in appropriate component type
        if component_type.lower() == "sensor":
            sensor_type = params.get("sensor_type", "generic")
            component = SensorInterface(component_id, sensor_type, interface)
        elif component_type.lower() == "actuator":
            actuator_type = params.get("actuator_type", "generic")
            component = ActuatorInterface(component_id, actuator_type, interface)
        elif component_type.lower() in ["controller", "processor"]:
            component = interface
        else:
            raise HTTPException(status_code=400, detail=f"Component type {component_type} not supported")

        # Register the component
        success = hardware_manager.register_component(component)

        if success:
            response = RegisterComponentResponse(
                success=True,
                message=f"Component {component_id} registered successfully"
            )
        else:
            response = RegisterComponentResponse(
                success=False,
                message=f"Component {component_id} already exists"
            )

        logger.info(f"Component registration: {component_id}, success: {success}")

        return response

    except Exception as e:
        logger.error(f"Error registering component: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error registering component: {str(e)}")


@router.post("/connect", response_model=ConnectResponse)
async def connect_component(request: ConnectRequest):
    """
    Connect to a registered hardware component
    """
    try:
        component_id = request.component_id

        component = hardware_manager.get_component(component_id)
        if not component:
            raise HTTPException(status_code=404, detail=f"Component {component_id} not found")

        success = await component.connect()

        if success:
            response = ConnectResponse(
                success=True,
                message=f"Component {component_id} connected successfully"
            )
        else:
            response = ConnectResponse(
                success=False,
                message=f"Failed to connect to component {component_id}"
            )

        logger.info(f"Component connection: {component_id}, success: {success}")

        return response

    except Exception as e:
        logger.error(f"Error connecting component: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error connecting component: {str(e)}")


@router.post("/disconnect", response_model=ConnectResponse)
async def disconnect_component(request: ConnectRequest):
    """
    Disconnect from a hardware component
    """
    try:
        component_id = request.component_id

        component = hardware_manager.get_component(component_id)
        if not component:
            raise HTTPException(status_code=404, detail=f"Component {component_id} not found")

        success = await component.disconnect()

        if success:
            response = ConnectResponse(
                success=True,
                message=f"Component {component_id} disconnected successfully"
            )
        else:
            response = ConnectResponse(
                success=False,
                message=f"Failed to disconnect component {component_id}"
            )

        logger.info(f"Component disconnection: {component_id}, success: {success}")

        return response

    except Exception as e:
        logger.error(f"Error disconnecting component: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error disconnecting component: {str(e)}")


@router.post("/send-command", response_model=SendCommandResponse)
async def send_command(request: SendCommandRequest):
    """
    Send a command to a hardware component
    """
    try:
        component_id = request.component_id
        command = request.command

        result = hardware_manager.execute_hardware_command(component_id, command)

        response = SendCommandResponse(
            success=result.get("success", False),
            response=result.get("data") if result.get("success") else None,
            message=result.get("message") if not result.get("success") else None
        )

        logger.info(f"Command sent to {component_id}, success: {result.get('success', False)}")

        return response

    except Exception as e:
        logger.error(f"Error sending command: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending command: {str(e)}")


@router.get("/read-data", response_model=ReadDataResponse)
async def read_data(component_id: str):
    """
    Read data from a hardware component
    """
    try:
        result = hardware_manager.read_hardware_data(component_id)

        response = ReadDataResponse(
            success=result.get("success", False),
            data=result.get("data") if result.get("success") else None,
            message=result.get("message") if not result.get("success") else None
        )

        logger.info(f"Data read from {component_id}, success: {result.get('success', False)}")

        return response

    except Exception as e:
        logger.error(f"Error reading data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")


@router.get("/safety-status", response_model=SafetyStatusResponse)
async def get_safety_status():
    """
    Get the current safety status of the system
    """
    try:
        safety_status = hardware_manager.safety_system.check_safety()

        response = SafetyStatusResponse(
            safe=safety_status["safe"],
            violations=safety_status["violations"],
            emergency_stop_active=hardware_manager.safety_system.emergency_stop_active,
            success=True
        )

        logger.info(f"Safety status retrieved, safe: {safety_status['safe']}")

        return response

    except Exception as e:
        logger.error(f"Error getting safety status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting safety status: {str(e)}")


@router.post("/emergency-stop", response_model=EmergencyStopResponse)
async def emergency_stop(request: EmergencyStopRequest):
    """
    Activate or deactivate emergency stop
    """
    try:
        if request.activate:
            hardware_manager.safety_system.activate_emergency_stop()
            message = "Emergency stop activated"
        else:
            hardware_manager.safety_system.deactivate_emergency_stop()
            message = "Emergency stop deactivated"

        response = EmergencyStopResponse(
            success=True,
            message=message
        )

        logger.info(message)

        return response

    except Exception as e:
        logger.error(f"Error controlling emergency stop: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error controlling emergency stop: {str(e)}")


@router.get("/component-status", response_model=ComponentStatusResponse)
async def get_component_status(component_id: str):
    """
    Get the status of a specific hardware component
    """
    try:
        component = hardware_manager.get_component(component_id)
        if not component:
            raise HTTPException(status_code=404, detail=f"Component {component_id} not found")

        if not hasattr(component, 'get_status'):
            raise HTTPException(status_code=400, detail=f"Component {component_id} doesn't support status queries")

        status = component.get_status()

        response = ComponentStatusResponse(
            component_id=status.component_id,
            status=status.status,
            temperature=status.temperature,
            voltage=status.voltage,
            current=status.current,
            last_update=status.last_update.isoformat(),
            error_count=status.error_count,
            safety_level=status.safety_level.value,
            success=True
        )

        logger.info(f"Status retrieved for component {component_id}")

        return response

    except Exception as e:
        logger.error(f"Error getting component status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting component status: {str(e)}")


@router.post("/calibrate", response_model=CalibrationResponse)
async def calibrate_component(request: CalibrationRequest):
    """
    Calibrate a hardware component
    """
    try:
        component_id = request.component_id
        calibration_data = request.calibration_data

        # Store calibration data
        hardware_manager.calibration_manager.store_calibration(component_id, calibration_data)

        response = CalibrationResponse(
            success=True,
            message=f"Component {component_id} calibrated successfully"
        )

        logger.info(f"Component calibrated: {component_id}")

        return response

    except Exception as e:
        logger.error(f"Error calibrating component: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calibrating component: {str(e)}")


@router.get("/components", response_model=List[str])
async def list_components():
    """
    List all registered hardware components
    """
    try:
        components = hardware_manager.list_components()

        logger.info(f"Listed {len(components)} components")

        return components

    except Exception as e:
        logger.error(f"Error listing components: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing components: {str(e)}")


@router.post("/start-monitoring", response_model=Dict[str, bool])
async def start_monitoring():
    """
    Start hardware monitoring
    """
    try:
        hardware_manager.start_monitoring()

        response = {"success": True}

        logger.info("Hardware monitoring started")

        return response

    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting monitoring: {str(e)}")


@router.post("/stop-monitoring", response_model=Dict[str, bool])
async def stop_monitoring():
    """
    Stop hardware monitoring
    """
    try:
        hardware_manager.stop_monitoring()

        response = {"success": True}

        logger.info("Hardware monitoring stopped")

        return response

    except Exception as e:
        logger.error(f"Error stopping monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping monitoring: {str(e)}")