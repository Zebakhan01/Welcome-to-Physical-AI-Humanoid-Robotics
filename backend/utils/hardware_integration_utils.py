import asyncio
import json
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import time
import uuid
from datetime import datetime
import threading
import socket
from abc import ABC, abstractmethod

# Try to import optional modules
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    SERIAL_AVAILABLE = False


class CommunicationProtocol(Enum):
    """Types of communication protocols for hardware"""
    SERIAL = "serial"
    USB = "usb"
    ETHERNET = "ethernet"
    CAN = "can"
    ETHERCAT = "ethercat"
    WIFI = "wifi"
    BLUETOOTH = "bluetooth"


class HardwareComponentType(Enum):
    """Types of hardware components"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    CONTROLLER = "controller"
    PROCESSOR = "processor"


class SafetyLevel(Enum):
    """Safety levels for hardware operations"""
    NORMAL = "normal"
    WARNING = "warning"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


@dataclass
class HardwareStatus:
    """Status of a hardware component"""
    component_id: str
    status: str  # operational, warning, error, maintenance
    temperature: float
    voltage: float
    current: float
    last_update: datetime
    error_count: int
    safety_level: SafetyLevel


@dataclass
class SensorData:
    """Data from a sensor component"""
    sensor_id: str
    timestamp: float
    data: Dict[str, Any]
    quality: float  # 0.0 to 1.0
    calibrated: bool


@dataclass
class ActuatorCommand:
    """Command for an actuator component"""
    actuator_id: str
    command_type: str  # position, velocity, torque, pwm
    value: float
    timestamp: float
    safety_limits: Dict[str, float]


class HardwareInterface(ABC):
    """Abstract base class for hardware interfaces"""

    def __init__(self, component_id: str, component_type: HardwareComponentType):
        self.component_id = component_id
        self.component_type = component_type
        self.is_connected = False
        self.last_communication = None
        self.status = "disconnected"
        self.error_count = 0

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the hardware component"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the hardware component"""
        pass

    @abstractmethod
    async def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command to the hardware component"""
        pass

    @abstractmethod
    async def read_data(self) -> Dict[str, Any]:
        """Read data from the hardware component"""
        pass

    def get_status(self) -> HardwareStatus:
        """Get the current status of the hardware component"""
        return HardwareStatus(
            component_id=self.component_id,
            status=self.status,
            temperature=25.0,  # Default temperature
            voltage=12.0,      # Default voltage
            current=0.5,       # Default current
            last_update=datetime.now(),
            error_count=self.error_count,
            safety_level=SafetyLevel.NORMAL
        )


class SerialInterface(HardwareInterface):
    """Serial communication interface"""

    def __init__(self, component_id: str, port: str, baudrate: int = 115200):
        super().__init__(component_id, HardwareComponentType.CONTROLLER)
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial module is not available. Install with 'pip install pyserial'")
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None

    async def connect(self) -> bool:
        """Connect to serial device"""
        if not SERIAL_AVAILABLE:
            return False

        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            self.is_connected = True
            self.status = "operational"
            self.last_communication = datetime.now()
            return True
        except Exception as e:
            self.error_count += 1
            self.status = f"error: {str(e)}"
            return False

    async def disconnect(self) -> bool:
        """Disconnect from serial device"""
        if not SERIAL_AVAILABLE:
            return False

        if self.serial_connection:
            self.serial_connection.close()
            self.serial_connection = None
        self.is_connected = False
        self.status = "disconnected"
        return True

    async def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send command via serial"""
        if not SERIAL_AVAILABLE or not self.is_connected or not self.serial_connection:
            return {"success": False, "error": "Serial communication not available or not connected"}

        try:
            command_json = json.dumps(command)
            self.serial_connection.write(command_json.encode() + b'\n')
            self.last_communication = datetime.now()

            # Read response
            response = self.serial_connection.readline().decode().strip()
            if response:
                return json.loads(response)
            else:
                return {"success": True, "message": "Command sent"}
        except Exception as e:
            self.error_count += 1
            return {"success": False, "error": str(e)}

    async def read_data(self) -> Dict[str, Any]:
        """Read data from serial"""
        if not SERIAL_AVAILABLE or not self.is_connected or not self.serial_connection:
            return {"success": False, "error": "Serial communication not available or not connected"}

        try:
            line = self.serial_connection.readline().decode().strip()
            if line:
                data = json.loads(line)
                self.last_communication = datetime.now()
                return {"success": True, "data": data}
            else:
                return {"success": False, "error": "No data received"}
        except Exception as e:
            self.error_count += 1
            return {"success": False, "error": str(e)}


class NetworkInterface(HardwareInterface):
    """Network communication interface"""

    def __init__(self, component_id: str, host: str, port: int):
        super().__init__(component_id, HardwareComponentType.CONTROLLER)
        self.host = host
        self.port = port
        self.socket_connection = None

    async def connect(self) -> bool:
        """Connect to network device"""
        try:
            self.socket_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_connection.connect((self.host, self.port))
            self.is_connected = True
            self.status = "operational"
            self.last_communication = datetime.now()
            return True
        except Exception as e:
            self.error_count += 1
            self.status = f"error: {str(e)}"
            return False

    async def disconnect(self) -> bool:
        """Disconnect from network device"""
        if self.socket_connection:
            self.socket_connection.close()
            self.socket_connection = None
        self.is_connected = False
        self.status = "disconnected"
        return True

    async def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send command via network"""
        if not self.is_connected or not self.socket_connection:
            return {"success": False, "error": "Not connected"}

        try:
            command_json = json.dumps(command) + '\n'
            self.socket_connection.send(command_json.encode())
            self.last_communication = datetime.now()

            # Receive response
            response = self.socket_connection.recv(1024).decode().strip()
            if response:
                return json.loads(response)
            else:
                return {"success": True, "message": "Command sent"}
        except Exception as e:
            self.error_count += 1
            return {"success": False, "error": str(e)}

    async def read_data(self) -> Dict[str, Any]:
        """Read data from network"""
        if not self.is_connected or not self.socket_connection:
            return {"success": False, "error": "Not connected"}

        try:
            data = self.socket_connection.recv(1024).decode().strip()
            if data:
                result = json.loads(data)
                self.last_communication = datetime.now()
                return {"success": True, "data": result}
            else:
                return {"success": False, "error": "No data received"}
        except Exception as e:
            self.error_count += 1
            return {"success": False, "error": str(e)}


class SensorInterface(HardwareInterface):
    """Generic sensor interface"""

    def __init__(self, sensor_id: str, sensor_type: str, interface: HardwareInterface):
        super().__init__(sensor_id, HardwareComponentType.SENSOR)
        self.sensor_type = sensor_type
        self.interface = interface
        self.calibration_data = {}
        self.data_buffer = []
        self.max_buffer_size = 100

    async def connect(self) -> bool:
        """Connect the sensor"""
        self.is_connected = await self.interface.connect()
        self.status = self.interface.status
        return self.is_connected

    async def disconnect(self) -> bool:
        """Disconnect the sensor"""
        result = await self.interface.disconnect()
        self.is_connected = False
        return result

    async def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to sensor"""
        return await self.interface.send_command(command)

    async def read_data(self) -> Dict[str, Any]:
        """Read sensor data"""
        raw_data = await self.interface.read_data()
        if raw_data.get("success"):
            sensor_data = SensorData(
                sensor_id=self.component_id,
                timestamp=time.time(),
                data=raw_data.get("data", {}),
                quality=0.95,  # Default quality
                calibrated=True
            )

            # Add to buffer
            self.data_buffer.append(sensor_data)
            if len(self.data_buffer) > self.max_buffer_size:
                self.data_buffer.pop(0)

            return {"success": True, "sensor_data": sensor_data}
        else:
            return raw_data

    def get_latest_data(self) -> Optional[SensorData]:
        """Get the latest sensor data"""
        if self.data_buffer:
            return self.data_buffer[-1]
        return None

    def calibrate(self, calibration_data: Dict[str, Any]) -> bool:
        """Calibrate the sensor"""
        self.calibration_data = calibration_data
        return True


class ActuatorInterface(HardwareInterface):
    """Generic actuator interface"""

    def __init__(self, actuator_id: str, actuator_type: str, interface: HardwareInterface):
        super().__init__(actuator_id, HardwareComponentType.ACTUATOR)
        self.actuator_type = actuator_type
        self.interface = interface
        self.current_position = 0.0
        self.current_velocity = 0.0
        self.current_effort = 0.0
        self.safety_limits = {
            "position_min": -100.0,
            "position_max": 100.0,
            "velocity_max": 10.0,
            "effort_max": 100.0
        }

    async def connect(self) -> bool:
        """Connect the actuator"""
        self.is_connected = await self.interface.connect()
        self.status = self.interface.status
        return self.is_connected

    async def disconnect(self) -> bool:
        """Disconnect the actuator"""
        result = await self.interface.disconnect()
        self.is_connected = False
        return result

    async def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to actuator"""
        # Validate safety limits
        if not self._validate_command(command):
            return {"success": False, "error": "Command violates safety limits"}

        # Update internal state
        if command.get("command_type") == "position":
            self.current_position = command.get("value", self.current_position)
        elif command.get("command_type") == "velocity":
            self.current_velocity = command.get("value", self.current_velocity)
        elif command.get("command_type") == "effort":
            self.current_effort = command.get("value", self.current_effort)

        # Send to hardware
        result = await self.interface.send_command(command)
        return result

    async def read_data(self) -> Dict[str, Any]:
        """Read actuator data"""
        raw_data = await self.interface.read_data()
        if raw_data.get("success"):
            return {
                "success": True,
                "position": self.current_position,
                "velocity": self.current_velocity,
                "effort": self.current_effort,
                "data": raw_data.get("data", {})
            }
        else:
            return raw_data

    def _validate_command(self, command: Dict[str, Any]) -> bool:
        """Validate command against safety limits"""
        cmd_type = command.get("command_type")
        value = command.get("value", 0)

        if cmd_type == "position":
            return (self.safety_limits["position_min"] <= value <=
                   self.safety_limits["position_max"])
        elif cmd_type == "velocity":
            return abs(value) <= self.safety_limits["velocity_max"]
        elif cmd_type == "effort":
            return abs(value) <= self.safety_limits["effort_max"]

        return True  # For other command types, assume valid

    def set_safety_limits(self, limits: Dict[str, float]):
        """Set safety limits for the actuator"""
        self.safety_limits.update(limits)


class SafetySystem:
    """Safety system for hardware operations"""

    def __init__(self):
        self.emergency_stop_active = False
        self.safety_monitors = {}
        self.safety_callbacks = []
        self.last_check_time = time.time()

    def activate_emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_stop_active = True
        self._trigger_safety_callbacks("emergency_stop")

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        self.emergency_stop_active = False
        self._trigger_safety_callbacks("emergency_stop_released")

    def add_safety_monitor(self, monitor_id: str, check_function: Callable):
        """Add a safety monitor function"""
        self.safety_monitors[monitor_id] = check_function

    def add_safety_callback(self, callback: Callable):
        """Add a callback function for safety events"""
        self.safety_callbacks.append(callback)

    def _trigger_safety_callbacks(self, event_type: str):
        """Trigger all safety callbacks"""
        for callback in self.safety_callbacks:
            try:
                callback(event_type)
            except Exception:
                pass  # Don't let callback errors affect safety system

    def check_safety(self) -> Dict[str, Any]:
        """Check safety status"""
        current_time = time.time()
        if current_time - self.last_check_time > 0.1:  # Check every 100ms
            self.last_check_time = current_time

            # Check all safety monitors
            safety_status = {"safe": True, "violations": []}

            for monitor_id, check_func in self.safety_monitors.items():
                try:
                    result = check_func()
                    if not result.get("safe", True):
                        safety_status["safe"] = False
                        safety_status["violations"].append({
                            "monitor": monitor_id,
                            "details": result.get("details", "Unknown violation")
                        })
                except Exception as e:
                    safety_status["safe"] = False
                    safety_status["violations"].append({
                        "monitor": monitor_id,
                        "details": f"Monitor error: {str(e)}"
                    })

            if self.emergency_stop_active:
                safety_status["safe"] = False
                safety_status["violations"].append({
                    "monitor": "emergency_stop",
                    "details": "Emergency stop is active"
                })

            return safety_status

        return {"safe": not self.emergency_stop_active, "violations": []}

    def is_safe(self) -> bool:
        """Check if system is safe to operate"""
        status = self.check_safety()
        return status["safe"]


class CalibrationManager:
    """Manages calibration for sensors and actuators"""

    def __init__(self):
        self.calibration_data = {}
        self.calibration_timestamps = {}

    def store_calibration(self, component_id: str, calibration_data: Dict[str, Any]):
        """Store calibration data for a component"""
        self.calibration_data[component_id] = calibration_data
        self.calibration_timestamps[component_id] = datetime.now()

    def get_calibration(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get calibration data for a component"""
        return self.calibration_data.get(component_id)

    def needs_calibration(self, component_id: str, max_age_hours: float = 24) -> bool:
        """Check if a component needs recalibration"""
        if component_id not in self.calibration_timestamps:
            return True

        timestamp = self.calibration_timestamps[component_id]
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        return age_hours > max_age_hours

    def calibrate_sensor(self, sensor_interface: SensorInterface) -> bool:
        """Perform calibration on a sensor"""
        # In a real implementation, this would perform actual calibration
        # For simulation, we'll just store some default calibration data
        calibration_data = {
            "offset": 0.0,
            "scale": 1.0,
            "temperature_compensation": 0.0
        }
        sensor_interface.calibrate(calibration_data)
        self.store_calibration(sensor_interface.component_id, calibration_data)
        return True


class HardwareManager:
    """Main manager for all hardware components"""

    def __init__(self):
        self.components = {}
        self.safety_system = SafetySystem()
        self.calibration_manager = CalibrationManager()
        self.communication_protocols = {}
        self.monitoring_thread = None
        self.monitoring_active = False

    def register_component(self, component: HardwareInterface) -> bool:
        """Register a hardware component"""
        if component.component_id in self.components:
            return False

        self.components[component.component_id] = component
        return True

    def get_component(self, component_id: str) -> Optional[HardwareInterface]:
        """Get a hardware component by ID"""
        return self.components.get(component_id)

    def list_components(self) -> List[str]:
        """List all registered component IDs"""
        return list(self.components.keys())

    async def connect_all(self) -> Dict[str, bool]:
        """Connect all registered components"""
        results = {}
        for comp_id, component in self.components.items():
            if self.safety_system.is_safe():
                results[comp_id] = await component.connect()
            else:
                results[comp_id] = False
        return results

    async def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect all registered components"""
        results = {}
        for comp_id, component in self.components.items():
            results[comp_id] = await component.disconnect()
        return results

    def start_monitoring(self):
        """Start hardware monitoring in background"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop hardware monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Check safety system
                safety_status = self.safety_system.check_safety()

                # Check component statuses
                for comp_id, component in self.components.items():
                    if hasattr(component, 'get_status'):
                        status = component.get_status()
                        # Log status or trigger alerts if needed

                time.sleep(0.1)  # 100ms between checks
            except Exception:
                time.sleep(0.1)  # Continue monitoring even if there's an error

    def execute_hardware_command(self, component_id: str, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command on a hardware component with safety checks"""
        if not self.safety_system.is_safe():
            return {"success": False, "error": "Safety system not safe"}

        component = self.get_component(component_id)
        if not component:
            return {"success": False, "error": f"Component {component_id} not found"}

        if not component.is_connected:
            return {"success": False, "error": "Component not connected"}

        try:
            # In a real implementation, we would await this
            # For this simulation, we'll call the method directly
            if hasattr(component, 'send_command'):
                return asyncio.run(component.send_command(command))
            else:
                return {"success": False, "error": "Component doesn't support commands"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def read_hardware_data(self, component_id: str) -> Dict[str, Any]:
        """Read data from a hardware component"""
        component = self.get_component(component_id)
        if not component:
            return {"success": False, "error": f"Component {component_id} not found"}

        if not component.is_connected:
            return {"success": False, "error": "Component not connected"}

        try:
            # In a real implementation, we would await this
            # For this simulation, we'll call the method directly
            if hasattr(component, 'read_data'):
                return asyncio.run(component.read_data())
            else:
                return {"success": False, "error": "Component doesn't support reading"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global hardware manager instance
hardware_manager = HardwareManager()