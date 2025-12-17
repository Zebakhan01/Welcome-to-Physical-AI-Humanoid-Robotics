---
sidebar_position: 6
---

# System Integration

## Introduction to System Integration

System integration in Physical AI and humanoid robotics involves combining various hardware and software components into a cohesive, functional system. This process requires careful consideration of interfaces, communication protocols, timing constraints, and safety requirements. Successful integration ensures that all subsystems work harmoniously to achieve the desired robotic capabilities while maintaining reliability and safety.

## Integration Architecture

### Hardware Integration Architecture

```python
# system_integration_architecture.py
import threading
import time
import logging
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from enum import Enum

class IntegrationPhase(Enum):
    """
    System integration phases
    """
    HARDWARE_INTEGRATION = "hardware_integration"
    SOFTWARE_INTEGRATION = "software_integration"
    SAFETY_INTEGRATION = "safety_integration"
    PERFORMANCE_INTEGRATION = "performance_integration"
    VALIDATION_INTEGRATION = "validation_integration"

@dataclass
class ComponentSpec:
    """
    Specification for system component
    """
    name: str
    type: str
    interfaces: List[str]
    dependencies: List[str]
    resource_requirements: Dict[str, Any]
    safety_requirements: List[str]
    performance_requirements: Dict[str, Any]

class SystemIntegrationManager:
    """
    Manages the integration of all system components
    """

    def __init__(self):
        self.components = {}
        self.interfaces = {}
        self.dependency_graph = {}
        self.integration_status = {}
        self.safety_systems = {}
        self.performance_monitors = {}
        self.communication_buses = {}
        self.is_initialized = False

        # Logging setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Integration phases
        self.integration_phases = [
            IntegrationPhase.HARDWARE_INTEGRATION,
            IntegrationPhase.SOFTWARE_INTEGRATION,
            IntegrationPhase.SAFETY_INTEGRATION,
            IntegrationPhase.PERFORMANCE_INTEGRATION,
            IntegrationPhase.VALIDATION_INTEGRATION
        ]

        # Component health monitoring
        self.component_health = {}
        self.health_check_interval = 1.0  # seconds

    def register_component(self, name: str, spec: ComponentSpec):
        """
        Register a component for integration
        """
        self.components[name] = {
            'spec': spec,
            'instance': None,
            'status': 'registered',
            'health': 'unknown',
            'interfaces': spec.interfaces,
            'dependencies': spec.dependencies
        }

        # Initialize dependency tracking
        self.dependency_graph[name] = spec.dependencies

        self.logger.info(f"Registered component: {name}")

    def setup_hardware_interfaces(self):
        """
        Setup hardware interfaces between components
        """
        self.logger.info("Setting up hardware interfaces...")

        # Setup communication buses
        self.setup_can_bus()
        self.setup_ethernet_network()
        self.setup_serial_connections()
        self.setup_power_distribution()

        # Verify all interfaces are functional
        interface_status = self.verify_interfaces()
        return interface_status

    def setup_can_bus(self):
        """
        Setup CAN bus communication
        """
        try:
            import can
            self.can_bus = can.interface.Bus(channel='can0', bustype='socketcan')
            self.communication_buses['can'] = self.can_bus
            self.logger.info("CAN bus initialized successfully")
        except ImportError:
            self.logger.warning("CAN bus not available (python-can not installed)")
            self.can_bus = None

    def setup_ethernet_network(self):
        """
        Setup Ethernet communication
        """
        # This would set up ROS communication, etc.
        self.communication_buses['ethernet'] = {
            'ip_address': '192.168.1.100',
            'subnet_mask': '255.255.255.0',
            'ports': [11311, 9090, 10000]  # ROS, rosbridge, custom
        }
        self.logger.info("Ethernet network configured")

    def setup_serial_connections(self):
        """
        Setup serial communication
        """
        import serial
        self.serial_ports = {}

        # Define serial connections for different components
        serial_configs = {
            'imu': {'port': '/dev/ttyUSB0', 'baudrate': 115200},
            'lidar': {'port': '/dev/ttyUSB1', 'baudrate': 115200},
            'motor_controllers': {'port': '/dev/ttyACM0', 'baudrate': 1000000}
        }

        for device_name, config in serial_configs.items():
            try:
                port = serial.Serial(config['port'], config['baudrate'])
                self.serial_ports[device_name] = port
                self.logger.info(f"Serial port {device_name} configured: {config['port']}")
            except serial.SerialException as e:
                self.logger.error(f"Failed to configure serial port for {device_name}: {e}")

    def setup_power_distribution(self):
        """
        Setup power distribution system
        """
        self.power_system = {
            'main_battery': {'voltage': 24.0, 'capacity': 10.0, 'status': 'ok'},
            'power_distributors': ['PD1', 'PD2', 'PD3'],
            'current_monitors': ['IM1', 'IM2', 'IM3'],
            'emergency_shutdown': False
        }
        self.logger.info("Power distribution system configured")

    def verify_interfaces(self) -> bool:
        """
        Verify all interfaces are functional
        """
        verification_results = {
            'can_bus': self.verify_can_bus(),
            'ethernet': self.verify_ethernet(),
            'serial': self.verify_serial(),
            'power': self.verify_power_system()
        }

        all_good = all(verification_results.values())
        self.logger.info(f"Interface verification: {verification_results}, All good: {all_good}")

        return all_good

    def verify_can_bus(self) -> bool:
        """
        Verify CAN bus functionality
        """
        if not self.can_bus:
            return False

        try:
            # Send test message
            test_msg = can.Message(arbitration_id=0x123, data=[1, 2, 3, 4, 5, 6, 7, 8])
            self.can_bus.send(test_msg)
            return True
        except Exception as e:
            self.logger.error(f"CAN bus verification failed: {e}")
            return False

    def verify_ethernet(self) -> bool:
        """
        Verify Ethernet communication
        """
        try:
            # This would check ROS communication
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(1)
            sock.connect(('8.8.8.8', 80))  # Google DNS
            sock.close()
            return True
        except Exception as e:
            self.logger.error(f"Ethernet verification failed: {e}")
            return False

    def verify_serial(self) -> bool:
        """
        Verify serial connections
        """
        all_connected = True
        for device_name, port in self.serial_ports.items():
            try:
                if port.is_open:
                    # Send test command
                    port.write(b'TEST\r\n')
                    response = port.read(100)
                    self.logger.debug(f"Serial test response from {device_name}: {response}")
                else:
                    all_connected = False
                    self.logger.error(f"Serial port not open for {device_name}")
            except Exception as e:
                self.logger.error(f"Serial verification failed for {device_name}: {e}")
                all_connected = False

        return all_connected

    def verify_power_system(self) -> bool:
        """
        Verify power system functionality
        """
        # Check main battery voltage
        if self.power_system['main_battery']['voltage'] < 20.0:
            self.logger.error("Main battery voltage too low")
            return False

        # Check power distributors
        # This would involve actual hardware checks
        return True

    def establish_component_interfaces(self):
        """
        Establish interfaces between components
        """
        for comp_name, comp_info in self.components.items():
            interfaces = comp_info['interfaces']

            for interface in interfaces:
                if interface.startswith('can_'):
                    self.establish_can_interface(comp_name, interface)
                elif interface.startswith('ethernet_'):
                    self.establish_ethernet_interface(comp_name, interface)
                elif interface.startswith('serial_'):
                    self.establish_serial_interface(comp_name, interface)
                elif interface.startswith('power_'):
                    self.establish_power_interface(comp_name, interface)

    def establish_can_interface(self, component_name: str, interface_name: str):
        """
        Establish CAN bus interface for component
        """
        # Assign CAN ID to component
        can_id = self.assign_can_id(component_name)
        self.interfaces[f"{component_name}_{interface_name}"] = {
            'type': 'can',
            'id': can_id,
            'bus': self.can_bus,
            'status': 'configured'
        }

    def assign_can_id(self, component_name: str) -> int:
        """
        Assign CAN ID to component
        """
        # Simple ID assignment based on component name
        # In practice, this would use a more sophisticated assignment scheme
        hash_val = hash(component_name) % 2048  # 11-bit CAN ID
        return hash_val

    def establish_ethernet_interface(self, component_name: str, interface_name: str):
        """
        Establish Ethernet interface for component
        """
        # This would typically involve ROS topic/service setup
        ethernet_config = self.communication_buses['ethernet']
        self.interfaces[f"{component_name}_{interface_name}"] = {
            'type': 'ethernet',
            'config': ethernet_config,
            'status': 'configured'
        }

    def establish_serial_interface(self, component_name: str, interface_name: str):
        """
        Establish serial interface for component
        """
        device_type = interface_name.split('_')[1]  # e.g., 'imu', 'lidar'
        if device_type in self.serial_ports:
            self.interfaces[f"{component_name}_{interface_name}"] = {
                'type': 'serial',
                'port': self.serial_ports[device_type],
                'status': 'configured'
            }

    def establish_power_interface(self, component_name: str, interface_name: str):
        """
        Establish power interface for component
        """
        self.interfaces[f"{component_name}_{interface_name}"] = {
            'type': 'power',
            'voltage': 24.0,
            'current_limit': 5.0,
            'status': 'configured'
        }

    def verify_component_dependencies(self) -> bool:
        """
        Verify all component dependencies are satisfied
        """
        for comp_name, comp_info in self.components.items():
            dependencies = comp_info['dependencies']

            for dep in dependencies:
                if dep not in self.components:
                    self.logger.error(f"Dependency {dep} not found for component {comp_name}")
                    return False

                dep_status = self.components[dep]['status']
                if dep_status != 'ready':
                    self.logger.error(f"Dependency {dep} not ready for component {comp_name}")
                    return False

        self.logger.info("All component dependencies verified")
        return True

    def initialize_components(self):
        """
        Initialize all registered components
        """
        initialization_order = self.calculate_initialization_order()

        for comp_name in initialization_order:
            comp_info = self.components[comp_name]
            try:
                # Initialize component
                comp_instance = self.create_component_instance(comp_name, comp_info)
                comp_info['instance'] = comp_instance
                comp_info['status'] = 'initialized'

                # Verify initialization
                if self.verify_component_initialization(comp_name):
                    comp_info['status'] = 'ready'
                    self.logger.info(f"Component {comp_name} initialized successfully")
                else:
                    comp_info['status'] = 'error'
                    self.logger.error(f"Component {comp_name} failed verification")

            except Exception as e:
                comp_info['status'] = 'error'
                self.logger.error(f"Failed to initialize component {comp_name}: {e}")

    def calculate_initialization_order(self) -> List[str]:
        """
        Calculate proper initialization order based on dependencies
        """
        from collections import defaultdict, deque

        # Build dependency graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        for comp_name, dependencies in self.dependency_graph.items():
            for dep in dependencies:
                graph[dep].append(comp_name)
                in_degree[comp_name] += 1
            if comp_name not in in_degree:
                in_degree[comp_name] = 0

        # Topological sort
        queue = deque([comp for comp, degree in in_degree.items() if degree == 0])
        order = []

        while queue:
            current = queue.popleft()
            order.append(current)

            for dependent in graph[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(self.components):
            raise ValueError("Circular dependency detected in component dependencies")

        return order

    def create_component_instance(self, name: str, info: Dict) -> Any:
        """
        Create instance of component based on its type
        """
        spec = info['spec']
        comp_type = spec.type

        if comp_type == 'motor_controller':
            return self.create_motor_controller(name)
        elif comp_type == 'sensor':
            return self.create_sensor(name)
        elif comp_type == 'computer_vision':
            return self.create_cv_system(name)
        elif comp_type == 'navigation':
            return self.create_navigation_system(name)
        elif comp_type == 'control':
            return self.create_control_system(name)
        else:
            raise ValueError(f"Unknown component type: {comp_type}")

    def create_motor_controller(self, name: str):
        """
        Create motor controller instance
        """
        # This would create actual motor controller
        # For now, return a mock controller
        class MockMotorController:
            def __init__(self, name):
                self.name = name
                self.is_initialized = True

            def enable(self):
                return True

            def disable(self):
                return True

            def set_position(self, position):
                return True

        return MockMotorController(name)

    def create_sensor(self, name: str):
        """
        Create sensor instance
        """
        # This would create actual sensor
        class MockSensor:
            def __init__(self, name):
                self.name = name
                self.is_initialized = True

            def start(self):
                return True

            def stop(self):
                return True

            def get_data(self):
                return {}

        return MockSensor(name)

    def verify_component_initialization(self, component_name: str) -> bool:
        """
        Verify component is properly initialized
        """
        comp_info = self.components[component_name]
        instance = comp_info['instance']

        # Check if component can respond to basic commands
        try:
            if hasattr(instance, 'is_initialized'):
                return instance.is_initialized
            elif hasattr(instance, 'get_status'):
                status = instance.get_status()
                return status.get('initialized', False)
            else:
                # Component doesn't have standard interface, assume OK
                return True
        except Exception as e:
            self.logger.error(f"Error verifying component {component_name}: {e}")
            return False

    def start_integration_monitoring(self):
        """
        Start continuous integration monitoring
        """
        self.monitoring_thread = threading.Thread(target=self._monitor_integration, daemon=True)
        self.monitoring_thread.start()

    def _monitor_integration(self):
        """
        Monitor integration status continuously
        """
        while True:
            try:
                # Check component health
                self._check_component_health()

                # Verify interfaces
                self._verify_interfaces_continuously()

                # Check performance metrics
                self._check_performance_metrics()

                time.sleep(self.health_check_interval)

            except Exception as e:
                self.logger.error(f"Error in integration monitoring: {e}")
                time.sleep(1.0)  # Longer sleep on error

    def _check_component_health(self):
        """
        Check health of all components
        """
        for comp_name, comp_info in self.components.items():
            if comp_info['instance'] is not None:
                try:
                    health = self._assess_component_health(comp_info['instance'])
                    self.component_health[comp_name] = health
                    comp_info['health'] = health

                    if health == 'error':
                        self.logger.warning(f"Component {comp_name} health degraded")
                except Exception as e:
                    self.logger.error(f"Error checking health of {comp_name}: {e}")
                    self.component_health[comp_name] = 'error'

    def _assess_component_health(self, component) -> str:
        """
        Assess health of a component instance
        """
        try:
            if hasattr(component, 'get_health_status'):
                return component.get_health_status()
            elif hasattr(component, 'is_operational'):
                return 'ok' if component.is_operational() else 'degraded'
            else:
                return 'ok'  # Assume healthy if no health method
        except Exception:
            return 'error'

    def _verify_interfaces_continuously(self):
        """
        Continuously verify interface functionality
        """
        # This would perform ongoing interface verification
        # For now, just log that it's running
        pass

    def _check_performance_metrics(self):
        """
        Check performance metrics
        """
        # This would monitor performance of integrated system
        # For now, just log that it's running
        pass

    def integrate_safety_systems(self):
        """
        Integrate safety systems across all components
        """
        self.logger.info("Integrating safety systems...")

        # Create safety manager
        self.safety_manager = SafetySystemManager()

        # Register safety-critical components
        for comp_name, comp_info in self.components.items():
            if 'safety' in comp_info['spec'].safety_requirements:
                self.safety_manager.register_component(comp_name, comp_info['instance'])

        # Configure safety protocols
        self.safety_manager.configure_protocols({
            'emergency_stop': True,
            'collision_detection': True,
            'force_limiting': True,
            'velocity_limiting': True
        })

        # Start safety monitoring
        self.safety_manager.start_monitoring()

        self.logger.info("Safety systems integrated")

    def integrate_performance_systems(self):
        """
        Integrate performance monitoring systems
        """
        self.logger.info("Integrating performance systems...")

        # Create performance monitor
        self.performance_monitor = PerformanceMonitor()

        # Register performance-critical components
        for comp_name, comp_info in self.components.items():
            if 'performance' in comp_info['spec'].performance_requirements:
                self.performance_monitor.register_component(comp_name)

        # Configure performance metrics
        self.performance_monitor.configure_metrics({
            'cpu_usage': True,
            'memory_usage': True,
            'network_latency': True,
            'control_loop_timing': True,
            'sensor_response_time': True
        })

        # Start performance monitoring
        self.performance_monitor.start_monitoring()

        self.logger.info("Performance systems integrated")

    def run_integration_validation(self):
        """
        Run comprehensive integration validation
        """
        self.logger.info("Running integration validation...")

        validation_results = {
            'hardware_interfaces': self.verify_interfaces(),
            'component_communication': self.verify_component_communication(),
            'safety_systems': self.verify_safety_systems(),
            'performance_metrics': self.verify_performance_metrics(),
            'system_stability': self.verify_system_stability()
        }

        overall_success = all(validation_results.values())

        self.logger.info(f"Integration validation results: {validation_results}")
        self.logger.info(f"Overall success: {overall_success}")

        return overall_success, validation_results

    def verify_component_communication(self) -> bool:
        """
        Verify communication between components
        """
        # Test communication paths
        communication_tests = []

        for comp_name, comp_info in self.components.items():
            if comp_info['instance'] is not None:
                try:
                    # Test if component can communicate
                    if hasattr(comp_info['instance'], 'test_communication'):
                        test_result = comp_info['instance'].test_communication()
                        communication_tests.append(test_result)
                    else:
                        # Assume communication works if no specific test method
                        communication_tests.append(True)
                except Exception as e:
                    self.logger.error(f"Communication test failed for {comp_name}: {e}")
                    communication_tests.append(False)

        return all(communication_tests)

    def verify_safety_systems(self) -> bool:
        """
        Verify safety systems are functional
        """
        if hasattr(self, 'safety_manager'):
            return self.safety_manager.verify_safety_systems()
        return True  # Assume safe if no safety manager

    def verify_performance_metrics(self) -> bool:
        """
        Verify performance metrics meet requirements
        """
        if hasattr(self, 'performance_monitor'):
            return self.performance_monitor.verify_performance_requirements()
        return True  # Assume acceptable if no monitor

    def verify_system_stability(self) -> bool:
        """
        Verify system stability over time
        """
        # Run stability test for a period
        test_duration = 30.0  # seconds
        start_time = time.time()

        while time.time() - start_time < test_duration:
            # Check that all components are still operational
            for comp_name, comp_info in self.components.items():
                if comp_info['instance'] is not None:
                    if hasattr(comp_info['instance'], 'is_operational'):
                        if not comp_info['instance'].is_operational():
                            self.logger.error(f"Component {comp_name} became non-operational during stability test")
                            return False

            time.sleep(0.1)  # Check every 100ms

        return True
```

## Communication Protocols Integration

### Multi-Protocol Communication System

```python
# communication_integration.py
import threading
import queue
import time
from abc import ABC, abstractmethod

class CommunicationProtocol(ABC):
    """
    Abstract base class for communication protocols
    """

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.is_connected = False
        self.message_queue = queue.Queue()
        self.handlers = {}

    @abstractmethod
    def connect(self):
        """Connect to the communication medium"""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the communication medium"""
        pass

    @abstractmethod
    def send_message(self, message):
        """Send a message"""
        pass

    @abstractmethod
    def receive_message(self):
        """Receive a message"""
        pass

    def register_handler(self, message_type, handler_func):
        """Register a handler for specific message type"""
        self.handlers[message_type] = handler_func

    def process_messages(self):
        """Process received messages"""
        while self.is_connected:
            try:
                message = self.receive_message()
                if message and 'type' in message:
                    msg_type = message['type']
                    if msg_type in self.handlers:
                        self.handlers[msg_type](message)
                    else:
                        print(f"No handler for message type: {msg_type}")
            except Exception as e:
                print(f"Error processing message: {e}")
                time.sleep(0.01)  # Small delay to prevent busy waiting


class CANProtocol(CommunicationProtocol):
    """
    CAN bus communication protocol
    """

    def __init__(self, name, config):
        super().__init__(name, config)
        self.can_bus = None
        self.node_id = config.get('node_id', 1)

    def connect(self):
        """Connect to CAN bus"""
        try:
            import can
            self.can_bus = can.interface.Bus(
                channel=self.config.get('channel', 'can0'),
                bustype=self.config.get('bustype', 'socketcan'),
                bitrate=self.config.get('bitrate', 500000)
            )
            self.is_connected = True
            print(f"Connected to CAN bus: {self.config.get('channel', 'can0')}")
            return True
        except Exception as e:
            print(f"Failed to connect to CAN bus: {e}")
            return False

    def disconnect(self):
        """Disconnect from CAN bus"""
        if self.can_bus:
            self.can_bus.shutdown()
            self.is_connected = False
            print("Disconnected from CAN bus")

    def send_message(self, message):
        """Send CAN message"""
        if not self.is_connected or not self.can_bus:
            return False

        try:
            # Create CAN message
            can_msg = can.Message(
                arbitration_id=message.get('id', self.node_id),
                data=message.get('data', []),
                is_extended_id=True
            )
            self.can_bus.send(can_msg)
            return True
        except Exception as e:
            print(f"Error sending CAN message: {e}")
            return False

    def receive_message(self):
        """Receive CAN message"""
        if not self.is_connected or not self.can_bus:
            return None

        try:
            msg = self.can_bus.recv(timeout=0.001)  # 1ms timeout
            if msg:
                return {
                    'type': 'can',
                    'id': msg.arbitration_id,
                    'data': list(msg.data),
                    'timestamp': msg.timestamp
                }
        except Exception as e:
            print(f"Error receiving CAN message: {e}")

        return None


class EthernetProtocol(CommunicationProtocol):
    """
    Ethernet/UDP communication protocol
    """

    def __init__(self, name, config):
        super().__init__(name, config)
        self.socket = None
        self.ip_address = config.get('ip_address', '127.0.0.1')
        self.port = config.get('port', 10000)
        self.socket_timeout = config.get('timeout', 0.1)

    def connect(self):
        """Connect to Ethernet socket"""
        try:
            import socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.socket_timeout)
            self.socket.bind((self.ip_address, self.port))
            self.is_connected = True
            print(f"Connected to Ethernet: {self.ip_address}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to Ethernet: {e}")
            return False

    def disconnect(self):
        """Disconnect from Ethernet"""
        if self.socket:
            self.socket.close()
            self.is_connected = False
            print("Disconnected from Ethernet")

    def send_message(self, message):
        """Send UDP message"""
        if not self.is_connected or not self.socket:
            return False

        try:
            import pickle
            serialized_msg = pickle.dumps(message)
            self.socket.sendto(serialized_msg, (self.ip_address, self.port))
            return True
        except Exception as e:
            print(f"Error sending UDP message: {e}")
            return False

    def receive_message(self):
        """Receive UDP message"""
        if not self.is_connected or not self.socket:
            return None

        try:
            import pickle
            data, addr = self.socket.recvfrom(4096)
            message = pickle.loads(data)
            message['sender_address'] = addr
            return message
        except socket.timeout:
            return None  # No message received
        except Exception as e:
            print(f"Error receiving UDP message: {e}")
            return None


class SerialProtocol(CommunicationProtocol):
    """
    Serial communication protocol
    """

    def __init__(self, name, config):
        super().__init__(name, config)
        self.serial_port = None
        self.port_name = config.get('port', '/dev/ttyUSB0')
        self.baudrate = config.get('baudrate', 115200)

    def connect(self):
        """Connect to serial port"""
        try:
            import serial
            self.serial_port = serial.Serial(
                port=self.port_name,
                baudrate=self.baudrate,
                timeout=self.config.get('timeout', 0.1)
            )
            self.is_connected = True
            print(f"Connected to serial port: {self.port_name}")
            return True
        except Exception as e:
            print(f"Failed to connect to serial port: {e}")
            return False

    def disconnect(self):
        """Disconnect from serial port"""
        if self.serial_port:
            self.serial_port.close()
            self.is_connected = False
            print("Disconnected from serial port")

    def send_message(self, message):
        """Send serial message"""
        if not self.is_connected or not self.serial_port:
            return False

        try:
            # Convert message to string and send
            msg_str = str(message) + '\n'
            self.serial_port.write(msg_str.encode())
            return True
        except Exception as e:
            print(f"Error sending serial message: {e}")
            return False

    def receive_message(self):
        """Receive serial message"""
        if not self.is_connected or not self.serial_port:
            return None

        try:
            if self.serial_port.in_waiting > 0:
                line = self.serial_port.readline().decode().strip()
                if line:
                    # Try to parse as JSON if possible
                    import json
                    try:
                        return json.loads(line)
                    except:
                        return {'type': 'serial', 'data': line}
        except Exception as e:
            print(f"Error receiving serial message: {e}")
            return None


class CommunicationBusManager:
    """
    Manager for multiple communication protocols
    """

    def __init__(self):
        self.protocols = {}
        self.protocol_routes = {}  # {message_type: [protocol_names]}
        self.message_router = MessageRouter()

    def add_protocol(self, name: str, protocol: CommunicationProtocol):
        """Add a communication protocol to the manager"""
        self.protocols[name] = protocol

    def connect_all_protocols(self):
        """Connect all registered protocols"""
        success_count = 0
        for name, protocol in self.protocols.items():
            if protocol.connect():
                success_count += 1
            else:
                print(f"Failed to connect protocol: {name}")

        return success_count == len(self.protocols)

    def disconnect_all_protocols(self):
        """Disconnect all protocols"""
        for protocol in self.protocols.values():
            protocol.disconnect()

    def route_message(self, message, destination_protocol=None):
        """
        Route message to appropriate protocol(s)
        """
        if destination_protocol:
            # Route to specific protocol
            if destination_protocol in self.protocols:
                return self.protocols[destination_protocol].send_message(message)
        else:
            # Route based on message type or destination
            message_type = message.get('type', 'generic')
            destination = message.get('destination', 'all')

            if destination == 'all':
                # Send to all protocols
                results = []
                for protocol in self.protocols.values():
                    results.append(protocol.send_message(message))
                return all(results)
            else:
                # Send to specific protocol based on routing table
                if destination in self.protocols:
                    return self.protocols[destination].send_message(message)
                else:
                    print(f"Unknown destination protocol: {destination}")
                    return False

    def start_message_processing(self):
        """Start processing messages from all protocols"""
        for name, protocol in self.protocols.items():
            # Start processing in separate thread for each protocol
            thread = threading.Thread(target=protocol.process_messages, daemon=True)
            thread.start()

    def broadcast_message(self, message):
        """Broadcast message to all connected protocols"""
        results = []
        for protocol in self.protocols.values():
            if protocol.is_connected:
                results.append(protocol.send_message(message))
        return all(results) if results else False

    def send_to_protocol(self, protocol_name, message):
        """Send message to specific protocol"""
        if protocol_name in self.protocols:
            return self.protocols[protocol_name].send_message(message)
        return False

    def send_to_multiple_protocols(self, protocol_names, message):
        """Send message to multiple protocols"""
        results = []
        for name in protocol_names:
            if name in self.protocols:
                results.append(self.protocols[name].send_message(message))
        return all(results) if results else False


class MessageRouter:
    """
    Advanced message routing system
    """

    def __init__(self):
        self.routes = {}
        self.filters = {}
        self.middlewares = []

    def add_route(self, pattern, protocol_name):
        """Add a route pattern"""
        self.routes[pattern] = protocol_name

    def add_filter(self, message_type, filter_func):
        """Add a filter for specific message type"""
        if message_type not in self.filters:
            self.filters[message_type] = []
        self.filters[message_type].append(filter_func)

    def add_middleware(self, middleware_func):
        """Add middleware function"""
        self.middlewares.append(middleware_func)

    def route_message(self, message):
        """Route message based on pattern matching"""
        # Apply middleware
        for middleware in self.middlewares:
            message = middleware(message)
            if message is None:
                return False  # Message was filtered out

        # Apply filters
        msg_type = message.get('type', 'generic')
        if msg_type in self.filters:
            for filter_func in self.filters[msg_type]:
                if not filter_func(message):
                    return False  # Message was filtered out

        # Determine destination protocol
        destination = self._determine_destination(message)

        # Route to appropriate protocol
        return destination

    def _determine_destination(self, message):
        """Determine destination protocol for message"""
        # This would implement sophisticated routing logic
        # For now, return a default protocol
        return 'default'


def create_robot_communication_system():
    """
    Create integrated communication system for robot
    """
    comm_manager = CommunicationBusManager()

    # Create protocols
    can_protocol = CANProtocol('can_protocol', {
        'channel': 'can0',
        'bustype': 'socketcan',
        'bitrate': 500000,
        'node_id': 1
    })

    ethernet_protocol = EthernetProtocol('ethernet_protocol', {
        'ip_address': '127.0.0.1',
        'port': 10000,
        'timeout': 0.1
    })

    serial_protocol = SerialProtocol('serial_protocol', {
        'port': '/dev/ttyUSB0',
        'baudrate': 115200,
        'timeout': 0.1
    })

    # Add protocols to manager
    comm_manager.add_protocol('can', can_protocol)
    comm_manager.add_protocol('ethernet', ethernet_protocol)
    comm_manager.add_protocol('serial', serial_protocol)

    # Connect all protocols
    if comm_manager.connect_all_protocols():
        print("All communication protocols connected successfully")
        comm_manager.start_message_processing()
        return comm_manager
    else:
        print("Failed to connect all communication protocols")
        return None
```

## Safety Integration Systems

### Comprehensive Safety Integration

```python
# safety_integration.py
import threading
import time
from enum import Enum
from typing import Dict, List, Any, Callable

class SafetyLevel(Enum):
    """
    Safety level classifications
    """
    SAFE = "safe"
    WARNING = "warning"
    ALERT = "alert"
    EMERGENCY = "emergency"
    CRITICAL = "critical"

class SafetyZone(Enum):
    """
    Safety zone classifications
    """
    OPERATIONAL = "operational"
    WORKSPACE = "workspace"
    COLLISION = "collision"
    HUMAN_INTERACTION = "human_interaction"
    ENVIRONMENTAL = "environmental"

class SafetySystemManager:
    """
    Comprehensive safety system manager
    """

    def __init__(self):
        self.safety_monitors = {}
        self.safety_protocols = {}
        self.emergency_stop_active = False
        self.safety_zones = {}
        self.safety_limits = {}
        self.violation_log = []
        self.is_active = True

        # Initialize safety components
        self._initialize_safety_monitors()
        self._initialize_safety_protocols()
        self._initialize_safety_zones()

    def _initialize_safety_monitors(self):
        """
        Initialize various safety monitors
        """
        self.safety_monitors = {
            'collision_monitor': CollisionMonitor(),
            'force_monitor': ForceMonitor(),
            'velocity_monitor': VelocityMonitor(),
            'position_monitor': PositionMonitor(),
            'temperature_monitor': TemperatureMonitor(),
            'power_monitor': PowerMonitor(),
            'emergency_stop_monitor': EmergencyStopMonitor()
        }

    def _initialize_safety_protocols(self):
        """
        Initialize safety protocols
        """
        self.safety_protocols = {
            'collision_avoidance': CollisionAvoidanceProtocol(),
            'force_limiting': ForceLimitingProtocol(),
            'velocity_capping': VelocityCappingProtocol(),
            'position_clamping': PositionClampingProtocol(),
            'emergency_procedures': EmergencyProceduresProtocol(),
            'human_safety': HumanSafetyProtocol()
        }

    def _initialize_safety_zones(self):
        """
        Initialize safety zones
        """
        self.safety_zones = {
            'workspace_bounds': {
                'zone_type': SafetyZone.WORKSPACE,
                'bounds': [-5, 5, -5, 5, 0, 3],  # [x_min, x_max, y_min, y_max, z_min, z_max]
                'safety_level': SafetyLevel.WARNING
            },
            'collision_zone': {
                'zone_type': SafetyZone.COLLISION,
                'bounds': [0, 0, 0, 0, 0, 0],  # Will be updated dynamically
                'safety_level': SafetyLevel.ALERT
            },
            'human_interaction_zone': {
                'zone_type': SafetyZone.HUMAN_INTERACTION,
                'bounds': [-1, 1, -1, 1, 0, 2],  # Around robot
                'safety_level': SafetyLevel.SAFE
            }
        }

    def start_safety_monitoring(self):
        """
        Start all safety monitoring systems
        """
        for name, monitor in self.safety_monitors.items():
            monitor.start_monitoring()

        # Start safety protocol execution
        self.protocol_thread = threading.Thread(target=self._execute_safety_protocols, daemon=True)
        self.protocol_thread.start()

        print("Safety monitoring started")

    def stop_safety_monitoring(self):
        """
        Stop all safety monitoring systems
        """
        for monitor in self.safety_monitors.values():
            monitor.stop_monitoring()

        self.is_active = False
        print("Safety monitoring stopped")

    def check_safety_status(self) -> Dict[str, Any]:
        """
        Check overall safety status
        """
        status = {
            'emergency_stop': self.emergency_stop_active,
            'safety_level': SafetyLevel.SAFE,
            'violations': [],
            'monitors_status': {},
            'zones_status': {}
        }

        # Check each monitor
        for name, monitor in self.safety_monitors.items():
            monitor_status = monitor.get_status()
            status['monitors_status'][name] = monitor_status

            if monitor_status['safety_level'] != SafetyLevel.SAFE:
                status['safety_level'] = monitor_status['safety_level']
                status['violations'].append({
                    'monitor': name,
                    'level': monitor_status['safety_level'],
                    'details': monitor_status.get('details', {})
                })

        # Check safety zones
        for zone_name, zone_config in self.safety_zones.items():
            zone_status = self._check_zone_status(zone_name, zone_config)
            status['zones_status'][zone_name] = zone_status

            if zone_status['safety_level'] != SafetyLevel.SAFE:
                status['safety_level'] = zone_status['safety_level']

        return status

    def _check_zone_status(self, zone_name: str, zone_config: Dict) -> Dict[str, Any]:
        """
        Check status of a safety zone
        """
        # This would check if robot is within zone bounds
        # For now, return safe status
        return {
            'zone_name': zone_name,
            'is_violated': False,
            'safety_level': SafetyLevel.SAFE,
            'details': 'Zone check passed'
        }

    def _execute_safety_protocols(self):
        """
        Execute safety protocols in background thread
        """
        while self.is_active:
            try:
                # Check safety status
                safety_status = self.check_safety_status()

                # If safety level is not safe, execute appropriate protocols
                if safety_status['safety_level'] in [SafetyLevel.WARNING, SafetyLevel.ALERT]:
                    self._execute_warning_protocols(safety_status)
                elif safety_status['safety_level'] in [SafetyLevel.EMERGENCY, SafetyLevel.CRITICAL]:
                    self._execute_emergency_protocols(safety_status)

                time.sleep(0.1)  # Check every 100ms

            except Exception as e:
                print(f"Error in safety protocol execution: {e}")
                time.sleep(1.0)  # Longer sleep on error

    def _execute_warning_protocols(self, safety_status: Dict):
        """
        Execute warning-level safety protocols
        """
        print(f"Warning protocols executed: {safety_status}")

    def _execute_emergency_protocols(self, safety_status: Dict):
        """
        Execute emergency safety protocols
        """
        print(f"Emergency protocols executed: {safety_status}")

        # Trigger emergency stop
        self.trigger_emergency_stop()

        # Log violation
        self._log_safety_violation(safety_status)

    def trigger_emergency_stop(self):
        """
        Trigger emergency stop across all systems
        """
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            print("EMERGENCY STOP ACTIVATED")

            # Stop all motion
            self._stop_all_motion()

            # Disable all actuators
            self._disable_all_actuators()

            # Log emergency event
            self._log_emergency_event()

    def clear_emergency_stop(self):
        """
        Clear emergency stop condition
        """
        self.emergency_stop_active = False
        print("Emergency stop cleared")

    def _stop_all_motion(self):
        """
        Stop all robot motion
        """
        # This would interface with motion control systems
        # For now, just log
        print("Stopping all motion")

    def _disable_all_actuators(self):
        """
        Disable all actuators
        """
        # This would interface with actuator control systems
        # For now, just log
        print("Disabling all actuators")

    def _log_safety_violation(self, violation_details: Dict):
        """
        Log safety violation
        """
        violation = {
            'timestamp': time.time(),
            'details': violation_details,
            'emergency_stop_activated': self.emergency_stop_active
        }
        self.violation_log.append(violation)

    def _log_emergency_event(self):
        """
        Log emergency event
        """
        emergency_event = {
            'timestamp': time.time(),
            'type': 'EMERGENCY_STOP',
            'safety_status': self.check_safety_status()
        }
        self.violation_log.append(emergency_event)

    def register_safety_zone(self, zone_name: str, zone_config: Dict):
        """
        Register a new safety zone
        """
        self.safety_zones[zone_name] = zone_config

    def set_safety_limit(self, limit_name: str, limit_value: Any):
        """
        Set a safety limit
        """
        self.safety_limits[limit_name] = limit_value

    def get_safety_report(self) -> str:
        """
        Generate safety report
        """
        report_lines = []
        report_lines.append("=== SAFETY REPORT ===")
        report_lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Emergency Stop: {'ACTIVE' if self.emergency_stop_active else 'INACTIVE'}")

        # Safety status
        status = self.check_safety_status()
        report_lines.append(f"Overall Safety Level: {status['safety_level'].value}")

        # Violations
        if status['violations']:
            report_lines.append("VIOLATIONS:")
            for violation in status['violations']:
                report_lines.append(f"  - {violation['monitor']}: {violation['level'].value}")
        else:
            report_lines.append("No violations detected")

        # Recent events
        recent_violations = self.violation_log[-10:]  # Last 10 events
        if recent_violations:
            report_lines.append("RECENT EVENTS:")
            for event in recent_violations:
                timestamp = time.strftime('%H:%M:%S', time.localtime(event['timestamp']))
                event_type = event.get('type', 'VIOLATION')
                report_lines.append(f"  {timestamp}: {event_type}")

        report_lines.append("=== END REPORT ===")

        return "\n".join(report_lines)


class CollisionMonitor:
    """
    Collision detection and monitoring system
    """

    def __init__(self):
        self.is_monitoring = False
        self.collision_threshold = 0.1  # meters
        self.last_scan_time = 0
        self.collision_detected = False
        self.collision_objects = []

    def start_monitoring(self):
        """Start collision monitoring"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_collisions, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop collision monitoring"""
        self.is_monitoring = False

    def _monitor_collisions(self):
        """Monitor for collisions in background thread"""
        while self.is_monitoring:
            try:
                # Get current sensor data (LIDAR, cameras, etc.)
                sensor_data = self._get_sensor_data()

                # Check for collisions
                self.collision_detected = self._check_collisions(sensor_data)

                if self.collision_detected:
                    self._handle_collision()

                time.sleep(0.05)  # Check every 50ms

            except Exception as e:
                print(f"Error in collision monitoring: {e}")
                time.sleep(0.1)

    def _get_sensor_data(self):
        """Get sensor data for collision detection"""
        # This would interface with actual sensors
        # For now, return mock data
        return {
            'lidar_ranges': [1.0] * 360,  # 360 degree scan
            'camera_data': None,
            'robot_pose': [0, 0, 0, 0, 0, 0]  # [x, y, z, rx, ry, rz]
        }

    def _check_collisions(self, sensor_data):
        """Check if collisions are detected"""
        # Check LIDAR data for obstacles
        lidar_ranges = sensor_data.get('lidar_ranges', [])
        if lidar_ranges:
            min_range = min(lidar_ranges) if lidar_ranges else float('inf')
            return min_range < self.collision_threshold

        # Check camera data for obstacles (simplified)
        camera_data = sensor_data.get('camera_data')
        if camera_data:
            # This would perform computer vision-based collision detection
            pass

        return False

    def _handle_collision(self):
        """Handle collision detection"""
        print("COLLISION DETECTED!")
        # This would trigger collision avoidance protocols
        # For now, just log

    def get_status(self):
        """Get collision monitor status"""
        return {
            'safety_level': SafetyLevel.ALERT if self.collision_detected else SafetyLevel.SAFE,
            'collision_detected': self.collision_detected,
            'collision_objects': self.collision_objects,
            'last_check_time': self.last_scan_time
        }


class ForceMonitor:
    """
    Force monitoring system
    """

    def __init__(self):
        self.is_monitoring = False
        self.force_thresholds = {
            'contact_force': 50.0,  # Newtons
            'gripper_force': 30.0,  # Newtons
            'torso_force': 100.0    # Newtons
        }
        self.current_forces = {}
        self.force_violations = []

    def start_monitoring(self):
        """Start force monitoring"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_forces, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop force monitoring"""
        self.is_monitoring = False

    def _monitor_forces(self):
        """Monitor forces in background thread"""
        while self.is_monitoring:
            try:
                # Get force sensor data
                force_data = self._get_force_data()

                # Check force limits
                violations = self._check_force_limits(force_data)

                if violations:
                    self.force_violations.extend(violations)

                time.sleep(0.01)  # Check every 10ms

            except Exception as e:
                print(f"Error in force monitoring: {e}")
                time.sleep(0.1)

    def _get_force_data(self):
        """Get force sensor data"""
        # This would interface with actual force sensors
        # For now, return mock data
        return {
            'contact_force': [0.1, 0.2, 0.1],  # [x, y, z]
            'gripper_force': 5.0,
            'torso_force': [0.5, 0.3, 0.2]
        }

    def _check_force_limits(self, force_data):
        """Check if forces exceed limits"""
        violations = []

        for force_type, threshold in self.force_thresholds.items():
            if force_type in force_data:
                current_force = force_data[force_type]

                if isinstance(current_force, list):
                    force_magnitude = sum(f**2 for f in current_force)**0.5
                else:
                    force_magnitude = abs(current_force)

                if force_magnitude > threshold:
                    violations.append({
                        'type': force_type,
                        'current': force_magnitude,
                        'threshold': threshold,
                        'timestamp': time.time()
                    })

        return violations

    def get_status(self):
        """Get force monitor status"""
        max_violation = max(
            (v['current'] for v in self.force_violations[-10:]), default=0
        ) if self.force_violations else 0

        safety_level = SafetyLevel.SAFE
        if max_violation > 0:
            safety_level = SafetyLevel.WARNING if max_violation < 2 * max(self.force_thresholds.values()) else SafetyLevel.EMERGENCY

        return {
            'safety_level': safety_level,
            'current_forces': self.current_forces,
            'recent_violations': self.force_violations[-5:],
            'max_force_violation': max_violation
        }


class VelocityMonitor:
    """
    Velocity monitoring system
    """

    def __init__(self):
        self.is_monitoring = False
        self.velocity_limits = {
            'linear': 1.0,      # m/s
            'angular': 0.5,     # rad/s
            'joint_velocity': 2.0  # rad/s
        }
        self.current_velocities = {}
        self.velocity_violations = []

    def start_monitoring(self):
        """Start velocity monitoring"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_velocities, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop velocity monitoring"""
        self.is_monitoring = False

    def _monitor_velocities(self):
        """Monitor velocities in background thread"""
        while self.is_monitoring:
            try:
                # Get velocity data
                velocity_data = self._get_velocity_data()

                # Check velocity limits
                violations = self._check_velocity_limits(velocity_data)

                if violations:
                    self.velocity_violations.extend(violations)

                time.sleep(0.01)  # Check every 10ms

            except Exception as e:
                print(f"Error in velocity monitoring: {e}")
                time.sleep(0.1)

    def _get_velocity_data(self):
        """Get velocity sensor data"""
        # This would interface with actual velocity sensors
        # For now, return mock data
        return {
            'linear_velocity': [0.1, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.1],
            'joint_velocities': [0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0]  # 7 joints
        }

    def _check_velocity_limits(self, velocity_data):
        """Check if velocities exceed limits"""
        violations = []

        # Check linear velocity
        lin_vel = velocity_data.get('linear_velocity', [0, 0, 0])
        lin_speed = sum(v**2 for v in lin_vel)**0.5
        if lin_speed > self.velocity_limits['linear']:
            violations.append({
                'type': 'linear_velocity',
                'current': lin_speed,
                'threshold': self.velocity_limits['linear'],
                'timestamp': time.time()
            })

        # Check angular velocity
        ang_vel = velocity_data.get('angular_velocity', [0, 0, 0])
        ang_speed = sum(v**2 for v in ang_vel)**0.5
        if ang_speed > self.velocity_limits['angular']:
            violations.append({
                'type': 'angular_velocity',
                'current': ang_speed,
                'threshold': self.velocity_limits['angular'],
                'timestamp': time.time()
            })

        # Check joint velocities
        joint_vels = velocity_data.get('joint_velocities', [])
        for i, vel in enumerate(joint_vels):
            if abs(vel) > self.velocity_limits['joint_velocity']:
                violations.append({
                    'type': f'joint_velocity_{i}',
                    'current': abs(vel),
                    'threshold': self.velocity_limits['joint_velocity'],
                    'timestamp': time.time()
                })

        return violations

    def get_status(self):
        """Get velocity monitor status"""
        max_violation = max(
            (v['current'] for v in self.velocity_violations[-10:]), default=0
        ) if self.velocity_violations else 0

        safety_level = SafetyLevel.SAFE
        if max_violation > 0:
            safety_level = SafetyLevel.WARNING

        return {
            'safety_level': safety_level,
            'current_velocities': self.current_velocities,
            'recent_violations': self.velocity_violations[-5:],
            'max_velocity_violation': max_violation
        }


class EmergencyStopMonitor:
    """
    Emergency stop monitoring system
    """

    def __init__(self):
        self.is_monitoring = False
        self.emergency_stop_pressed = False
        self.hardware_es_button = None
        self.software_es_trigger = False

    def start_monitoring(self):
        """Start emergency stop monitoring"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_emergency_stop, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop emergency stop monitoring"""
        self.is_monitoring = False

    def _monitor_emergency_stop(self):
        """Monitor emergency stop in background thread"""
        while self.is_monitoring:
            try:
                # Check hardware emergency stop button
                hw_es_pressed = self._check_hardware_emergency_stop()

                # Check software emergency stop trigger
                sw_es_triggered = self.software_es_trigger

                # Update emergency stop status
                self.emergency_stop_pressed = hw_es_pressed or sw_es_triggered

                time.sleep(0.001)  # Check every 1ms (high priority)

            except Exception as e:
                print(f"Error in emergency stop monitoring: {e}")
                time.sleep(0.01)

    def _check_hardware_emergency_stop(self):
        """Check hardware emergency stop button"""
        # This would interface with actual hardware
        # For now, return False (no emergency stop pressed)
        return False

    def trigger_software_emergency_stop(self):
        """Trigger software emergency stop"""
        self.software_es_trigger = True

    def clear_software_emergency_stop(self):
        """Clear software emergency stop"""
        self.software_es_trigger = False

    def get_status(self):
        """Get emergency stop monitor status"""
        return {
            'safety_level': SafetyLevel.CRITICAL if self.emergency_stop_pressed else SafetyLevel.SAFE,
            'emergency_stop_active': self.emergency_stop_pressed,
            'hardware_es_pressed': self.hardware_es_button,
            'software_es_triggered': self.software_es_trigger
        }
```

## Performance Integration

### Performance Monitoring and Optimization

```python
# performance_integration.py
import time
import threading
import psutil
import GPUtil
from collections import deque
import numpy as np

class PerformanceMonitor:
    """
    Performance monitoring for integrated systems
    """

    def __init__(self, update_interval=0.1):
        self.update_interval = update_interval
        self.is_monitoring = False

        # Performance metrics
        self.metrics = {
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'gpu_usage': deque(maxlen=1000),
            'gpu_memory': deque(maxlen=1000),
            'control_loop_time': deque(maxlen=1000),
            'sensor_update_time': deque(maxlen=1000),
            'communication_latency': deque(maxlen=1000),
            'throughput': deque(maxlen=1000)
        }

        # Performance thresholds
        self.thresholds = {
            'cpu_usage_max': 80.0,      # %
            'memory_usage_max': 85.0,   # %
            'gpu_usage_max': 85.0,      # %
            'control_loop_time_max': 0.02,  # seconds (50Hz)
            'sensor_update_time_max': 0.05, # seconds (20Hz)
            'communication_latency_max': 0.01  # seconds (10ms)
        }

        # Performance counters
        self.message_counts = {}
        self.error_counts = {}
        self.warning_counts = {}

    def start_monitoring(self):
        """Start performance monitoring"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False

    def _monitor_performance(self):
        """Monitor performance in background thread"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Collect application metrics
                self._collect_application_metrics()

                time.sleep(self.update_interval)

            except Exception as e:
                print(f"Error in performance monitoring: {e}")
                time.sleep(0.1)

    def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.metrics['cpu_usage'].append(cpu_percent)

        # Memory usage
        memory_percent = psutil.virtual_memory().percent
        self.metrics['memory_usage'].append(memory_percent)

        # GPU usage (if available)
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Use first GPU
            self.metrics['gpu_usage'].append(gpu.load * 100)
            self.metrics['gpu_memory'].append(gpu.memoryUtil * 100)
        else:
            self.metrics['gpu_usage'].append(0)
            self.metrics['gpu_memory'].append(0)

    def _collect_application_metrics(self):
        """Collect application-level performance metrics"""
        # This would interface with application metrics
        # For now, append dummy values
        self.metrics['control_loop_time'].append(0.01)  # 10ms average
        self.metrics['sensor_update_time'].append(0.02)  # 20ms average
        self.metrics['communication_latency'].append(0.005)  # 5ms average
        self.metrics['throughput'].append(1000)  # 1000 messages/sec

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}

        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'current': values[-1] if values else 0,
                    'average': np.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'std': np.std(values) if len(values) > 1 else 0
                }
            else:
                summary[metric_name] = {
                    'current': 0,
                    'average': 0,
                    'min': 0,
                    'max': 0,
                    'std': 0
                }

        # Add performance status
        summary['performance_status'] = self._assess_performance_status(summary)

        return summary

    def _assess_performance_status(self, summary: Dict) -> str:
        """Assess overall performance status"""
        issues = []

        # Check CPU usage
        if summary['cpu_usage']['average'] > self.thresholds['cpu_usage_max']:
            issues.append('high_cpu_usage')

        # Check memory usage
        if summary['memory_usage']['average'] > self.thresholds['memory_usage_max']:
            issues.append('high_memory_usage')

        # Check GPU usage
        if summary['gpu_usage']['average'] > self.thresholds['gpu_usage_max']:
            issues.append('high_gpu_usage')

        # Check control loop timing
        if summary['control_loop_time']['average'] > self.thresholds['control_loop_time_max']:
            issues.append('slow_control_loop')

        # Check sensor update timing
        if summary['sensor_update_time']['average'] > self.thresholds['sensor_update_time_max']:
            issues.append('slow_sensor_updates')

        # Check communication latency
        if summary['communication_latency']['average'] > self.thresholds['communication_latency_max']:
            issues.append('high_communication_latency')

        if issues:
            return 'degraded'
        else:
            return 'optimal'

    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts"""
        alerts = []
        summary = self.get_performance_summary()

        for metric_name, data in summary.items():
            threshold_name = f"{metric_name.replace(' ', '_')}_max"
            if threshold_name in self.thresholds:
                if data['current'] > self.thresholds[threshold_name]:
                    alerts.append({
                        'metric': metric_name,
                        'current': data['current'],
                        'threshold': self.thresholds[threshold_name],
                        'type': 'performance_warning'
                    })

        return alerts

    def add_message_count(self, topic_name: str, count: int = 1):
        """Add to message count for a topic"""
        if topic_name not in self.message_counts:
            self.message_counts[topic_name] = 0
        self.message_counts[topic_name] += count

    def add_error(self, error_type: str):
        """Add to error count"""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

    def add_warning(self, warning_type: str):
        """Add to warning count"""
        if warning_type not in self.warning_counts:
            self.warning_counts[warning_type] = 0
        self.warning_counts[warning_type] += 1

    def get_throughput_metrics(self) -> Dict[str, Any]:
        """Get throughput metrics"""
        return {
            'messages_per_second': {
                topic: count / (len(self.metrics['throughput']) * self.update_interval)
                for topic, count in self.message_counts.items()
            },
            'error_rate': sum(self.error_counts.values()) / max(sum(self.message_counts.values()), 1),
            'warning_rate': sum(self.warning_counts.values()) / max(sum(self.message_counts.values()), 1)
        }


class SystemOptimizer:
    """
    System optimization for performance
    """

    def __init__(self, performance_monitor):
        self.monitor = performance_monitor
        self.optimization_strategies = {}
        self.is_optimizing = False

    def register_optimization_strategy(self, name: str, strategy_func: Callable):
        """Register an optimization strategy"""
        self.optimization_strategies[name] = strategy_func

    def start_optimization(self):
        """Start automatic optimization"""
        self.is_optimizing = True
        self.optimization_thread = threading.Thread(target=self._optimize_system, daemon=True)
        self.optimization_thread.start()

    def stop_optimization(self):
        """Stop automatic optimization"""
        self.is_optimizing = False

    def _optimize_system(self):
        """Optimize system in background thread"""
        while self.is_optimizing:
            try:
                # Get current performance metrics
                perf_summary = self.monitor.get_performance_summary()
                alerts = self.monitor.get_performance_alerts()

                # Apply optimizations based on alerts
                for alert in alerts:
                    self._apply_optimization_for_alert(alert)

                time.sleep(1.0)  # Optimize every second

            except Exception as e:
                print(f"Error in system optimization: {e}")
                time.sleep(1.0)

    def _apply_optimization_for_alert(self, alert: Dict[str, Any]):
        """Apply optimization based on performance alert"""
        metric = alert['metric']

        if 'cpu' in metric:
            self._optimize_cpu_usage()
        elif 'memory' in metric:
            self._optimize_memory_usage()
        elif 'gpu' in metric:
            self._optimize_gpu_usage()
        elif 'control_loop' in metric:
            self._optimize_control_loop()
        elif 'sensor' in metric:
            self._optimize_sensor_processing()

    def _optimize_cpu_usage(self):
        """Optimize CPU usage"""
        # This could involve:
        # - Reducing update rates for non-critical systems
        # - Offloading computation to GPU
        # - Threading optimizations
        # - Algorithm optimizations
        print("Applying CPU usage optimization")

    def _optimize_memory_usage(self):
        """Optimize memory usage"""
        # This could involve:
        # - Memory pooling
        # - Garbage collection optimization
        # - Data structure optimization
        print("Applying memory usage optimization")

    def _optimize_gpu_usage(self):
        """Optimize GPU usage"""
        # This could involve:
        # - Batch size optimization
        # - Memory management
        # - Kernel optimization
        print("Applying GPU usage optimization")

    def _optimize_control_loop(self):
        """Optimize control loop performance"""
        # This could involve:
        # - Control algorithm simplification
        # - Predictive control
        # - Sampling rate optimization
        print("Applying control loop optimization")

    def _optimize_sensor_processing(self):
        """Optimize sensor processing performance"""
        # This could involve:
        # - Sensor fusion optimization
        # - Data compression
        # - Parallel processing
        print("Applying sensor processing optimization")

    def suggest_optimizations(self, perf_summary: Dict) -> List[str]:
        """Suggest optimizations based on performance summary"""
        suggestions = []

        if perf_summary['cpu_usage']['average'] > 70:
            suggestions.append("Consider reducing update rates for non-critical systems")
            suggestions.append("Implement more efficient algorithms for CPU-intensive tasks")

        if perf_summary['memory_usage']['average'] > 80:
            suggestions.append("Implement memory pooling for frequently allocated objects")
            suggestions.append("Consider data compression for large datasets")

        if perf_summary['control_loop_time']['average'] > 0.015:
            suggestions.append("Optimize control algorithms for better performance")
            suggestions.append("Consider predictive control techniques")

        if perf_summary['communication_latency']['average'] > 0.008:
            suggestions.append("Optimize network configuration and protocols")
            suggestions.append("Consider message batching for high-frequency communications")

        return suggestions


def integrate_system_components():
    """
    Integrate all system components with monitoring and safety
    """
    print("Starting system integration...")

    # Create integration manager
    integration_manager = SystemIntegrationManager()

    # Register components
    integration_manager.register_component('motor_controller', ComponentSpec(
        name='motor_controller',
        type='actuator',
        interfaces=['can_motor_control', 'power_supply'],
        dependencies=[],
        resource_requirements={'cpu': 10, 'memory': 100},
        safety_requirements=['velocity_limiting', 'force_limiting'],
        performance_requirements={'update_rate': 1000}
    ))

    integration_manager.register_component('camera_system', ComponentSpec(
        name='camera_system',
        type='sensor',
        interfaces=['ethernet_video_stream', 'power_usb'],
        dependencies=['motor_controller'],
        resource_requirements={'cpu': 20, 'memory': 500, 'gpu': 30},
        safety_requirements=['vision_processing'],
        performance_requirements={'update_rate': 30, 'resolution': '640x480'}
    ))

    integration_manager.register_component('lidar_system', ComponentSpec(
        name='lidar_system',
        type='sensor',
        interfaces=['serial_lidar', 'power_supply'],
        dependencies=[],
        resource_requirements={'cpu': 15, 'memory': 200},
        safety_requirements=['collision_detection'],
        performance_requirements={'update_rate': 10, 'range': 25.0}
    ))

    integration_manager.register_component('control_system', ComponentSpec(
        name='control_system',
        type='control',
        interfaces=['ethernet_ros', 'can_motor_control'],
        dependencies=['motor_controller', 'camera_system', 'lidar_system'],
        resource_requirements={'cpu': 30, 'memory': 300},
        safety_requirements=['emergency_stop', 'safety_limits'],
        performance_requirements={'update_rate': 100}
    ))

    # Setup hardware interfaces
    if not integration_manager.setup_hardware_interfaces():
        print("Failed to setup hardware interfaces")
        return False

    # Establish component interfaces
    integration_manager.establish_component_interfaces()

    # Verify dependencies
    if not integration_manager.verify_component_dependencies():
        print("Component dependencies not satisfied")
        return False

    # Initialize components
    integration_manager.initialize_components()

    # Create and integrate safety system
    safety_manager = SafetySystemManager()
    safety_manager.start_safety_monitoring()

    # Create and integrate performance monitoring
    performance_monitor = PerformanceMonitor()
    performance_monitor.start_monitoring()

    # Create system optimizer
    optimizer = SystemOptimizer(performance_monitor)
    optimizer.start_optimization()

    # Start integration monitoring
    integration_manager.start_integration_monitoring()

    print("System integration completed successfully")

    # Run validation
    success, results = integration_manager.run_integration_validation()
    print(f"Integration validation: {'PASSED' if success else 'FAILED'}")
    print(f"Validation results: {results}")

    return success


def run_system_integration():
    """
    Main function to run system integration
    """
    success = integrate_system_components()

    if success:
        print("\nSystem integration successful!")
        print("All components integrated and validated.")
    else:
        print("\nSystem integration failed!")
        print("Check error messages above.")

    return success


if __name__ == "__main__":
    success = run_system_integration()
    exit(0 if success else 1)
```

## Integration Validation and Testing

### Comprehensive Integration Testing

```python
# integration_validation.py
import unittest
import time
from typing import Dict, Any

class IntegrationTestSuite(unittest.TestCase):
    """
    Comprehensive test suite for system integration
    """

    def setUp(self):
        """Setup test environment"""
        self.integration_manager = SystemIntegrationManager()
        self.safety_manager = SafetySystemManager()
        self.performance_monitor = PerformanceMonitor()

    def test_hardware_interface_integration(self):
        """Test hardware interface integration"""
        # Setup hardware interfaces
        interfaces_ok = self.integration_manager.setup_hardware_interfaces()
        self.assertTrue(interfaces_ok, "Hardware interfaces setup failed")

        # Verify interfaces
        verification_ok = self.integration_manager.verify_interfaces()
        self.assertTrue(verification_ok, "Interface verification failed")

    def test_component_communication(self):
        """Test component-to-component communication"""
        # Register test components
        test_spec = ComponentSpec(
            name='test_component',
            type='test',
            interfaces=['ethernet_test', 'can_test'],
            dependencies=[],
            resource_requirements={'cpu': 1, 'memory': 1},
            safety_requirements=[],
            performance_requirements={}
        )

        self.integration_manager.register_component('test_component', test_spec)

        # Verify communication
        communication_ok = self.integration_manager.verify_component_communication()
        self.assertTrue(communication_ok, "Component communication failed")

    def test_safety_system_integration(self):
        """Test safety system integration"""
        # Integrate safety systems
        self.integration_manager.integrate_safety_systems()

        # Verify safety systems
        safety_ok = self.integration_manager.verify_safety_systems()
        self.assertTrue(safety_ok, "Safety system verification failed")

    def test_performance_integration(self):
        """Test performance monitoring integration"""
        # Integrate performance systems
        self.integration_manager.integrate_performance_systems()

        # Verify performance metrics
        performance_ok = self.integration_manager.verify_performance_metrics()
        self.assertTrue(performance_ok, "Performance verification failed")

    def test_system_stability(self):
        """Test system stability over time"""
        # Run stability test
        stability_ok = self.integration_manager.verify_system_stability()
        self.assertTrue(stability_ok, "System stability test failed")

    def test_emergency_stop_functionality(self):
        """Test emergency stop functionality"""
        # Verify safety manager is working
        self.assertIsNotNone(self.safety_manager, "Safety manager not initialized")

        # Test emergency stop activation
        self.safety_manager.trigger_emergency_stop()
        time.sleep(0.1)  # Allow time for ES to propagate

        status = self.safety_manager.check_safety_status()
        self.assertTrue(status['emergency_stop'], "Emergency stop not activated")

        # Test emergency stop clearing
        self.safety_manager.clear_emergency_stop()
        time.sleep(0.1)

        status = self.safety_manager.check_safety_status()
        self.assertFalse(status['emergency_stop'], "Emergency stop not cleared")

    def test_collision_detection(self):
        """Test collision detection functionality"""
        # This would test actual collision detection
        # For now, just verify monitor exists and can be queried
        collision_monitor = CollisionMonitor()
        collision_monitor.start_monitoring()
        time.sleep(0.1)

        status = collision_monitor.get_status()
        self.assertIsNotNone(status, "Collision monitor status not available")

        collision_monitor.stop_monitoring()

    def test_force_monitoring(self):
        """Test force monitoring functionality"""
        # Test force monitoring
        force_monitor = ForceMonitor()
        force_monitor.start_monitoring()
        time.sleep(0.1)

        status = force_monitor.get_status()
        self.assertIsNotNone(status, "Force monitor status not available")

        force_monitor.stop_monitoring()

    def test_velocity_monitoring(self):
        """Test velocity monitoring functionality"""
        # Test velocity monitoring
        velocity_monitor = VelocityMonitor()
        velocity_monitor.start_monitoring()
        time.sleep(0.1)

        status = velocity_monitor.get_status()
        self.assertIsNotNone(status, "Velocity monitor status not available")

        velocity_monitor.stop_monitoring()

    def test_communication_reliability(self):
        """Test communication reliability"""
        # Create test communication system
        comm_system = create_robot_communication_system()
        self.assertIsNotNone(comm_system, "Communication system not created")

        # Test message broadcasting
        test_message = {'type': 'test', 'data': 'integration_test'}
        broadcast_success = comm_system.broadcast_message(test_message)
        self.assertTrue(broadcast_success, "Message broadcasting failed")

    def test_performance_monitoring(self):
        """Test performance monitoring functionality"""
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        time.sleep(1.0)  # Allow some time for metrics collection

        # Get performance summary
        summary = self.performance_monitor.get_performance_summary()
        self.assertIsNotNone(summary, "Performance summary not available")

        # Check that metrics are being collected
        self.assertGreater(len(self.performance_monitor.metrics['cpu_usage']), 0,
                          "CPU usage metrics not being collected")

        self.performance_monitor.stop_monitoring()

    def test_system_shutdown_procedure(self):
        """Test proper system shutdown"""
        # Test that all systems can be properly shut down
        self.safety_manager.stop_safety_monitoring()
        self.performance_monitor.stop_monitoring()

        # Verify shutdown
        self.assertFalse(self.safety_manager.is_active, "Safety manager not stopped")
        self.assertFalse(self.performance_monitor.is_monitoring, "Performance monitor not stopped")

    def tearDown(self):
        """Cleanup after tests"""
        try:
            self.safety_manager.stop_safety_monitoring()
        except:
            pass

        try:
            self.performance_monitor.stop_monitoring()
        except:
            pass


class IntegrationValidator:
    """
    Validates integration completeness and correctness
    """

    def __init__(self):
        self.validation_results = {}
        self.compliance_checks = []
        self.performance_benchmarks = {}

    def run_comprehensive_validation(self, system_components):
        """
        Run comprehensive validation of integrated system
        """
        validation_results = {
            'component_integration': self.validate_component_integration(system_components),
            'interface_compatibility': self.validate_interface_compatibility(system_components),
            'safety_systems': self.validate_safety_systems(system_components),
            'performance_metrics': self.validate_performance_metrics(system_components),
            'communication_reliability': self.validate_communication_reliability(system_components),
            'error_handling': self.validate_error_handling(system_components)
        }

        overall_success = all(validation_results.values())

        self.validation_results = validation_results
        return overall_success, validation_results

    def validate_component_integration(self, components):
        """
        Validate that all components are properly integrated
        """
        for name, comp_info in components.items():
            if comp_info['instance'] is None:
                print(f"Component {name} not initialized")
                return False

            if comp_info['status'] != 'ready':
                print(f"Component {name} not ready: {comp_info['status']}")
                return False

        return True

    def validate_interface_compatibility(self, components):
        """
        Validate that component interfaces are compatible
        """
        # Check that all required interfaces are available
        for name, comp_info in components.items():
            for interface in comp_info['spec'].interfaces:
                if interface not in self.integration_manager.interfaces:
                    print(f"Interface {interface} not available for component {name}")
                    return False

        return True

    def validate_safety_systems(self, components):
        """
        Validate safety system functionality
        """
        safety_status = self.safety_manager.check_safety_status()

        # Check that safety level is appropriate
        if safety_status['safety_level'] != SafetyLevel.SAFE:
            print(f"Safety level not safe: {safety_status['safety_level']}")
            return False

        # Check that all safety monitors are operational
        for monitor_name, monitor_status in safety_status['monitors_status'].items():
            if monitor_status['safety_level'] == SafetyLevel.CRITICAL:
                print(f"Critical safety issue in {monitor_name}")
                return False

        return True

    def validate_performance_metrics(self, components):
        """
        Validate that performance metrics meet requirements
        """
        perf_summary = self.performance_monitor.get_performance_summary()

        # Check CPU usage
        if perf_summary['cpu_usage']['average'] > 80:
            print(f"High CPU usage: {perf_summary['cpu_usage']['average']:.1f}%")
            return False

        # Check memory usage
        if perf_summary['memory_usage']['average'] > 85:
            print(f"High memory usage: {perf_summary['memory_usage']['average']:.1f}%")
            return False

        # Check control loop timing
        if perf_summary['control_loop_time']['average'] > 0.02:  # 20ms max
            print(f"Slow control loop: {perf_summary['control_loop_time']['average']:.3f}s")
            return False

        return True

    def validate_communication_reliability(self, components):
        """
        Validate communication system reliability
        """
        # Test communication by sending messages between components
        test_message = {'type': 'validation', 'data': 'test_message'}

        # Try broadcasting to all systems
        comm_success = self.comm_manager.broadcast_message(test_message)
        if not comm_success:
            print("Communication system not reliable")
            return False

        # Check for message delivery
        time.sleep(0.1)  # Allow time for message processing

        return True

    def validate_error_handling(self, components):
        """
        Validate error handling capabilities
        """
        # Simulate errors and verify system response
        error_conditions = [
            self._test_sensor_error_handling,
            self._test_actuator_error_handling,
            self._test_communication_error_handling
        ]

        for error_test in error_conditions:
            if not error_test():
                return False

        return True

    def _test_sensor_error_handling(self):
        """
        Test sensor error handling
        """
        # This would simulate sensor errors and verify graceful handling
        print("Testing sensor error handling...")
        return True

    def _test_actuator_error_handling(self):
        """
        Test actuator error handling
        """
        # This would simulate actuator errors and verify safety responses
        print("Testing actuator error handling...")
        return True

    def _test_communication_error_handling(self):
        """
        Test communication error handling
        """
        # This would simulate communication errors and verify fallback behavior
        print("Testing communication error handling...")
        return True

    def generate_validation_report(self):
        """
        Generate comprehensive validation report
        """
        report = []
        report.append("# System Integration Validation Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overall status
        overall_success = all(self.validation_results.values())
        report.append(f"## Overall Status: {'PASS' if overall_success else 'FAIL'}")
        report.append("")

        # Detailed results
        for test_name, result in self.validation_results.items():
            status = "PASS" if result else "FAIL"
            report.append(f"### {test_name.replace('_', ' ').title()}: {status}")

        # Performance metrics
        if hasattr(self, 'performance_monitor'):
            perf_summary = self.performance_monitor.get_performance_summary()
            report.append("")
            report.append("## Performance Metrics:")
            for metric, values in perf_summary.items():
                if isinstance(values, dict) and 'average' in values:
                    report.append(f"- {metric}: {values['average']:.2f} (avg), {values['max']:.2f} (max)")

        # Safety status
        if hasattr(self, 'safety_manager'):
            safety_status = self.safety_manager.check_safety_status()
            report.append("")
            report.append(f"## Safety Status: {safety_status['safety_level'].value}")
            if safety_status['violations']:
                report.append("### Violations:")
                for violation in safety_status['violations']:
                    report.append(f"- {violation['monitor']}: {violation['level'].value}")

        return "\n".join(report)


def main():
    """
    Main integration validation function
    """
    # Perform system integration
    integration_success = integrate_system_components()

    if not integration_success:
        print("System integration failed, cannot proceed with validation")
        return False

    # Create validator
    validator = IntegrationValidator()

    # Run comprehensive validation
    validation_success, results = validator.run_comprehensive_validation(
        integration_manager.components
    )

    # Generate report
    report = validator.generate_validation_report()
    print(report)

    # Save report to file
    with open('integration_validation_report.txt', 'w') as f:
        f.write(report)

    print("Integration validation report saved to integration_validation_report.txt")

    return validation_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

## Week Summary

This section covered comprehensive system integration for Physical AI and humanoid robotics applications. We explored hardware integration architecture, communication protocol integration, safety system integration, performance monitoring, and validation procedures. The integration process requires careful attention to interfaces, timing, safety, and performance to ensure that all components work together effectively in a cohesive robotic system. Proper system integration is critical for the success of complex robotic applications.