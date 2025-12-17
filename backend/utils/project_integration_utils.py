import asyncio
import json
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import uuid
from datetime import datetime
import threading
import cProfile  # For performance profiling
import pstats
from io import StringIO
import queue
import weakref

# Try to import optional modules
try:
    import psutil  # For system resource monitoring
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


class IntegrationStatus(Enum):
    """Status of integration components"""
    INITIALIZING = "initializing"
    CONNECTED = "connected"
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class ComponentType(Enum):
    """Types of system components"""
    PERCEPTION = "perception"
    PLANNING = "planning"
    CONTROL = "control"
    COMMUNICATION = "communication"
    SENSING = "sensing"
    ACTUATION = "actuation"
    LEARNING = "learning"
    VLA = "vla"  # Vision-Language-Action
    SIMULATION = "simulation"
    HARDWARE = "hardware"
    HUMANOID = "humanoid"


@dataclass
class ComponentInfo:
    """Information about a system component"""
    id: str
    name: str
    type: ComponentType
    status: IntegrationStatus
    dependencies: List[str]  # Component IDs this component depends on
    resources: Dict[str, float]  # CPU, memory, etc. usage
    last_update: datetime
    performance_metrics: Dict[str, Any]


@dataclass
class DataFlow:
    """Represents data flow between components"""
    source_component: str
    target_component: str
    data_type: str  # message type
    frequency: float  # Hz
    bandwidth: float  # bytes/s
    latency: float  # ms
    reliability: float  # 0.0 to 1.0


@dataclass
class IntegrationMetrics:
    """Performance metrics for the integrated system"""
    total_components: int
    operational_components: int
    system_cpu: float
    system_memory: float
    network_usage: float
    data_throughput: float  # messages/s
    average_latency: float  # ms
    error_rate: float
    bottleneck_components: List[str]
    system_health_score: float  # 0.0 to 1.0


class MessageBus:
    """Central message bus for component communication"""

    def __init__(self):
        self.topics = {}
        self.subscribers = {}  # topic -> list of callbacks
        self.publish_queue = queue.Queue()
        self.message_history = {}  # topic -> last N messages
        self.max_history = 100

    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from a topic"""
        if topic in self.subscribers:
            self.subscribers[topic].remove(callback)

    def publish(self, topic: str, message: Any):
        """Publish a message to a topic"""
        # Add to history
        if topic not in self.message_history:
            self.message_history[topic] = []
        self.message_history[topic].append({
            "timestamp": time.time(),
            "message": message
        })
        if len(self.message_history[topic]) > self.max_history:
            self.message_history[topic].pop(0)

        # Notify subscribers
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(message)
                except Exception as e:
                    print(f"Error in subscriber callback for topic {topic}: {e}")

    def get_topic_history(self, topic: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages from a topic"""
        if topic in self.message_history:
            return self.message_history[topic][-count:]
        return []


class ComponentManager:
    """Manages individual system components"""

    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.components = {}
        self.data_flows = []
        self.health_check_interval = 1.0  # seconds
        self.health_thread = None
        self.monitoring_active = False

    def register_component(self, component_id: str, name: str, component_type: ComponentType,
                          dependencies: List[str] = None) -> bool:
        """Register a new component"""
        if dependencies is None:
            dependencies = []

        component_info = ComponentInfo(
            id=component_id,
            name=name,
            type=component_type,
            status=IntegrationStatus.INITIALIZING,
            dependencies=dependencies,
            resources={"cpu": 0.0, "memory": 0.0},
            last_update=datetime.now(),
            performance_metrics={}
        )

        self.components[component_id] = component_info
        return True

    def update_component_status(self, component_id: str, status: IntegrationStatus):
        """Update the status of a component"""
        if component_id in self.components:
            self.components[component_id].status = status
            self.components[component_id].last_update = datetime.now()

    def update_component_resources(self, component_id: str, resources: Dict[str, float]):
        """Update resource usage for a component"""
        if component_id in self.components:
            self.components[component_id].resources.update(resources)
            self.components[component_id].last_update = datetime.now()

    def add_data_flow(self, source: str, target: str, data_type: str,
                     frequency: float = 1.0, bandwidth: float = 0.0):
        """Add a data flow between components"""
        data_flow = DataFlow(
            source_component=source,
            target_component=target,
            data_type=data_type,
            frequency=frequency,
            bandwidth=bandwidth,
            latency=0.0,
            reliability=1.0
        )
        self.data_flows.append(data_flow)

    def get_component_dependencies(self, component_id: str) -> List[str]:
        """Get dependencies for a component"""
        if component_id in self.components:
            return self.components[component_id].dependencies
        return []

    def get_component_info(self, component_id: str) -> Optional[ComponentInfo]:
        """Get information about a specific component"""
        return self.components.get(component_id)

    def get_all_components(self) -> List[ComponentInfo]:
        """Get information about all components"""
        return list(self.components.values())

    def start_monitoring(self):
        """Start component monitoring in background"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self.health_thread.start()

    def stop_monitoring(self):
        """Stop component monitoring"""
        self.monitoring_active = False
        if self.health_thread:
            self.health_thread.join(timeout=1.0)

    def _health_check_loop(self):
        """Background health check loop"""
        while self.monitoring_active:
            try:
                self._check_component_health()
                time.sleep(self.health_check_interval)
            except Exception:
                time.sleep(self.health_check_interval)  # Continue monitoring even if there's an error

    def _check_component_health(self):
        """Check health of all components"""
        for comp_id, comp_info in self.components.items():
            # Update resource usage (if psutil is available)
            if PSUTIL_AVAILABLE and psutil:
                try:
                    current_cpu = psutil.cpu_percent()
                    current_memory = psutil.virtual_memory().percent
                except:
                    # Fallback to default values if psutil fails
                    current_cpu = 0.0
                    current_memory = 0.0
            else:
                # Use default values when psutil is not available
                current_cpu = 0.0
                current_memory = 0.0

            # Update resource info
            self.update_component_resources(comp_id, {
                "cpu": current_cpu,
                "memory": current_memory
            })

            # Determine health based on resource usage and status
            if current_cpu > 90 or current_memory > 90:
                self.update_component_status(comp_id, IntegrationStatus.DEGRADED)
            elif comp_info.status == IntegrationStatus.INITIALIZING:
                self.update_component_status(comp_id, IntegrationStatus.OPERATIONAL)


class PerformanceProfiler:
    """Performance profiling and optimization tools"""

    def __init__(self):
        self.profiles = {}
        self.bottleneck_threshold = 0.8  # 80% CPU or memory usage

    def start_profiling(self, name: str):
        """Start profiling a function or operation"""
        profiler = cProfile.Profile()
        profiler.enable()
        self.profiles[name] = profiler

    def stop_profiling(self, name: str) -> Dict[str, Any]:
        """Stop profiling and return results"""
        if name not in self.profiles:
            return {}

        profiler = self.profiles[name]
        profiler.disable()

        # Get stats
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions

        # Extract performance metrics
        stats_str = s.getvalue()
        # Parse the stats to extract meaningful metrics
        lines = stats_str.split('\n')
        top_functions = []
        for line in lines[3:13]:  # Skip header lines
            if line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    func_info = {
                        "function": ' '.join(parts[5:]),
                        "calls": parts[0] if parts[0].isdigit() else parts[1],
                        "cumulative_time": parts[3] if len(parts) > 3 and parts[3].replace('.', '').isdigit() else "0.0"
                    }
                    top_functions.append(func_info)

        return {
            "top_functions": top_functions,
            "stats": stats_str
        }

    def identify_bottlenecks(self, system_metrics: IntegrationMetrics) -> List[str]:
        """Identify system bottlenecks"""
        bottlenecks = []

        # Check for high CPU usage components
        if system_metrics.system_cpu > self.bottleneck_threshold * 100:
            bottlenecks.append("High system CPU usage")

        # Check for high memory usage
        if system_metrics.system_memory > self.bottleneck_threshold * 100:
            bottlenecks.append("High system memory usage")

        # Check for high error rate
        if system_metrics.error_rate > self.bottleneck_threshold:
            bottlenecks.append("High system error rate")

        # Check for low throughput
        if system_metrics.data_throughput < 1.0:  # Less than 1 message per second
            bottlenecks.append("Low data throughput")

        return bottlenecks


class SystemValidator:
    """System validation and quality assurance tools"""

    def __init__(self):
        self.test_results = {}
        self.validation_rules = []

    def add_validation_rule(self, name: str, check_function: Callable, description: str = ""):
        """Add a validation rule"""
        self.validation_rules.append({
            "name": name,
            "check_function": check_function,
            "description": description
        })

    def run_validation_tests(self) -> Dict[str, Any]:
        """Run all validation tests"""
        results = {
            "overall_success": True,
            "test_results": {},
            "errors": [],
            "warnings": []
        }

        for rule in self.validation_rules:
            try:
                success = rule["check_function"]()
                results["test_results"][rule["name"]] = success
                if not success:
                    results["overall_success"] = False
                    results["errors"].append(f"Validation failed: {rule['name']}")
            except Exception as e:
                results["overall_success"] = False
                results["errors"].append(f"Validation error in {rule['name']}: {str(e)}")

        return results

    def validate_component_connectivity(self, component_manager: ComponentManager) -> bool:
        """Validate that all components are properly connected"""
        components = component_manager.get_all_components()
        operational_count = sum(1 for comp in components if comp.status == IntegrationStatus.OPERATIONAL)
        return operational_count == len(components)

    def validate_data_flow(self, component_manager: ComponentManager) -> bool:
        """Validate data flow between components"""
        # Check if there are any data flows defined
        return len(component_manager.data_flows) > 0

    def validate_performance_thresholds(self, metrics: IntegrationMetrics) -> bool:
        """Validate that performance metrics are within acceptable thresholds"""
        return (
            metrics.system_cpu < 80.0 and
            metrics.system_memory < 80.0 and
            metrics.error_rate < 0.1 and
            metrics.system_health_score > 0.7
        )


class IntegrationManager:
    """Main manager for project integration"""

    def __init__(self):
        self.message_bus = MessageBus()
        self.component_manager = ComponentManager(self.message_bus)
        self.performance_profiler = PerformanceProfiler()
        self.system_validator = SystemValidator()
        self.integration_metrics = None
        self.integration_active = False

        # Add default validation rules
        self._setup_default_validations()

    def _setup_default_validations(self):
        """Set up default validation rules"""
        # These will be filled in based on actual system state
        pass

    def integrate_component(self, component_id: str, name: str, component_type: ComponentType,
                          dependencies: List[str] = None) -> bool:
        """Integrate a new component into the system"""
        success = self.component_manager.register_component(
            component_id, name, component_type, dependencies
        )
        return success

    def setup_data_flow(self, source: str, target: str, data_type: str,
                       frequency: float = 1.0, bandwidth: float = 0.0):
        """Setup data flow between integrated components"""
        self.component_manager.add_data_flow(source, target, data_type, frequency, bandwidth)

    def start_integration(self):
        """Start the integrated system"""
        if not self.integration_active:
            self.component_manager.start_monitoring()
            self.integration_active = True

            # Update component statuses to operational
            for comp_id in self.component_manager.components:
                self.component_manager.update_component_status(
                    comp_id, IntegrationStatus.OPERATIONAL
                )

    def stop_integration(self):
        """Stop the integrated system"""
        self.component_manager.stop_monitoring()
        self.integration_active = False

        # Update component statuses to disconnected
        for comp_id in self.component_manager.components:
            self.component_manager.update_component_status(
                comp_id, IntegrationStatus.DISCONNECTED
            )

    def get_system_metrics(self) -> IntegrationMetrics:
        """Get current system integration metrics"""
        components = self.component_manager.get_all_components()
        operational_count = sum(1 for comp in components if comp.status == IntegrationStatus.OPERATIONAL)

        # Get system-level metrics (with fallback when psutil is not available)
        if PSUTIL_AVAILABLE and psutil:
            try:
                system_cpu = psutil.cpu_percent()
                system_memory = psutil.virtual_memory().percent
                network_io = psutil.net_io_counters()
                network_usage = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)  # MB
            except:
                # Fallback values if psutil operations fail
                system_cpu = 0.0
                system_memory = 0.0
                network_usage = 0.0
        else:
            # Use default values when psutil is not available
            system_cpu = 0.0
            system_memory = 0.0
            network_usage = 0.0

        # Calculate average latency and other metrics
        avg_latency = 0.0
        error_count = 0
        total_messages = 0

        for topic, messages in self.message_bus.message_history.items():
            total_messages += len(messages)

        # Calculate health score (simplified)
        health_score = operational_count / len(components) if components else 1.0

        # Identify bottlenecks
        bottlenecks = []
        if system_cpu > 80:
            bottlenecks.append("main_system")
        for comp in components:
            if comp.resources.get("cpu", 0) > 80:
                bottlenecks.append(comp.id)

        self.integration_metrics = IntegrationMetrics(
            total_components=len(components),
            operational_components=operational_count,
            system_cpu=system_cpu,
            system_memory=system_memory,
            network_usage=network_usage,
            data_throughput=total_messages / 10.0 if total_messages > 0 else 0.0,  # Simplified
            average_latency=avg_latency,
            error_rate=error_count / total_messages if total_messages > 0 else 0.0,
            bottleneck_components=bottlenecks,
            system_health_score=health_score
        )

        return self.integration_metrics

    def run_system_validation(self) -> Dict[str, Any]:
        """Run comprehensive system validation"""
        # Update validation rules based on current system state
        self.system_validator.validation_rules = []

        # Add validation rules for current components
        def check_connectivity():
            return self.system_validator.validate_component_connectivity(self.component_manager)

        def check_data_flow():
            return self.system_validator.validate_data_flow(self.component_manager)

        def check_performance():
            metrics = self.get_system_metrics()
            return self.system_validator.validate_performance_thresholds(metrics)

        self.system_validator.add_validation_rule(
            "component_connectivity", check_connectivity, "All components should be connected"
        )
        self.system_validator.add_validation_rule(
            "data_flow", check_data_flow, "Data flows should be established"
        )
        self.system_validator.add_validation_rule(
            "performance_thresholds", check_performance, "Performance should be within thresholds"
        )

        return self.system_validator.run_validation_tests()

    def profile_system_performance(self, operation_name: str, operation: Callable) -> Dict[str, Any]:
        """Profile performance of a system operation"""
        self.performance_profiler.start_profiling(operation_name)
        result = operation()
        profile_results = self.performance_profiler.stop_profiling(operation_name)
        return {
            "operation_result": result,
            "profile_data": profile_results
        }

    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        metrics = self.get_system_metrics()
        validation_results = self.run_system_validation()

        report = {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                "total_components": metrics.total_components,
                "operational_components": metrics.operational_components,
                "system_cpu": metrics.system_cpu,
                "system_memory": metrics.system_memory,
                "data_throughput": metrics.data_throughput,
                "system_health_score": metrics.system_health_score
            },
            "component_status": {
                comp.id: {
                    "name": comp.name,
                    "type": comp.type.value,
                    "status": comp.status.value,
                    "resources": comp.resources
                }
                for comp in self.component_manager.get_all_components()
            },
            "validation_results": validation_results,
            "bottlenecks": metrics.bottleneck_components,
            "message_bus_stats": {
                "topics_count": len(self.message_bus.subscribers),
                "total_messages": sum(len(hist) for hist in self.message_bus.message_history.values())
            }
        }

        return report

    def get_data_flow_analysis(self) -> Dict[str, Any]:
        """Get analysis of data flows in the system"""
        flows = self.component_manager.data_flows
        analysis = {
            "total_flows": len(flows),
            "by_type": {},
            "by_frequency": {},
            "components_involved": set()
        }

        for flow in flows:
            analysis["components_involved"].add(flow.source_component)
            analysis["components_involved"].add(flow.target_component)

            # Count by data type
            if flow.data_type not in analysis["by_type"]:
                analysis["by_type"][flow.data_type] = 0
            analysis["by_type"][flow.data_type] += 1

            # Count by frequency range
            freq_range = f"{int(flow.frequency // 10) * 10}-{int(flow.frequency // 10 + 1) * 10}Hz"
            if freq_range not in analysis["by_frequency"]:
                analysis["by_frequency"][freq_range] = 0
            analysis["by_frequency"][freq_range] += 1

        analysis["components_involved"] = list(analysis["components_involved"])
        return analysis


# Global integration manager instance
integration_manager = IntegrationManager()