from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
from backend.utils.project_integration_utils import (
    integration_manager, ComponentType, IntegrationStatus,
    ComponentManager, MessageBus, PerformanceProfiler, SystemValidator
)
from backend.utils.logger import logger


router = APIRouter()


class IntegrateComponentRequest(BaseModel):
    component_id: str
    name: str
    component_type: str  # perception, planning, control, etc.
    dependencies: List[str] = []


class IntegrateComponentResponse(BaseModel):
    success: bool
    message: str


class SetupDataFlowRequest(BaseModel):
    source: str
    target: str
    data_type: str
    frequency: float = 1.0
    bandwidth: float = 0.0


class SetupDataFlowResponse(BaseModel):
    success: bool
    message: str


class SystemMetricsResponse(BaseModel):
    total_components: int
    operational_components: int
    system_cpu: float
    system_memory: float
    network_usage: float
    data_throughput: float
    average_latency: float
    error_rate: float
    bottleneck_components: List[str]
    system_health_score: float
    success: bool


class ValidationResponse(BaseModel):
    overall_success: bool
    test_results: Dict[str, bool]
    errors: List[str]
    warnings: List[str]
    success: bool


class HealthReportResponse(BaseModel):
    timestamp: str
    system_metrics: Dict[str, Any]
    component_status: Dict[str, Dict[str, Any]]
    validation_results: Dict[str, Any]
    bottlenecks: List[str]
    message_bus_stats: Dict[str, Any]
    success: bool


class DataFlowAnalysisResponse(BaseModel):
    total_flows: int
    by_type: Dict[str, int]
    by_frequency: Dict[str, int]
    components_involved: List[str]
    success: bool


class StartIntegrationResponse(BaseModel):
    success: bool
    message: str


class StopIntegrationResponse(BaseModel):
    success: bool
    message: str


class SubscribeToTopicRequest(BaseModel):
    topic: str


class SubscribeToTopicResponse(BaseModel):
    success: bool
    message: str


class PublishToTopicRequest(BaseModel):
    topic: str
    message: Dict[str, Any]


class PublishToTopicResponse(BaseModel):
    success: bool
    message: str


@router.post("/integrate-component", response_model=IntegrateComponentResponse)
async def integrate_component(request: IntegrateComponentRequest):
    """
    Integrate a new component into the system
    """
    try:
        component_type = ComponentType(request.component_type.lower())

        success = integration_manager.integrate_component(
            request.component_id,
            request.name,
            component_type,
            request.dependencies
        )

        if success:
            response = IntegrateComponentResponse(
                success=True,
                message=f"Component {request.name} integrated successfully"
            )
        else:
            response = IntegrateComponentResponse(
                success=False,
                message=f"Failed to integrate component {request.name}"
            )

        logger.info(f"Component integration: {request.name}, success: {success}")

        return response

    except ValueError:
        logger.error(f"Invalid component type: {request.component_type}")
        raise HTTPException(status_code=400, detail=f"Invalid component type: {request.component_type}")
    except Exception as e:
        logger.error(f"Error integrating component: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error integrating component: {str(e)}")


@router.post("/setup-data-flow", response_model=SetupDataFlowResponse)
async def setup_data_flow(request: SetupDataFlowRequest):
    """
    Setup data flow between integrated components
    """
    try:
        integration_manager.setup_data_flow(
            request.source,
            request.target,
            request.data_type,
            request.frequency,
            request.bandwidth
        )

        response = SetupDataFlowResponse(
            success=True,
            message=f"Data flow from {request.source} to {request.target} established"
        )

        logger.info(f"Data flow established: {request.source} -> {request.target}")

        return response

    except Exception as e:
        logger.error(f"Error setting up data flow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting up data flow: {str(e)}")


@router.get("/system-metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """
    Get current system integration metrics
    """
    try:
        metrics = integration_manager.get_system_metrics()

        response = SystemMetricsResponse(
            total_components=metrics.total_components,
            operational_components=metrics.operational_components,
            system_cpu=metrics.system_cpu,
            system_memory=metrics.system_memory,
            network_usage=metrics.network_usage,
            data_throughput=metrics.data_throughput,
            average_latency=metrics.average_latency,
            error_rate=metrics.error_rate,
            bottleneck_components=metrics.bottleneck_components,
            system_health_score=metrics.system_health_score,
            success=True
        )

        logger.info(f"System metrics retrieved: {metrics.system_health_score} health score")

        return response

    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting system metrics: {str(e)}")


@router.post("/validate-system", response_model=ValidationResponse)
async def validate_system():
    """
    Run comprehensive system validation
    """
    try:
        validation_results = integration_manager.run_system_validation()

        response = ValidationResponse(
            overall_success=validation_results["overall_success"],
            test_results=validation_results["test_results"],
            errors=validation_results["errors"],
            warnings=validation_results["warnings"],
            success=True
        )

        logger.info(f"System validation completed: {validation_results['overall_success']}")

        return response

    except Exception as e:
        logger.error(f"Error running system validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running system validation: {str(e)}")


@router.get("/health-report", response_model=HealthReportResponse)
async def get_health_report():
    """
    Get comprehensive system health report
    """
    try:
        report = integration_manager.get_system_health_report()

        response = HealthReportResponse(
            timestamp=report["timestamp"],
            system_metrics=report["system_metrics"],
            component_status=report["component_status"],
            validation_results=report["validation_results"],
            bottlenecks=report["bottlenecks"],
            message_bus_stats=report["message_bus_stats"],
            success=True
        )

        logger.info(f"Health report generated with {report['system_metrics']['total_components']} components")

        return response

    except Exception as e:
        logger.error(f"Error generating health report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating health report: {str(e)}")


@router.get("/data-flow-analysis", response_model=DataFlowAnalysisResponse)
async def get_data_flow_analysis():
    """
    Get analysis of data flows in the system
    """
    try:
        analysis = integration_manager.get_data_flow_analysis()

        response = DataFlowAnalysisResponse(
            total_flows=analysis["total_flows"],
            by_type=analysis["by_type"],
            by_frequency=analysis["by_frequency"],
            components_involved=analysis["components_involved"],
            success=True
        )

        logger.info(f"Data flow analysis completed: {analysis['total_flows']} flows")

        return response

    except Exception as e:
        logger.error(f"Error getting data flow analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting data flow analysis: {str(e)}")


@router.post("/start-integration", response_model=StartIntegrationResponse)
async def start_integration():
    """
    Start the integrated system
    """
    try:
        integration_manager.start_integration()

        response = StartIntegrationResponse(
            success=True,
            message="Integrated system started successfully"
        )

        logger.info("Integrated system started")

        return response

    except Exception as e:
        logger.error(f"Error starting integration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting integration: {str(e)}")


@router.post("/stop-integration", response_model=StopIntegrationResponse)
async def stop_integration():
    """
    Stop the integrated system
    """
    try:
        integration_manager.stop_integration()

        response = StopIntegrationResponse(
            success=True,
            message="Integrated system stopped successfully"
        )

        logger.info("Integrated system stopped")

        return response

    except Exception as e:
        logger.error(f"Error stopping integration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping integration: {str(e)}")


@router.post("/subscribe", response_model=SubscribeToTopicResponse)
async def subscribe_to_topic(request: SubscribeToTopicRequest):
    """
    Subscribe to a message bus topic
    """
    try:
        def dummy_callback(message):
            # In a real implementation, this would process the message
            logger.info(f"Received message on {request.topic}: {message}")

        integration_manager.message_bus.subscribe(request.topic, dummy_callback)

        response = SubscribeToTopicResponse(
            success=True,
            message=f"Subscribed to topic {request.topic}"
        )

        logger.info(f"Subscribed to topic: {request.topic}")

        return response

    except Exception as e:
        logger.error(f"Error subscribing to topic: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error subscribing to topic: {str(e)}")


@router.post("/publish", response_model=PublishToTopicResponse)
async def publish_to_topic(request: PublishToTopicRequest):
    """
    Publish a message to a message bus topic
    """
    try:
        integration_manager.message_bus.publish(request.topic, request.message)

        response = PublishToTopicResponse(
            success=True,
            message=f"Published message to topic {request.topic}"
        )

        logger.info(f"Published message to topic: {request.topic}")

        return response

    except Exception as e:
        logger.error(f"Error publishing to topic: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error publishing to topic: {str(e)}")


@router.get("/components", response_model=List[Dict[str, Any]])
async def list_components():
    """
    List all integrated components
    """
    try:
        components = integration_manager.component_manager.get_all_components()

        components_list = []
        for comp in components:
            components_list.append({
                "id": comp.id,
                "name": comp.name,
                "type": comp.type.value,
                "status": comp.status.value,
                "dependencies": comp.dependencies,
                "last_update": comp.last_update.isoformat()
            })

        logger.info(f"Listed {len(components_list)} integrated components")

        return components_list

    except Exception as e:
        logger.error(f"Error listing components: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing components: {str(e)}")


@router.get("/bottlenecks", response_model=List[str])
async def identify_bottlenecks():
    """
    Identify system bottlenecks
    """
    try:
        metrics = integration_manager.get_system_metrics()
        bottlenecks = metrics.bottleneck_components

        logger.info(f"Identified {len(bottlenecks)} bottlenecks: {bottlenecks}")

        return bottlenecks

    except Exception as e:
        logger.error(f"Error identifying bottlenecks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error identifying bottlenecks: {str(e)}")