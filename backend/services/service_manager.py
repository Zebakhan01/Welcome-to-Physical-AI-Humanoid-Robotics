"""
Service Manager for Phase 2 RAG Services
Handles initialization and management of all RAG services
"""
from typing import Dict, Any, Optional
from ...config.rag_settings import settings
from ...utils.logger import logger
from ..rag.rag_service import RAGService


class ServiceManager:
    """
    Manages the lifecycle of all RAG services
    """

    def __init__(self):
        """Initialize the service manager"""
        self.services: Dict[str, Any] = {}
        self._initialized = False
        logger.info("Service Manager initialized")

    async def initialize_services(self) -> bool:
        """
        Initialize all RAG services

        Returns:
            True if all services were initialized successfully
        """
        try:
            # Validate required environment variables
            if not settings.COHERE_API_KEY:
                raise ValueError("COHERE_API_KEY environment variable is required")
            if not settings.QDRANT_URL:
                raise ValueError("QDRANT_URL environment variable is required")
            if not settings.QDRANT_API_KEY:
                raise ValueError("QDRANT_API_KEY environment variable is required")
            if not settings.NEON_DATABASE_URL:
                raise ValueError("NEON_DATABASE_URL environment variable is required")

            # Initialize the main RAG service (which initializes all sub-services)
            rag_service = RAGService()
            self.services["rag"] = rag_service

            # Verify all services are working
            health_status = await rag_service.health_check()
            all_healthy = all(health_status.get(service, False) for service in
                            ["content_loader", "embedding_service", "vector_store",
                             "database_service", "retrieval_service", "answer_generation_service"])

            if not all_healthy:
                logger.error("Some services failed to initialize properly")
                return False

            self._initialized = True
            logger.info("✅ All RAG services initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Error initializing services: {str(e)}")
            return False

    def get_service(self, service_name: str) -> Optional[Any]:
        """
        Get a specific service by name

        Args:
            service_name: Name of the service to retrieve

        Returns:
            Service instance or None if not found
        """
        return self.services.get(service_name)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all managed services

        Returns:
            Dictionary with health status of all services
        """
        if not self._initialized:
            return {"status": "not_initialized", "services": {}}

        try:
            rag_service = self.get_service("rag")
            if rag_service:
                return await rag_service.health_check()
            else:
                return {"status": "error", "services": {}, "error": "RAG service not available"}
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            return {"status": "error", "services": {}, "error": str(e)}

    async def shutdown(self):
        """
        Shutdown all managed services
        """
        logger.info("Shutting down all services")
        # Currently, our services don't require explicit shutdown,
        # but this could be extended if needed in the future
        self.services.clear()
        self._initialized = False
        logger.info("All services shut down")


# Global service manager instance
service_manager = ServiceManager()