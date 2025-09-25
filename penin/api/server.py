"""
PENIN API Server - REST API for system interaction
FastAPI-based server with authentication, monitoring, and neural core integration
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Import internal modules
try:
    from config.config_manager import get_config_manager, get_config
    from penin.logging.logger import get_logger, log_function_calls, MetricType
    from opt.et_ultimate.agents.brain.neural_core import create_neural_core, ProcessingMode, LearningStrategy
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback for development
    get_config_manager = lambda: None
    get_config = lambda key, default=None: default
    get_logger = lambda: None

# Initialize components
logger = get_logger("penin_api") if get_logger("penin_api") else None
config_manager = get_config_manager()
neural_core = None

# Security
security = HTTPBearer()

# Pydantic models
class ProcessRequest(BaseModel):
    input_data: Union[str, Dict[str, Any]]
    mode: Optional[str] = "hybrid"
    correlation_id: Optional[str] = None

class ProcessResponse(BaseModel):
    success: bool
    result: Dict[str, Any]
    correlation_id: str
    timestamp: str
    processing_time: float

class LearnRequest(BaseModel):
    data: Union[str, Dict[str, Any]]
    feedback: Optional[Dict[str, Any]] = None
    strategy: Optional[str] = "supervised"

class LearnResponse(BaseModel):
    success: bool
    learning_result: Dict[str, Any]
    timestamp: str

class EvolveRequest(BaseModel):
    force: bool = False
    target_modules: Optional[List[str]] = None

class EvolveResponse(BaseModel):
    success: bool
    evolution_result: Dict[str, Any]
    timestamp: str

class SystemStatus(BaseModel):
    status: str
    version: str
    uptime: float
    neural_core_status: Dict[str, Any]
    system_health: float
    active_connections: int
    last_evolution: Optional[str]

class ConfigUpdate(BaseModel):
    key: str
    value: Any

class MetricRequest(BaseModel):
    name: str
    value: float
    metric_type: str = "gauge"
    labels: Optional[Dict[str, str]] = None

# FastAPI app initialization
app = FastAPI(
    title="PENIN Evolution System API",
    description="REST API for interacting with the PENIN neural evolution system",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Global state
app_state = {
    "start_time": time.time(),
    "active_connections": 0,
    "request_count": 0
}

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_config("api.cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=get_config("api.allowed_hosts", ["*"])
)

# Authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token and return user info"""
    if not get_config("api.authentication.enabled", False):
        return {"user_id": "anonymous", "permissions": ["read", "write"]}
    
    # TODO: Implement proper JWT validation
    token = credentials.credentials
    if not token or token == "invalid":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {"user_id": "authenticated_user", "permissions": ["read", "write"]}

# Request middleware
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Request logging and metrics middleware"""
    start_time = time.time()
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    
    # Add correlation ID to request state
    request.state.correlation_id = correlation_id
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Add headers
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log request
    if logger:
        with logger.correlation(correlation_id):
            logger.info(
                f"API Request: {request.method} {request.url.path}",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                process_time=process_time,
                user_agent=request.headers.get("user-agent", "unknown")
            )
            
            # Record metrics
            logger.metric("requests_total", 1, MetricType.COUNTER, {
                "method": request.method,
                "endpoint": request.url.path,
                "status": str(response.status_code)
            })
            
            logger.metric("request_duration", process_time, MetricType.HISTOGRAM, {
                "method": request.method,
                "endpoint": request.url.path
            })
    
    # Update global state
    app_state["request_count"] += 1
    
    return response

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    global neural_core
    
    if logger:
        logger.info("Starting PENIN API Server")
    
    # Initialize neural core
    try:
        neural_core = create_neural_core()
        if logger:
            logger.info("Neural core initialized successfully")
    except Exception as e:
        if logger:
            logger.error("Failed to initialize neural core", exception=e)
        neural_core = None
    
    # Start background tasks
    asyncio.create_task(health_monitor())
    
    if logger:
        logger.info("PENIN API Server started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if logger:
        logger.info("Shutting down PENIN API Server")

# Health monitoring background task
async def health_monitor():
    """Monitor system health in background"""
    while True:
        try:
            if neural_core and logger:
                status = neural_core.get_status()
                health_score = status.get("system_health", 0.0)
                
                logger.metric("system_health", health_score, MetricType.GAUGE, {
                    "component": "neural_core"
                })
                
                # Check for alerts
                if health_score < 0.5:
                    logger.warning(f"Low system health detected: {health_score}")
                
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            if logger:
                logger.error("Health monitor error", exception=e)
            await asyncio.sleep(60)  # Wait longer on error

# API Routes

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PENIN Evolution System API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.time() - app_state["start_time"],
        "neural_core_available": neural_core is not None
    }

@app.get("/status", response_model=SystemStatus)
async def get_system_status(user: Dict = Depends(get_current_user)):
    """Get comprehensive system status"""
    if not neural_core:
        raise HTTPException(status_code=503, detail="Neural core not available")
    
    neural_status = neural_core.get_status()
    
    return SystemStatus(
        status="operational",
        version="2.0.0",
        uptime=time.time() - app_state["start_time"],
        neural_core_status=neural_status,
        system_health=neural_status.get("system_health", 0.0),
        active_connections=app_state["active_connections"],
        last_evolution=neural_status.get("state", {}).get("last_evolution")
    )

@app.post("/neural/process", response_model=ProcessResponse)
async def process_input(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    user: Dict = Depends(get_current_user)
):
    """Process input through neural core"""
    if not neural_core:
        raise HTTPException(status_code=503, detail="Neural core not available")
    
    start_time = time.time()
    correlation_id = request.correlation_id or str(uuid.uuid4())
    
    try:
        # Parse processing mode
        mode = ProcessingMode.HYBRID
        if request.mode:
            try:
                mode = ProcessingMode(request.mode.lower())
            except ValueError:
                if logger:
                    logger.warning(f"Invalid processing mode: {request.mode}, using hybrid")
        
        # Process input
        if logger:
            with logger.correlation(correlation_id):
                logger.info("Processing neural input", mode=mode.value, input_type=type(request.input_data).__name__)
        
        result = neural_core.process(request.input_data, mode)
        
        processing_time = time.time() - start_time
        
        # Record metrics
        if logger:
            logger.metric("neural_operations", 1, MetricType.COUNTER, {
                "operation": "process",
                "mode": mode.value
            })
        
        return ProcessResponse(
            success=True,
            result=result,
            correlation_id=correlation_id,
            timestamp=datetime.utcnow().isoformat(),
            processing_time=processing_time
        )
        
    except Exception as e:
        if logger:
            logger.error("Neural processing failed", exception=e, correlation_id=correlation_id)
        
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/neural/learn", response_model=LearnResponse)
async def learn_from_data(
    request: LearnRequest,
    user: Dict = Depends(get_current_user)
):
    """Train neural core with new data"""
    if not neural_core:
        raise HTTPException(status_code=503, detail="Neural core not available")
    
    try:
        # Parse learning strategy
        strategy = LearningStrategy.SUPERVISED
        if request.strategy:
            try:
                strategy = LearningStrategy(request.strategy.lower())
            except ValueError:
                if logger:
                    logger.warning(f"Invalid learning strategy: {request.strategy}, using supervised")
        
        # Learn from data
        if logger:
            logger.info("Learning from data", strategy=strategy.value, data_type=type(request.data).__name__)
        
        learning_result = neural_core.learn(request.data, request.feedback, strategy)
        
        # Record metrics
        if logger:
            logger.metric("neural_operations", 1, MetricType.COUNTER, {
                "operation": "learn",
                "strategy": strategy.value
            })
        
        return LearnResponse(
            success=True,
            learning_result=learning_result,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        if logger:
            logger.error("Neural learning failed", exception=e)
        
        raise HTTPException(status_code=500, detail=f"Learning failed: {str(e)}")

@app.post("/neural/evolve", response_model=EvolveResponse)
async def trigger_evolution(
    request: EvolveRequest,
    user: Dict = Depends(get_current_user)
):
    """Trigger neural core evolution"""
    if not neural_core:
        raise HTTPException(status_code=503, detail="Neural core not available")
    
    # Check permissions
    if "admin" not in user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Evolution requires admin permissions")
    
    try:
        if logger:
            logger.info("Triggering evolution", force=request.force, target_modules=request.target_modules)
        
        evolution_result = neural_core.evolve(force=request.force)
        
        # Record metrics
        if logger:
            logger.metric("evolution_count", 1, MetricType.COUNTER, {
                "component": "neural_core"
            })
        
        return EvolveResponse(
            success=True,
            evolution_result=evolution_result,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        if logger:
            logger.error("Evolution failed", exception=e)
        
        raise HTTPException(status_code=500, detail=f"Evolution failed: {str(e)}")

@app.get("/neural/modules")
async def get_neural_modules(user: Dict = Depends(get_current_user)):
    """Get information about neural modules"""
    if not neural_core:
        raise HTTPException(status_code=503, detail="Neural core not available")
    
    status = neural_core.get_status()
    return {
        "modules": status.get("modules", {}),
        "active_modules": status.get("state", {}).get("active_modules", [])
    }

@app.get("/config")
async def get_configuration(user: Dict = Depends(get_current_user)):
    """Get system configuration"""
    if not config_manager:
        raise HTTPException(status_code=503, detail="Configuration manager not available")
    
    # Return safe configuration (no secrets)
    config = config_manager.config_data.copy()
    
    # Remove sensitive information
    if "database" in config:
        config["database"] = {k: "***" if "password" in k.lower() or "key" in k.lower() 
                             else v for k, v in config["database"].items()}
    
    return {"configuration": config}

@app.post("/config/update")
async def update_configuration(
    update: ConfigUpdate,
    user: Dict = Depends(get_current_user)
):
    """Update system configuration"""
    if not config_manager:
        raise HTTPException(status_code=503, detail="Configuration manager not available")
    
    # Check permissions
    if "admin" not in user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Configuration update requires admin permissions")
    
    try:
        config_manager.set(update.key, update.value)
        
        if logger:
            logger.info("Configuration updated", key=update.key, value=update.value)
        
        return {"success": True, "message": f"Configuration {update.key} updated"}
        
    except Exception as e:
        if logger:
            logger.error("Configuration update failed", exception=e)
        
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

@app.post("/metrics/record")
async def record_metric(
    metric: MetricRequest,
    user: Dict = Depends(get_current_user)
):
    """Record a custom metric"""
    if not logger:
        raise HTTPException(status_code=503, detail="Metrics system not available")
    
    try:
        metric_type = MetricType(metric.metric_type.lower())
        logger.metric(metric.name, metric.value, metric_type, metric.labels)
        
        return {"success": True, "message": f"Metric {metric.name} recorded"}
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid metric type: {metric.metric_type}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metric recording failed: {str(e)}")

@app.get("/metrics")
async def get_metrics(user: Dict = Depends(get_current_user)):
    """Get metrics summary"""
    if not logger:
        raise HTTPException(status_code=503, detail="Metrics system not available")
    
    return {
        "metrics_summary": logger.get_metrics_summary(),
        "system_stats": {
            "uptime": time.time() - app_state["start_time"],
            "request_count": app_state["request_count"],
            "active_connections": app_state["active_connections"]
        }
    }

@app.websocket("/ws/neural")
async def neural_websocket(websocket):
    """WebSocket endpoint for real-time neural interaction"""
    await websocket.accept()
    app_state["active_connections"] += 1
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if neural_core:
                # Process through neural core
                result = neural_core.process(data.get("input", ""))
                
                # Send response
                await websocket.send_json({
                    "type": "neural_response",
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "Neural core not available"
                })
                
    except Exception as e:
        if logger:
            logger.error("WebSocket error", exception=e)
    finally:
        app_state["active_connections"] -= 1

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))
    
    if logger:
        logger.warning(
            f"HTTP Exception: {exc.status_code} - {exc.detail}",
            status_code=exc.status_code,
            detail=exc.detail,
            correlation_id=correlation_id
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))
    
    if logger:
        logger.error("Unhandled exception", exception=exc, correlation_id=correlation_id)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": "Internal server error",
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Server runner function
def run(host: str = None, port: int = None, **kwargs):
    """Run the API server"""
    host = host or get_config("api.host", "0.0.0.0")
    port = port or get_config("api.port", 8000)
    workers = get_config("api.workers", 1)
    
    print(f"Starting PENIN API Server on {host}:{port}")
    
    uvicorn.run(
        "penin.api.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=get_config("development.hot_reload", False),
        log_level=get_config("logging.level", "info").lower(),
        **kwargs
    )

if __name__ == "__main__":
    run()