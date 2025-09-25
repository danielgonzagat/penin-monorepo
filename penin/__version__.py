"""
PENIN Evolution System Version Information
"""

__version__ = "2.0.0"
__version_info__ = (2, 0, 0)

# Build information
__build__ = "2025.09.25"
__author__ = "PENIN Evolution Team"
__email__ = "evolution@penin.ai"
__license__ = "MIT"
__copyright__ = "2025 PENIN Evolution System"

# System information
SYSTEM_NAME = "PENIN Evolution System"
SYSTEM_DESCRIPTION = "Sistema de IA com arquitetura modular e capacidades de auto-evolução"
NEURAL_CORE_VERSION = "2.0.0"
API_VERSION = "2.0.0"
EVOLUTION_ENGINE_VERSION = "1.0.0"

# Evolution tracking
EVOLUTION_COUNT = 12
LAST_EVOLUTION = "2025-09-25T00:00:00Z"
EVOLUTION_HISTORY = [
    "Enhanced Neural Core with modular architecture",
    "Implemented comprehensive configuration management",
    "Added advanced logging and monitoring system",
    "Created REST API with FastAPI and authentication",
    "Built PENIN Omega evolution engine",
    "Developed neural language models",
    "Implemented comprehensive test suite",
    "Added Docker containerization",
    "Created complete documentation",
    "Optimized performance and security",
    "Added monitoring and observability",
    "Completed production deployment setup"
]

# Feature flags
FEATURES = {
    "neural_core": True,
    "auto_evolution": True,
    "api_server": True,
    "ml_models": True,
    "monitoring": True,
    "testing": True,
    "docker": True,
    "kubernetes": True,
    "authentication": True,
    "rate_limiting": True,
    "cors": True,
    "websockets": True,
    "background_tasks": True,
    "scheduled_jobs": True,
    "metrics": True,
    "logging": True,
    "configuration": True,
    "documentation": True
}

def get_version_info():
    """Get comprehensive version information"""
    return {
        "version": __version__,
        "build": __build__,
        "system_name": SYSTEM_NAME,
        "description": SYSTEM_DESCRIPTION,
        "evolution_count": EVOLUTION_COUNT,
        "last_evolution": LAST_EVOLUTION,
        "features": FEATURES,
        "components": {
            "neural_core": NEURAL_CORE_VERSION,
            "api": API_VERSION,
            "evolution_engine": EVOLUTION_ENGINE_VERSION
        }
    }

def print_version_info():
    """Print version information"""
    info = get_version_info()
    print(f"{info['system_name']} v{info['version']}")
    print(f"Build: {info['build']}")
    print(f"Evolution Count: {info['evolution_count']}")
    print(f"Last Evolution: {info['last_evolution']}")
    print(f"Features: {len([f for f, enabled in info['features'].items() if enabled])}/{len(info['features'])} enabled")

if __name__ == "__main__":
    print_version_info()