"""
PENIN Evolution System
Sistema de IA com arquitetura modular e capacidades de auto-evolução
"""

from ..__version__ import (
    __version__,
    __version_info__,
    __author__,
    __email__,
    __license__,
    SYSTEM_NAME,
    SYSTEM_DESCRIPTION,
    get_version_info,
    print_version_info
)

# Core imports
try:
    from ..opt.et_ultimate.agents.brain.neural_core import create_neural_core, NeuralCore
except ImportError:
    create_neural_core = None
    NeuralCore = None

try:
    from ..penin_omega.evolution_engine import create_evolution_engine, EvolutionEngine
except ImportError:
    create_evolution_engine = None
    EvolutionEngine = None

try:
    from ..config.config_manager import get_config_manager, get_config
except ImportError:
    get_config_manager = None
    get_config = None

try:
    from .logging.logger import get_logger, setup_logging
except ImportError:
    get_logger = None
    setup_logging = None

# Export main components
__all__ = [
    # Version info
    "__version__",
    "__version_info__",
    "__author__",
    "__email__",
    "__license__",
    "SYSTEM_NAME",
    "SYSTEM_DESCRIPTION",
    "get_version_info",
    "print_version_info",
    
    # Core components
    "create_neural_core",
    "NeuralCore",
    "create_evolution_engine", 
    "EvolutionEngine",
    "get_config_manager",
    "get_config",
    "get_logger",
    "setup_logging",
]

# System initialization
def initialize_system(config_path=None):
    """Initialize the PENIN system with all components"""
    components = {}
    
    # Initialize configuration
    if get_config_manager:
        try:
            config_manager = get_config_manager()
            components["config"] = config_manager
        except Exception as e:
            print(f"Warning: Failed to initialize config manager: {e}")
    
    # Initialize logging
    if setup_logging:
        try:
            logger = setup_logging({})
            components["logger"] = logger
        except Exception as e:
            print(f"Warning: Failed to initialize logging: {e}")
    
    # Initialize neural core
    if create_neural_core:
        try:
            neural_core = create_neural_core(config_path)
            components["neural_core"] = neural_core
        except Exception as e:
            print(f"Warning: Failed to initialize neural core: {e}")
    
    # Initialize evolution engine
    if create_evolution_engine:
        try:
            evolution_engine = create_evolution_engine()
            components["evolution_engine"] = evolution_engine
        except Exception as e:
            print(f"Warning: Failed to initialize evolution engine: {e}")
    
    return components

def get_system_status():
    """Get overall system status"""
    status = {
        "system_name": SYSTEM_NAME,
        "version": __version__,
        "components": {
            "neural_core": NeuralCore is not None,
            "evolution_engine": EvolutionEngine is not None,
            "config_manager": get_config_manager is not None,
            "logger": get_logger is not None,
        }
    }
    
    return status