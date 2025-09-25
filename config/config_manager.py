"""
Configuration Manager - Sistema de gerenciamento de configurações
Suporte para YAML, JSON e variáveis de ambiente
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class ConfigValidation:
    """Validação de configuração"""
    required_keys: list
    type_checks: dict
    range_checks: dict

class ConfigManager:
    """
    Gerenciador centralizado de configurações com suporte a:
    - Arquivos YAML e JSON
    - Variáveis de ambiente
    - Validação de esquemas
    - Hot-reload de configurações
    - Interpolação de variáveis
    """
    
    def __init__(self, config_path: str = "/workspace/config/system_config.yaml"):
        self.config_path = Path(config_path)
        self.config_data = {}
        self.watchers = []
        self.validation_rules = {}
        
        # Load configuration
        self.load_config()
        
        logger.info(f"Configuration loaded from {self.config_path}")
    
    def load_config(self) -> None:
        """Carrega configuração do arquivo"""
        try:
            if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                with open(self.config_path, 'r') as f:
                    self.config_data = yaml.safe_load(f)
            elif self.config_path.suffix.lower() == '.json':
                with open(self.config_path, 'r') as f:
                    self.config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")
            
            # Interpola variáveis de ambiente
            self._interpolate_env_vars()
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtém valor de configuração usando notação de ponto
        Exemplo: get('neural_core.learning_rate')
        """
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Define valor de configuração usando notação de ponto
        """
        keys = key.split('.')
        config = self.config_data
        
        # Navega até o penúltimo nível
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Define o valor final
        config[keys[-1]] = value
        
        # Notifica watchers
        self._notify_watchers(key, value)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Atualiza múltiplas configurações"""
        for key, value in updates.items():
            self.set(key, value)
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Salva configuração atual no arquivo"""
        save_path = Path(path) if path else self.config_path
        
        try:
            if save_path.suffix.lower() in ['.yaml', '.yml']:
                with open(save_path, 'w') as f:
                    yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            elif save_path.suffix.lower() == '.json':
                with open(save_path, 'w') as f:
                    json.dump(self.config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise
    
    def reload(self) -> None:
        """Recarrega configuração do arquivo"""
        old_config = self.config_data.copy()
        self.load_config()
        
        # Detecta mudanças e notifica watchers
        self._detect_changes(old_config, self.config_data)
        
        logger.info("Configuration reloaded")
    
    def watch(self, key: str, callback: callable) -> None:
        """
        Registra callback para mudanças em configuração específica
        """
        self.watchers.append({
            'key': key,
            'callback': callback
        })
    
    def validate_config(self, rules: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Valida configuração contra regras definidas
        """
        validation_rules = rules or self.validation_rules
        errors = []
        warnings = []
        
        for section, rules in validation_rules.items():
            section_config = self.get(section, {})
            
            # Verifica chaves obrigatórias
            for required_key in rules.get('required', []):
                if required_key not in section_config:
                    errors.append(f"Missing required key: {section}.{required_key}")
            
            # Verifica tipos
            for key, expected_type in rules.get('types', {}).items():
                if key in section_config:
                    value = section_config[key]
                    if not isinstance(value, expected_type):
                        errors.append(f"Type error: {section}.{key} should be {expected_type.__name__}")
            
            # Verifica ranges
            for key, (min_val, max_val) in rules.get('ranges', {}).items():
                if key in section_config:
                    value = section_config[key]
                    if isinstance(value, (int, float)):
                        if value < min_val or value > max_val:
                            warnings.append(f"Range warning: {section}.{key} should be between {min_val} and {max_val}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Retorna configurações específicas do ambiente"""
        env = self.get('system.environment', 'development')
        return self.get(env, {})
    
    def get_neural_config(self) -> Dict[str, Any]:
        """Retorna configuração do neural core"""
        return self.get('neural_core', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Retorna configuração da API"""
        return self.get('api', {})
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Retorna configuração de ML"""
        return self.get('ml', {})
    
    def get_security_config(self) -> Dict[str, Any]:
        """Retorna configuração de segurança"""
        return self.get('security', {})
    
    def export_config(self, format: str = 'yaml') -> str:
        """Exporta configuração como string"""
        if format.lower() == 'yaml':
            return yaml.dump(self.config_data, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            return json.dumps(self.config_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _interpolate_env_vars(self) -> None:
        """Interpola variáveis de ambiente no formato ${VAR_NAME}"""
        def interpolate_value(value):
            if isinstance(value, str):
                # Procura por padrões ${VAR_NAME}
                pattern = r'\$\{([^}]+)\}'
                matches = re.findall(pattern, value)
                
                for match in matches:
                    env_value = os.getenv(match, f"${{{match}}}")  # Mantém original se não encontrar
                    value = value.replace(f"${{{match}}}", env_value)
                
                return value
            elif isinstance(value, dict):
                return {k: interpolate_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [interpolate_value(item) for item in value]
            else:
                return value
        
        self.config_data = interpolate_value(self.config_data)
    
    def _notify_watchers(self, key: str, value: Any) -> None:
        """Notifica watchers sobre mudanças"""
        for watcher in self.watchers:
            if watcher['key'] == key or key.startswith(watcher['key'] + '.'):
                try:
                    watcher['callback'](key, value)
                except Exception as e:
                    logger.error(f"Error in config watcher callback: {e}")
    
    def _detect_changes(self, old_config: Dict, new_config: Dict, prefix: str = '') -> None:
        """Detecta mudanças entre configurações"""
        for key, new_value in new_config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in old_config:
                self._notify_watchers(full_key, new_value)
            elif old_config[key] != new_value:
                if isinstance(new_value, dict) and isinstance(old_config[key], dict):
                    self._detect_changes(old_config[key], new_value, full_key)
                else:
                    self._notify_watchers(full_key, new_value)

# Singleton instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Retorna instância singleton do gerenciador de configuração"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config(key: str, default: Any = None) -> Any:
    """Função de conveniência para obter configuração"""
    return get_config_manager().get(key, default)

def set_config(key: str, value: Any) -> None:
    """Função de conveniência para definir configuração"""
    get_config_manager().set(key, value)

# Validation rules for the system
VALIDATION_RULES = {
    'neural_core': {
        'required': ['version', 'processing_mode', 'learning_rate'],
        'types': {
            'learning_rate': float,
            'confidence_threshold': float,
            'max_evolutions': int
        },
        'ranges': {
            'learning_rate': (0.0001, 1.0),
            'confidence_threshold': (0.0, 1.0)
        }
    },
    'api': {
        'required': ['host', 'port'],
        'types': {
            'port': int,
            'workers': int,
            'timeout': int
        },
        'ranges': {
            'port': (1, 65535),
            'workers': (1, 32),
            'timeout': (1, 300)
        }
    },
    'system': {
        'required': ['name', 'version'],
        'types': {
            'debug': bool,
            'version': str
        }
    }
}

# Example usage and testing
if __name__ == "__main__":
    # Demonstração do sistema de configuração
    config = get_config_manager()
    
    # Testa leitura de configurações
    print("Neural Core Config:", config.get_neural_config())
    print("API Config:", config.get_api_config())
    
    # Testa validação
    config.validation_rules = VALIDATION_RULES
    validation_result = config.validate_config()
    print("Validation Result:", validation_result)
    
    # Testa watcher
    def on_config_change(key, value):
        print(f"Config changed: {key} = {value}")
    
    config.watch('neural_core.learning_rate', on_config_change)
    
    # Testa mudança
    config.set('neural_core.learning_rate', 0.02)
    
    # Testa exportação
    yaml_export = config.export_config('yaml')
    print("YAML Export (first 500 chars):", yaml_export[:500])