"""
PENIN Omega - Evolution Engine
Sistema avançado de auto-evolução e modificação de código
"""

import os
import ast
import sys
import json
import time
import hashlib
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import asyncio
import shutil
import tempfile

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.config_manager import get_config
    from penin.logging.logger import get_logger
except ImportError:
    get_config = lambda key, default=None: default
    get_logger = lambda: None

logger = get_logger("penin_omega") if get_logger("penin_omega") else None

class EvolutionType(Enum):
    CODE_OPTIMIZATION = "code_optimization"
    ARCHITECTURE_REFACTOR = "architecture_refactor"
    DEPENDENCY_UPDATE = "dependency_update"
    PERFORMANCE_ENHANCEMENT = "performance_enhancement"
    SECURITY_PATCH = "security_patch"
    FEATURE_ADDITION = "feature_addition"
    BUG_FIX = "bug_fix"
    DOCUMENTATION_UPDATE = "documentation_update"

class SafetyLevel(Enum):
    SAFE = "safe"           # Minimal risk changes
    MODERATE = "moderate"   # Some risk, requires testing
    HIGH = "high"          # High risk, requires approval
    CRITICAL = "critical"  # Critical changes, requires manual review

@dataclass
class EvolutionPlan:
    """Plano de evolução do sistema"""
    id: str
    type: EvolutionType
    description: str
    target_files: List[str]
    safety_level: SafetyLevel
    estimated_impact: float  # 0.0 to 1.0
    prerequisites: List[str]
    rollback_plan: Dict[str, Any]
    created_at: str
    status: str = "planned"

@dataclass
class EvolutionResult:
    """Resultado de uma evolução"""
    plan_id: str
    success: bool
    changes_made: List[str]
    performance_impact: Dict[str, float]
    errors: List[str]
    warnings: List[str]
    execution_time: float
    rollback_available: bool
    timestamp: str

class CodeAnalyzer:
    """Analisador de código para identificar oportunidades de melhoria"""
    
    def __init__(self):
        self.analysis_cache = {}
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analisa um arquivo Python para identificar melhorias"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            analysis = {
                'complexity': self._calculate_complexity(tree),
                'code_smells': self._detect_code_smells(tree, content),
                'optimization_opportunities': self._find_optimizations(tree, content),
                'security_issues': self._check_security(tree, content),
                'performance_issues': self._analyze_performance(tree, content),
                'maintainability_score': 0.0,
                'test_coverage': self._estimate_test_coverage(file_path),
                'dependencies': self._extract_dependencies(tree)
            }
            
            # Calculate overall maintainability score
            analysis['maintainability_score'] = self._calculate_maintainability(analysis)
            
            return analysis
            
        except Exception as e:
            if logger:
                logger.error(f"Failed to analyze file {file_path}", exception=e)
            return {'error': str(e)}
    
    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """Calcula métricas de complexidade"""
        complexity = {
            'cyclomatic': 0,
            'cognitive': 0,
            'lines_of_code': 0,
            'functions': 0,
            'classes': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity['cyclomatic'] += 1
            elif isinstance(node, ast.FunctionDef):
                complexity['functions'] += 1
            elif isinstance(node, ast.ClassDef):
                complexity['classes'] += 1
        
        return complexity
    
    def _detect_code_smells(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detecta code smells"""
        smells = []
        
        # Long methods
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if func_lines > 50:
                    smells.append({
                        'type': 'long_method',
                        'description': f"Method '{node.name}' is too long ({func_lines} lines)",
                        'line': node.lineno,
                        'severity': 'moderate'
                    })
        
        # Large classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 20:
                    smells.append({
                        'type': 'large_class',
                        'description': f"Class '{node.name}' has too many methods ({len(methods)})",
                        'line': node.lineno,
                        'severity': 'high'
                    })
        
        # Duplicate code (simplified detection)
        lines = content.split('\n')
        line_hashes = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) > 10:  # Ignore short lines
                line_hash = hashlib.md5(stripped.encode()).hexdigest()
                if line_hash in line_hashes:
                    smells.append({
                        'type': 'duplicate_code',
                        'description': f"Potential duplicate code at lines {line_hashes[line_hash]} and {i+1}",
                        'line': i+1,
                        'severity': 'moderate'
                    })
                else:
                    line_hashes[line_hash] = i+1
        
        return smells
    
    def _find_optimizations(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Encontra oportunidades de otimização"""
        optimizations = []
        
        # String concatenation in loops
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                        if isinstance(child.target, ast.Name) and isinstance(child.value, ast.Str):
                            optimizations.append({
                                'type': 'string_concatenation',
                                'description': 'Use join() instead of += for string concatenation in loops',
                                'line': child.lineno,
                                'priority': 'high'
                            })
        
        # List comprehensions vs loops
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Simplified detection of appendable loops
                for stmt in node.body:
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        if hasattr(stmt.value.func, 'attr') and stmt.value.func.attr == 'append':
                            optimizations.append({
                                'type': 'list_comprehension',
                                'description': 'Consider using list comprehension instead of append in loop',
                                'line': node.lineno,
                                'priority': 'moderate'
                            })
        
        return optimizations
    
    def _check_security(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Verifica problemas de segurança"""
        security_issues = []
        
        # SQL injection risks
        if 'execute(' in content and any(op in content for op in ['%', 'format', 'f"']):
            security_issues.append({
                'type': 'sql_injection',
                'description': 'Potential SQL injection vulnerability - use parameterized queries',
                'severity': 'critical'
            })
        
        # Hardcoded secrets
        for node in ast.walk(tree):
            if isinstance(node, ast.Str):
                if any(keyword in node.s.lower() for keyword in ['password', 'secret', 'key', 'token']):
                    if len(node.s) > 10:  # Likely a real secret, not just the word
                        security_issues.append({
                            'type': 'hardcoded_secret',
                            'description': 'Potential hardcoded secret found',
                            'line': node.lineno,
                            'severity': 'high'
                        })
        
        return security_issues
    
    def _analyze_performance(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Analisa problemas de performance"""
        performance_issues = []
        
        # Nested loops
        nested_loops = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if child != node and isinstance(child, (ast.For, ast.While)):
                        nested_loops.append(child.lineno)
        
        if nested_loops:
            performance_issues.append({
                'type': 'nested_loops',
                'description': f'Nested loops found at lines {nested_loops} - consider optimization',
                'severity': 'moderate'
            })
        
        # Global variable access in loops
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                        # Simplified check for global access
                        performance_issues.append({
                            'type': 'global_access_in_loop',
                            'description': 'Consider caching global variable access in loops',
                            'line': node.lineno,
                            'severity': 'low'
                        })
                        break  # Only report once per loop
        
        return performance_issues
    
    def _estimate_test_coverage(self, file_path: str) -> float:
        """Estima cobertura de testes (simplificado)"""
        test_file = file_path.replace('.py', '_test.py')
        if not os.path.exists(test_file):
            test_file = file_path.replace('.py', '') + '/test_' + os.path.basename(file_path)
        
        if os.path.exists(test_file):
            # Simplified estimation based on file sizes
            try:
                main_size = os.path.getsize(file_path)
                test_size = os.path.getsize(test_file)
                return min(1.0, test_size / main_size)
            except:
                return 0.0
        
        return 0.0
    
    def _extract_dependencies(self, tree: ast.AST) -> List[str]:
        """Extrai dependências do arquivo"""
        dependencies = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.append(node.module)
        
        return dependencies
    
    def _calculate_maintainability(self, analysis: Dict[str, Any]) -> float:
        """Calcula score de maintainability"""
        score = 1.0
        
        # Penalize complexity
        complexity = analysis.get('complexity', {})
        if complexity.get('cyclomatic', 0) > 10:
            score -= 0.2
        
        # Penalize code smells
        smells = analysis.get('code_smells', [])
        score -= len(smells) * 0.05
        
        # Reward test coverage
        coverage = analysis.get('test_coverage', 0)
        score += coverage * 0.2
        
        return max(0.0, min(1.0, score))

class EvolutionEngine:
    """Motor de evolução principal do sistema PENIN Omega"""
    
    def __init__(self, workspace_path: str = "/workspace"):
        self.workspace_path = Path(workspace_path)
        self.analyzer = CodeAnalyzer()
        self.evolution_history = []
        self.active_plans = []
        self.safety_checks_enabled = get_config("penin_omega.safety_checks", True)
        self.backup_enabled = get_config("penin_omega.backup_before_evolution", True)
        self.auto_evolution = get_config("penin_omega.auto_modification", False)
        
        # Evolution statistics
        self.stats = {
            'total_evolutions': 0,
            'successful_evolutions': 0,
            'rollbacks': 0,
            'files_modified': 0,
            'performance_improvements': 0.0
        }
        
        if logger:
            logger.info("PENIN Omega Evolution Engine initialized")
    
    def scan_system(self) -> Dict[str, Any]:
        """Escaneia o sistema em busca de oportunidades de evolução"""
        scan_results = {
            'timestamp': datetime.now().isoformat(),
            'files_analyzed': 0,
            'issues_found': 0,
            'optimization_opportunities': 0,
            'security_issues': 0,
            'files': {}
        }
        
        # Scan Python files
        for py_file in self.workspace_path.rglob("*.py"):
            if self._should_analyze_file(py_file):
                analysis = self.analyzer.analyze_file(str(py_file))
                
                if 'error' not in analysis:
                    relative_path = str(py_file.relative_to(self.workspace_path))
                    scan_results['files'][relative_path] = analysis
                    scan_results['files_analyzed'] += 1
                    
                    # Count issues
                    scan_results['issues_found'] += len(analysis.get('code_smells', []))
                    scan_results['optimization_opportunities'] += len(analysis.get('optimization_opportunities', []))
                    scan_results['security_issues'] += len(analysis.get('security_issues', []))
        
        if logger:
            logger.info(f"System scan completed: {scan_results['files_analyzed']} files analyzed")
        
        return scan_results
    
    def generate_evolution_plan(self, scan_results: Dict[str, Any]) -> List[EvolutionPlan]:
        """Gera plano de evolução baseado nos resultados do scan"""
        plans = []
        
        for file_path, analysis in scan_results.get('files', {}).items():
            # Security fixes (highest priority)
            for issue in analysis.get('security_issues', []):
                plan = EvolutionPlan(
                    id=self._generate_plan_id(),
                    type=EvolutionType.SECURITY_PATCH,
                    description=f"Fix security issue: {issue['description']}",
                    target_files=[file_path],
                    safety_level=SafetyLevel.HIGH,
                    estimated_impact=0.8,
                    prerequisites=[],
                    rollback_plan={'backup_file': f"{file_path}.backup"},
                    created_at=datetime.now().isoformat()
                )
                plans.append(plan)
            
            # Performance optimizations
            for opt in analysis.get('optimization_opportunities', []):
                plan = EvolutionPlan(
                    id=self._generate_plan_id(),
                    type=EvolutionType.PERFORMANCE_ENHANCEMENT,
                    description=f"Performance optimization: {opt['description']}",
                    target_files=[file_path],
                    safety_level=SafetyLevel.MODERATE,
                    estimated_impact=0.3,
                    prerequisites=[],
                    rollback_plan={'backup_file': f"{file_path}.backup"},
                    created_at=datetime.now().isoformat()
                )
                plans.append(plan)
            
            # Code smell fixes
            high_priority_smells = [s for s in analysis.get('code_smells', []) 
                                  if s.get('severity') in ['high', 'critical']]
            
            if high_priority_smells:
                plan = EvolutionPlan(
                    id=self._generate_plan_id(),
                    type=EvolutionType.CODE_OPTIMIZATION,
                    description=f"Fix {len(high_priority_smells)} code smells in {file_path}",
                    target_files=[file_path],
                    safety_level=SafetyLevel.SAFE,
                    estimated_impact=0.2,
                    prerequisites=[],
                    rollback_plan={'backup_file': f"{file_path}.backup"},
                    created_at=datetime.now().isoformat()
                )
                plans.append(plan)
        
        # Sort plans by priority (safety level and impact)
        plans.sort(key=lambda p: (p.safety_level.value, -p.estimated_impact))
        
        if logger:
            logger.info(f"Generated {len(plans)} evolution plans")
        
        return plans
    
    def execute_evolution_plan(self, plan: EvolutionPlan, dry_run: bool = False) -> EvolutionResult:
        """Executa um plano de evolução"""
        start_time = time.time()
        
        result = EvolutionResult(
            plan_id=plan.id,
            success=False,
            changes_made=[],
            performance_impact={},
            errors=[],
            warnings=[],
            execution_time=0.0,
            rollback_available=False,
            timestamp=datetime.now().isoformat()
        )
        
        try:
            if logger:
                logger.info(f"Executing evolution plan: {plan.description}")
            
            # Safety checks
            if self.safety_checks_enabled and not self._safety_check(plan):
                result.errors.append("Safety check failed")
                return result
            
            # Create backup if enabled
            backup_paths = {}
            if self.backup_enabled and not dry_run:
                backup_paths = self._create_backup(plan.target_files)
                result.rollback_available = True
            
            # Execute evolution based on type
            if not dry_run:
                if plan.type == EvolutionType.CODE_OPTIMIZATION:
                    self._execute_code_optimization(plan, result)
                elif plan.type == EvolutionType.PERFORMANCE_ENHANCEMENT:
                    self._execute_performance_enhancement(plan, result)
                elif plan.type == EvolutionType.SECURITY_PATCH:
                    self._execute_security_patch(plan, result)
                else:
                    result.warnings.append(f"Evolution type {plan.type} not implemented yet")
            else:
                result.changes_made.append(f"DRY RUN: Would execute {plan.type.value}")
            
            # Run tests if available
            if not dry_run and self._has_tests():
                test_result = self._run_tests()
                if not test_result:
                    result.errors.append("Tests failed after evolution")
                    if backup_paths:
                        self._restore_backup(backup_paths)
                    return result
            
            result.success = len(result.errors) == 0
            
            # Update statistics
            if result.success and not dry_run:
                self.stats['successful_evolutions'] += 1
                self.stats['files_modified'] += len(plan.target_files)
            
        except Exception as e:
            result.errors.append(f"Evolution execution failed: {str(e)}")
            if logger:
                logger.error("Evolution execution failed", exception=e)
        
        finally:
            result.execution_time = time.time() - start_time
            self.stats['total_evolutions'] += 1
        
        return result
    
    def auto_evolve(self) -> List[EvolutionResult]:
        """Executa evolução automática do sistema"""
        if not self.auto_evolution:
            if logger:
                logger.warning("Auto-evolution is disabled")
            return []
        
        results = []
        
        try:
            # Scan system
            scan_results = self.scan_system()
            
            # Generate plans
            plans = self.generate_evolution_plan(scan_results)
            
            # Execute safe plans automatically
            safe_plans = [p for p in plans if p.safety_level == SafetyLevel.SAFE]
            
            for plan in safe_plans[:5]:  # Limit to 5 plans per run
                result = self.execute_evolution_plan(plan)
                results.append(result)
                
                if not result.success:
                    if logger:
                        logger.warning(f"Auto-evolution failed for plan {plan.id}")
                    break  # Stop on first failure
            
            if logger:
                successful = sum(1 for r in results if r.success)
                logger.info(f"Auto-evolution completed: {successful}/{len(results)} successful")
        
        except Exception as e:
            if logger:
                logger.error("Auto-evolution failed", exception=e)
        
        return results
    
    def rollback_evolution(self, plan_id: str) -> bool:
        """Reverte uma evolução"""
        try:
            # Find the evolution in history
            evolution = None
            for evo in self.evolution_history:
                if evo.get('plan_id') == plan_id:
                    evolution = evo
                    break
            
            if not evolution:
                if logger:
                    logger.error(f"Evolution {plan_id} not found in history")
                return False
            
            # Restore from backup
            backup_paths = evolution.get('backup_paths', {})
            if backup_paths:
                self._restore_backup(backup_paths)
                
                if logger:
                    logger.info(f"Evolution {plan_id} rolled back successfully")
                
                self.stats['rollbacks'] += 1
                return True
            else:
                if logger:
                    logger.error(f"No backup available for evolution {plan_id}")
                return False
        
        except Exception as e:
            if logger:
                logger.error("Rollback failed", exception=e)
            return False
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Retorna status do sistema de evolução"""
        return {
            'stats': self.stats,
            'active_plans': len(self.active_plans),
            'evolution_history': len(self.evolution_history),
            'auto_evolution_enabled': self.auto_evolution,
            'safety_checks_enabled': self.safety_checks_enabled,
            'last_scan': getattr(self, 'last_scan_time', None)
        }
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determina se um arquivo deve ser analisado"""
        # Skip test files, migrations, etc.
        skip_patterns = ['test_', '_test.', 'migration', '__pycache__', '.git']
        
        file_str = str(file_path)
        return not any(pattern in file_str for pattern in skip_patterns)
    
    def _generate_plan_id(self) -> str:
        """Gera ID único para plano de evolução"""
        return f"evo_{int(time.time())}_{len(self.active_plans)}"
    
    def _safety_check(self, plan: EvolutionPlan) -> bool:
        """Executa verificações de segurança"""
        # Check if files exist
        for file_path in plan.target_files:
            full_path = self.workspace_path / file_path
            if not full_path.exists():
                return False
        
        # Check safety level
        if plan.safety_level == SafetyLevel.CRITICAL:
            return False  # Require manual approval
        
        return True
    
    def _create_backup(self, file_paths: List[str]) -> Dict[str, str]:
        """Cria backup dos arquivos"""
        backup_paths = {}
        
        for file_path in file_paths:
            full_path = self.workspace_path / file_path
            if full_path.exists():
                backup_path = f"{full_path}.backup.{int(time.time())}"
                shutil.copy2(full_path, backup_path)
                backup_paths[file_path] = backup_path
        
        return backup_paths
    
    def _restore_backup(self, backup_paths: Dict[str, str]) -> None:
        """Restaura arquivos do backup"""
        for original_path, backup_path in backup_paths.items():
            if os.path.exists(backup_path):
                full_path = self.workspace_path / original_path
                shutil.copy2(backup_path, full_path)
                os.remove(backup_path)  # Clean up backup
    
    def _execute_code_optimization(self, plan: EvolutionPlan, result: EvolutionResult) -> None:
        """Executa otimização de código"""
        # Placeholder for code optimization logic
        result.changes_made.append("Code optimization applied")
        result.warnings.append("Code optimization is not fully implemented yet")
    
    def _execute_performance_enhancement(self, plan: EvolutionPlan, result: EvolutionResult) -> None:
        """Executa melhorias de performance"""
        # Placeholder for performance enhancement logic
        result.changes_made.append("Performance enhancement applied")
        result.warnings.append("Performance enhancement is not fully implemented yet")
    
    def _execute_security_patch(self, plan: EvolutionPlan, result: EvolutionResult) -> None:
        """Executa correções de segurança"""
        # Placeholder for security patch logic
        result.changes_made.append("Security patch applied")
        result.warnings.append("Security patch is not fully implemented yet")
    
    def _has_tests(self) -> bool:
        """Verifica se há testes disponíveis"""
        test_files = list(self.workspace_path.rglob("test_*.py"))
        test_files.extend(list(self.workspace_path.rglob("*_test.py")))
        return len(test_files) > 0
    
    def _run_tests(self) -> bool:
        """Executa testes do sistema"""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", str(self.workspace_path)],
                capture_output=True,
                text=True,
                cwd=str(self.workspace_path)
            )
            return result.returncode == 0
        except:
            return False

# Factory function
def create_evolution_engine(workspace_path: str = "/workspace") -> EvolutionEngine:
    """Cria instância do motor de evolução"""
    return EvolutionEngine(workspace_path)

# CLI interface
def main():
    """Interface CLI para o PENIN Omega"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PENIN Omega Evolution Engine")
    parser.add_argument("--scan", action="store_true", help="Scan system for evolution opportunities")
    parser.add_argument("--auto-evolve", action="store_true", help="Run automatic evolution")
    parser.add_argument("--status", action="store_true", help="Show evolution status")
    parser.add_argument("--workspace", default="/workspace", help="Workspace path")
    
    args = parser.parse_args()
    
    engine = create_evolution_engine(args.workspace)
    
    if args.scan:
        print("Scanning system...")
        results = engine.scan_system()
        print(f"Scan completed: {results['files_analyzed']} files analyzed")
        print(f"Issues found: {results['issues_found']}")
        print(f"Optimization opportunities: {results['optimization_opportunities']}")
        print(f"Security issues: {results['security_issues']}")
    
    elif args.auto_evolve:
        print("Starting auto-evolution...")
        results = engine.auto_evolve()
        successful = sum(1 for r in results if r.success)
        print(f"Auto-evolution completed: {successful}/{len(results)} successful")
    
    elif args.status:
        status = engine.get_evolution_status()
        print("Evolution Status:")
        print(json.dumps(status, indent=2))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()