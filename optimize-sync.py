#!/usr/bin/env python3
"""
PENIN Intelligent Sync Optimizer

This script optimizes the auto-sync process to prevent redundant commits
and improve overall repository quality through intelligent change detection.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import hashlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PENINSyncOptimizer:
    """
    Intelligent sync optimizer that prevents redundant commits and
    optimizes the synchronization process based on meaningful changes.
    """
    
    def __init__(self, config_path: str = ".penin-sync-config.json"):
        """Initialize the sync optimizer with configuration."""
        self.config_path = Path(config_path)
        self.repo_root = Path.cwd()
        self.config = self._load_config()
        self.last_sync_hash = self._get_last_sync_hash()
        
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "commit_optimization": {"enabled": True, "min_changes_threshold": 3},
            "timestamp_management": {"enabled": True, "update_frequency": "on_meaningful_change_only"},
            "performance_monitoring": {"enabled": True},
            "quality_gates": {"enabled": True},
            "auto_optimization": {"enabled": True}
        }
    
    def _get_last_sync_hash(self) -> Optional[str]:
        """Get hash of last synchronized state."""
        hash_file = self.repo_root / ".last_sync_hash"
        if hash_file.exists():
            try:
                return hash_file.read_text().strip()
            except Exception:
                return None
        return None
    
    def _save_sync_hash(self, hash_value: str) -> None:
        """Save hash of current synchronized state."""
        hash_file = self.repo_root / ".last_sync_hash"
        try:
            hash_file.write_text(hash_value)
        except Exception as e:
            logger.error(f"Error saving sync hash: {e}")
    
    def _calculate_repo_hash(self) -> str:
        """Calculate hash of meaningful repository content."""
        hasher = hashlib.sha256()
        
        # Get all tracked files
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            tracked_files = result.stdout.strip().split('\n') if result.stdout else []
        except Exception as e:
            logger.error(f"Error getting tracked files: {e}")
            return ""
        
        # Include meaningful files in hash calculation
        meaningful_patterns = self.config.get("file_patterns", {}).get("always_sync", ["*.py", "*.md"])
        
        for file_path in sorted(tracked_files):
            if self._is_meaningful_file(file_path, meaningful_patterns):
                try:
                    file_full_path = self.repo_root / file_path
                    if file_full_path.exists() and file_full_path.is_file():
                        content = file_full_path.read_bytes()
                        hasher.update(file_path.encode())
                        hasher.update(content)
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
        
        return hasher.hexdigest()
    
    def _is_meaningful_file(self, file_path: str, patterns: List[str]) -> bool:
        """Check if file matches meaningful patterns."""
        for pattern in patterns:
            if pattern.endswith('*'):
                if file_path.endswith(pattern[:-1]):
                    return True
            elif pattern.startswith('*'):
                if file_path.endswith(pattern[1:]):
                    return True
            elif file_path == pattern:
                return True
        return False
    
    def _get_changed_files(self) -> List[str]:
        """Get list of changed files since last commit."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            changed_files = result.stdout.strip().split('\n') if result.stdout else []
            
            # Also check staged files
            result = subprocess.run(
                ["git", "diff", "--name-only", "--cached"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            staged_files = result.stdout.strip().split('\n') if result.stdout else []
            
            # Combine and deduplicate
            all_changed = list(set(changed_files + staged_files))
            return [f for f in all_changed if f]  # Remove empty strings
            
        except Exception as e:
            logger.error(f"Error getting changed files: {e}")
            return []
    
    def _is_timestamp_only_change(self, changed_files: List[str]) -> bool:
        """Check if changes are only timestamp updates."""
        if not changed_files:
            return True
            
        # If only README.md changed, check if it's just timestamps
        if len(changed_files) == 1 and "README.md" in changed_files[0]:
            try:
                result = subprocess.run(
                    ["git", "diff", "HEAD", "README.md"],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_root
                )
                diff_content = result.stdout
                
                # Check if diff only contains timestamp changes
                lines = diff_content.split('\n')
                meaningful_changes = 0
                
                for line in lines:
                    if line.startswith('+') or line.startswith('-'):
                        # Skip diff headers
                        if line.startswith('+++') or line.startswith('---'):
                            continue
                        # Check if line contains only timestamp/date changes
                        if any(pattern in line for pattern in [
                            'Última Sincronização',
                            'Sincronizado em',
                            '2025-09-25',
                            'T19:1',  # Time pattern
                            'auto: sync from server'
                        ]):
                            continue
                        meaningful_changes += 1
                
                return meaningful_changes == 0
                
            except Exception as e:
                logger.error(f"Error checking timestamp changes: {e}")
                return False
        
        return False
    
    def _optimize_files(self, files: List[str]) -> None:
        """Optimize files before committing."""
        if not self.config.get("auto_optimization", {}).get("enabled", False):
            return
            
        for file_path in files:
            if file_path.endswith('.py'):
                self._optimize_python_file(file_path)
            elif file_path.endswith('.json'):
                self._optimize_json_file(file_path)
    
    def _optimize_python_file(self, file_path: str) -> None:
        """Optimize Python file."""
        try:
            # Run basic formatting check (without external dependencies)
            logger.info(f"Checking Python file: {file_path}")
            
            full_path = self.repo_root / file_path
            if full_path.exists():
                # Basic syntax check
                subprocess.run([
                    sys.executable, "-m", "py_compile", str(full_path)
                ], check=False, capture_output=True)
                
        except Exception as e:
            logger.warning(f"Error optimizing {file_path}: {e}")
    
    def _optimize_json_file(self, file_path: str) -> None:
        """Optimize JSON file."""
        try:
            full_path = self.repo_root / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Rewrite with consistent formatting
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
                logger.info(f"Optimized JSON file: {file_path}")
                
        except Exception as e:
            logger.warning(f"Error optimizing JSON {file_path}: {e}")
    
    def _update_metrics(self) -> None:
        """Update repository metrics."""
        if not self.config.get("metrics_tracking", {}).get("enabled", False):
            return
            
        try:
            # Count files and lines
            total_files = 0
            total_lines = 0
            
            for file_path in self.repo_root.rglob("*"):
                if (file_path.is_file() and 
                    not any(ignore in str(file_path) for ignore in ['.git', '__pycache__', '.pyc'])):
                    total_files += 1
                    if file_path.suffix in ['.py', '.md', '.json', '.txt']:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                total_lines += len(f.readlines())
                        except Exception:
                            pass
            
            logger.info(f"Repository metrics: {total_files} files, {total_lines} lines")
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def should_sync(self) -> Tuple[bool, str]:
        """Determine if sync should proceed."""
        if not self.config.get("commit_optimization", {}).get("enabled", True):
            return True, "Optimization disabled"
        
        # Get current repository state
        current_hash = self._calculate_repo_hash()
        
        # Check if meaningful changes occurred
        if current_hash == self.last_sync_hash:
            return False, "No meaningful changes detected"
        
        # Get changed files
        changed_files = self._get_changed_files()
        
        # Check minimum changes threshold
        min_changes = self.config.get("commit_optimization", {}).get("min_changes_threshold", 1)
        if len(changed_files) < min_changes:
            return False, f"Below minimum change threshold ({len(changed_files)} < {min_changes})"
        
        # Check if changes are timestamp-only
        if (self.config.get("timestamp_management", {}).get("update_frequency") == "on_meaningful_change_only" and
            self._is_timestamp_only_change(changed_files)):
            return False, "Only timestamp changes detected"
        
        return True, f"Meaningful changes detected in {len(changed_files)} files"
    
    def optimize_and_sync(self) -> bool:
        """Optimize repository and perform sync if needed."""
        logger.info("Starting PENIN sync optimization...")
        
        # Check if sync should proceed
        should_sync, reason = self.should_sync()
        logger.info(f"Sync decision: {should_sync} - {reason}")
        
        if not should_sync:
            logger.info("Skipping sync - no meaningful changes")
            return False
        
        try:
            # Get changed files
            changed_files = self._get_changed_files()
            
            # Optimize files before committing
            if changed_files:
                logger.info(f"Optimizing {len(changed_files)} changed files...")
                self._optimize_files(changed_files)
            
            # Update metrics
            self._update_metrics()
            
            # Update sync hash
            current_hash = self._calculate_repo_hash()
            self._save_sync_hash(current_hash)
            
            logger.info("✅ Sync optimization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during sync optimization: {e}")
            return False
    
    def generate_smart_commit_message(self) -> str:
        """Generate intelligent commit message based on changes."""
        changed_files = self._get_changed_files()
        
        if not changed_files:
            return "🔄 Auto-sync: Minor updates"
        
        # Categorize changes
        categories = {
            'python': [f for f in changed_files if f.endswith('.py')],
            'docs': [f for f in changed_files if f.endswith('.md')],
            'config': [f for f in changed_files if f.endswith(('.json', '.yml', '.yaml'))],
            'tests': [f for f in changed_files if 'test' in f.lower()],
            'other': []
        }
        
        # Classify remaining files
        for f in changed_files:
            if not any(f in cat_files for cat_files in categories.values()):
                categories['other'].append(f)
        
        # Build commit message
        message_parts = []
        
        if categories['python']:
            message_parts.append(f"🐍 Python: {len(categories['python'])} files")
        if categories['docs']:
            message_parts.append(f"📝 Docs: {len(categories['docs'])} files")
        if categories['config']:
            message_parts.append(f"⚙️ Config: {len(categories['config'])} files")
        if categories['tests']:
            message_parts.append(f"🧪 Tests: {len(categories['tests'])} files")
        if categories['other']:
            message_parts.append(f"📁 Other: {len(categories['other'])} files")
        
        if message_parts:
            return f"🚀 PENIN Optimizer: {', '.join(message_parts)}"
        else:
            return "🔄 PENIN Auto-sync: Repository updates"


def main():
    """Main function."""
    print("🧠 PENIN Code Optimizer - Intelligent Sync System")
    print("=" * 50)
    
    optimizer = PENINSyncOptimizer()
    
    # Run optimization
    success = optimizer.optimize_and_sync()
    
    if success:
        print("✅ Optimization completed - ready for sync")
        commit_msg = optimizer.generate_smart_commit_message()
        print(f"💬 Suggested commit message: {commit_msg}")
    else:
        print("ℹ️ No sync needed - repository is optimized")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())