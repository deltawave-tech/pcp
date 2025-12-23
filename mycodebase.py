#!/usr/bin/env python3
"""
Script to generate a structured markdown file containing the entire codebase.
Supports ignoring specific folders and file suffixes.
"""

import fnmatch
import os
import argparse
from pathlib import Path
from typing import Set, List

# Default ignore patterns
DEFAULT_IGNORE_FOLDERS = {
    '.git', '__pycache__', 'node_modules', '.pytest_cache', 
    '.mypy_cache', 'dist', 'build', '.venv', 'venv', 
    '.idea', '.vscode', 'env','.env','.github','runs_metadata', 'scripts',
    'backend/paper-qa','backend/gurnemanz','backend/tests'
    'backend/plip','backend/data', 'backend/.credentials/', 'backend/.cursor/', 'backend/plip',
    'backend/tests','backend/eve_app/ipc/tests','backend/pipeline_run_cache', 'backend/data','backend/--no-cache',
    'paper-qa/','backend/eva_app/src/transition'
}

DEFAULT_IGNORE_SUFFIXES = {
    '.pyc', '.pyo', '.pyd', '.so', '.dylib', '.dll',
    '.egg-info', '.DS_Store', '.gitignore', '.log',
    '.tmp', '.temp', '.swp', '.swo', '~','.pdb','.pkl',
    '.sdf','.gz','.mol','.pdf','.csv','.log', '.md','.ipynb','.code-workspace',
    '.html', '.pdf'
}

def build_include_patterns(patterns: List[str]) -> Set[str]:
    """Normalize include patterns so they can be matched against relative paths."""
    normalized_patterns: Set[str] = set()
    for pattern in patterns:
        cleaned = pattern.strip()
        if not cleaned:
            continue
        cleaned = cleaned.replace('\\', '/').lstrip('./').lower()
        normalized_patterns.add(cleaned)
    return normalized_patterns

def should_ignore(path: Path,
                  root_path: Path,
                  ignore_folders: Set[str],
                  ignore_suffixes: Set[str],
                  include_patterns: Set[str]) -> bool:
    """Determine if a path should be ignored, respecting include overrides."""
    # Convert path to string for checking relative paths
    path_str = str(path)
    filename = path.name.lower()
    relative_path = path.relative_to(root_path).as_posix().lower()

    # Allow include overrides before applying any ignore checks
    for pattern in include_patterns:
        if fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(filename, pattern):
            return False
    
    # ROBUST: Ignore specific filenames ANYWHERE in the project (case insensitive)
    ignore_filenames = {
        'package.json', 'package-lock.json', 'yarn.lock', 'bun.lock',
        'composer.json', 'composer.lock', 'pipfile', 'pipfile.lock',
        'requirements.txt', 'pyproject.toml', 'poetry.lock',
        'cargo.toml', 'cargo.lock', 'go.mod', 'go.sum',
        'ngl.js', 'three.js', 'jquery.js', 'bootstrap.js'
    }
    
    if filename in ignore_filenames:
        return True
    
    # ROBUST: Ignore any file matching these patterns ANYWHERE
    ignore_patterns = [
        'min.js', 'bundle.js', 'chunk.js', 'vendor.js', 'runtime.js',
        '.min.css', '.bundle.css', '.chunk.css',
        '.d.ts', '.map'
    ]
    
    for pattern in ignore_patterns:
        if filename.endswith(pattern):
            return True
    
    # Check if any ignore pattern matches the path
    for ignore_pattern in ignore_folders:
        # Handle both single folder names and relative paths
        if '/' in ignore_pattern:
            # For patterns like 'backend/.github', check if the path contains this pattern
            if ignore_pattern in path_str or path_str.startswith(ignore_pattern + '/'):
                return True
        else:
            # For single folder names, check if any part matches
            if ignore_pattern in path.parts:
                return True
    
    # Special case: ignore compiled JS/CSS files from node_modules even if JS/CSS aren't globally ignored
    if 'node_modules' in path.parts:
        if path.suffix in {'.js', '.mjs', '.css', '.map', '.d.ts', '.ts', '.tsx', '.jsx'}:
            return True
        # Also ignore minified files and build artifacts
        if any(part in filename for part in ['min.', 'bundle.', 'chunk.', 'vendor.', 'runtime.', 'dist.']):
            return True
        # Ignore any file in node_modules/*/dist/ or node_modules/*/lib/ or node_modules/*/build/
        if any(part in path.parts for part in ['dist', 'lib', 'build', 'esm', 'cjs', 'umd']):
            return True
    
    # Check file suffix
    if path.suffix in ignore_suffixes:
        return True
    
    # Check if filename starts with dot (hidden files)
    if path.name.startswith('.') and path.name not in {'.gitignore', '.env.example'}:
        return True
    
    return False

def get_file_tree(root_path: Path,
                  ignore_folders: Set[str],
                  ignore_suffixes: Set[str],
                  include_patterns: Set[str]) -> List[Path]:
    """Get all files in the directory tree, respecting ignore patterns."""
    files = []
    
    for file_path in root_path.rglob('*'):
        if file_path.is_file() and not should_ignore(file_path, root_path, ignore_folders, ignore_suffixes, include_patterns):
            files.append(file_path)
    
    return sorted(files)

def get_language_from_extension(suffix: str) -> str:
    """Map file extension to markdown language identifier."""
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'jsx',
        '.tsx': 'tsx',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'zsh',
        '.fish': 'fish',
        '.ps1': 'powershell',
        '.sql': 'sql',
        '.html': 'html',
        '.htm': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.less': 'less',
        '.xml': 'xml',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'ini',
        '.conf': 'ini',
        '.md': 'markdown',
        '.txt': 'text',
        '.dockerfile': 'dockerfile',
        '.makefile': 'makefile',
        '.r': 'r',
        '.R': 'r',
        '.m': 'matlab',
        '.scala': 'scala',
        '.kt': 'kotlin',
        '.swift': 'swift',
        '.dart': 'dart',
        '.lua': 'lua',
        '.pl': 'perl',
        '.vim': 'vim'
    }
    return language_map.get(suffix.lower(), 'text')

def generate_markdown(root_path: Path, files: List[Path], output_file: str):
    """Generate the markdown documentation."""
    project_name = root_path.name
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# {project_name} - Complete Codebase\n\n")
        f.write(f"Generated from: `{root_path.absolute()}`\n\n")
        f.write(f"Total files: {len(files)}\n\n")
        
        # Table of Contents
        f.write("## Table of Contents\n\n")
        for file_path in files:
            relative_path = file_path.relative_to(root_path)
            anchor = str(relative_path).replace('/', '').replace('.', '').replace('_', '').replace('-', '').lower()
            f.write(f"- [{relative_path}](#{anchor})\n")
        f.write("\n")
        
        # File contents
        f.write("## Files\n\n")
        
        for file_path in files:
            relative_path = file_path.relative_to(root_path)
            f.write(f"### {relative_path}\n\n")
            f.write(f"**Path:** `{relative_path}`\n\n")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file_content:
                    content = file_content.read()
                    language = get_language_from_extension(file_path.suffix)
                    f.write(f"```{language}\n{content}\n```\n\n")
            except UnicodeDecodeError:
                f.write("*Binary file - content not displayed*\n\n")
            except Exception as e:
                f.write(f"*Error reading file: {str(e)}*\n\n")
            
            f.write("---\n\n")

def main():
    parser = argparse.ArgumentParser(description='Generate markdown documentation of codebase')
    parser.add_argument('--root', '-r', type=str, default='.', 
                       help='Root directory to scan (default: current directory)')
    # parser.add_argument('--output', '-o', type=str, default='codebase.md',
    #                    help='Output markdown file (default: codebase.md)')
    parser.add_argument('--output', '-o', type=str, default='~/Downloads/codebase.md',
                   help='Output markdown file (default: ~/Downloads/codebase.md)')
    parser.add_argument('--ignore-folders', type=str, nargs='*', default=[],
                       help='Additional folders to ignore')
    parser.add_argument('--ignore-suffixes', type=str, nargs='*', default=[],
                       help='Additional file suffixes to ignore')
    parser.add_argument('--include', '-i', type=str, nargs='*', default=[],
                       help='Paths or glob patterns to always include even if ignored')
    parser.add_argument('--include-default-ignores', action='store_true', default=True,
                       help='Include default ignore patterns (default: True)')
    parser.add_argument('--no-default-ignores', action='store_true',
                       help='Disable default ignore patterns')
    
    args = parser.parse_args()
    
    root_path = Path(args.root).resolve()
    
    if not root_path.exists() or not root_path.is_dir():
        print(f"Error: {root_path} is not a valid directory")
        return 1
    
    # Setup ignore patterns
    ignore_folders = set(args.ignore_folders)
    ignore_suffixes = set(args.ignore_suffixes)
    include_overrides = args.include or []
    include_patterns = build_include_patterns(include_overrides)
    
    # Comprehensive ignore patterns for metadata, data, env, and shell files
    ignore_folders.update({
        # Version control and config
        '.claude', '.github', 'backend/.github', 'backend/.claude',
        
        # Data and metadata directories
        'backend/runs_metadata', 'backend/plip_reports',
        'backend/transition', 'runs_metadata', 'plip_reports',
        'old.md','backend/gurnemanz'
        
        # Build and distribution directories
        'frontend/node_modules', 'node_modules', 'dist', 'build', 'dist-newstyle',
        '.next', 'frontend/.next', '.nuxt', '.output',
        'frontend/public/lib', 'public/lib', 'lib', 'libs', 'vendor',
        
        # Temporary and cache directories
        'logs', 'temp', 'tmp', 'cache', '__pycache__', '.pytest_cache', 
        '.mypy_cache', 'docking', 'mol_gen', 'active_inactive_compounds',
        'pdb_files', 'pubmed_articles',
        
        # Environment and virtual env
        '.venv', 'venv', 'env', '.env',
        
        # IDE and editor
        '.idea', '.vscode',
        
        # SSL certificates
        'certs',
        
        # Test and development
        'test_real_data', 'quick_test', 'confidence_fix_test', 'test_run',

        # Specific frontend files to ignore
        'frontend/eslint.config.mjs',
        'frontend/next.config.ts',
        'frontend/postcss.config.mjs',
        'frontend/public/drugbank_compounds.tsv',
        'frontend/public/studiomilitary.wav',
        'frontend/public/test.html',
        'frontend/tailwind.config.ts',
        'frontend/tsconfig.json'
    })
    
    # Additional file suffixes to ignore
    ignore_suffixes.update({
        # Shell scripts and environment files
        '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
        '.env', '.env.local', '.env.development', '.env.production', '.env.example',
        
        # Data files
        '.csv', '.xml', '.yaml', '.yml', '.toml', 
        '.pdb', '.sdf', '.mol', '.mol2',
        
        # CSS and styling files (usually from node_modules)
        '.css', '.scss', '.sass', '.less', '.styl',
        
        # Log files
        '.log', '.out', '.err',
        
        # Archive and compressed files
        '.tar', '.tar.gz', '.tgz', '.zip', '.rar', '.7z', '.gz', '.bz2',
        
        # Binary and compiled files
        '.exe', '.dll', '.so', '.dylib', '.o', '.obj', '.lib', '.a',
        
        # Database files
        '.db', '.sqlite', '.sqlite3', '.mdb',
        
        # Image files (usually not needed in code analysis)
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.ico',
        
        # Package manager and dependency files
        '.lock', 'package.json', 'package-lock.json', 'yarn.lock', 'bun.lock',
        'composer.json', 'composer.lock', 'Pipfile', 'Pipfile.lock',
        'requirements.txt', 'pyproject.toml', 'poetry.lock',
        
        # Temporary files
        '.tmp', '.temp', '.bak', '.backup', '.swp', '.swo', '~',
        
        # Certificate and key files
        '.pem', '.key', '.cert', '.crt', '.p12', '.pfx',
        
        # Pickle and serialized files
        '.pkl', '.pickle', '.joblib'
    })
    
    if not args.no_default_ignores:
        ignore_folders.update(DEFAULT_IGNORE_FOLDERS)
        ignore_suffixes.update(DEFAULT_IGNORE_SUFFIXES)
    
    print(f"Scanning directory: {root_path}")
    print(f"Ignoring folders: {', '.join(sorted(ignore_folders))}")
    print(f"Ignoring suffixes: {', '.join(sorted(ignore_suffixes))}")
    if include_overrides:
        print(f"Forcing inclusion of: {', '.join(sorted(include_overrides))}")
    
    # Get all files
    files = get_file_tree(root_path, ignore_folders, ignore_suffixes, include_patterns)
    print(f"Found {len(files)} files")
    
    # Generate markdown
    print(f"Generating markdown file: {args.output}")
    output_path = Path(args.output)
    output_path = Path('~/Downloads/') / output_path
    output_path = output_path.expanduser()
    generate_markdown(root_path, files, output_path)
    # generate_markdown(root_path, files, args.output)
    print("Done!")
    
    return 0

if __name__ == '__main__':
    exit(main())
