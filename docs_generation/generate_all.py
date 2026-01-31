"""
Generate all documentation artifacts for GitHub Pages.

This is the main entry point for generating the interactive documentation.
It runs all generation scripts in sequence:
1. Network graphs (per-line maps with period filtering)
2. EDA plots (delay distributions, hourly patterns, etc.)

Usage:
    python docs_generation/generate_all.py

Or from the docs_generation directory:
    python generate_all.py
"""

import subprocess
import sys
from pathlib import Path
import time

SCRIPT_DIR = Path(__file__).parent


def run_script(script_name: str, description: str) -> bool:
    """Run a generation script and return success status."""
    script_path = SCRIPT_DIR / script_name
    
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script: {script_path.name}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SCRIPT_DIR.parent),  # Run from code/ directory
            capture_output=False,
            text=True,
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ {description} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"\n✗ {description} failed (exit code {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n✗ {description} failed with error: {e}")
        return False


def main():
    print("="*70)
    print("DOCUMENTATION GENERATION")
    print("Generating all artifacts for GitHub Pages")
    print("="*70)
    
    start_time = time.time()
    
    # Define scripts to run in order
    scripts = [
        ("network_graphs.py", "Network Graphs (per-line maps)"),
        ("eda_plots.py", "EDA Plots (delay distributions)"),
    ]
    
    results = []
    for script_name, description in scripts:
        success = run_script(script_name, description)
        results.append((description, success))
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)
    
    all_success = True
    for description, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {description}")
        if not success:
            all_success = False
    
    print(f"\nTotal time: {elapsed:.1f}s")
    
    if all_success:
        print("\n✓ All documentation generated successfully!")
        print("\nOutput: docs/")
        print("  - lines/{all,pre,post}/  : Network maps")
        print("  - data/{all,pre,post}/   : JSON data")
        print("  - plots/{all,pre,post}/  : EDA plots")
        print("  - index.html             : Main page with period toggle")
        print("\nTo deploy: push docs/ to GitHub and enable GitHub Pages")
    else:
        print("\n✗ Some scripts failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
