#!/usr/bin/env python3
"""
Setup script for Modal integration with Supabase Agent MVP.
Run this after setting up your Modal account.
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n📋 {description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   {e.stderr.strip()}")
        return False

def main():
    print("🚀 Setting up Modal for Supabase Agent MVP")
    print("=" * 50)
    
    # Check if Modal is installed
    print("\n1. Checking Modal installation...")
    if not run_command("modal --version", "Checking Modal version"):
        print("\n💡 Installing Modal...")
        run_command("pip install modal", "Installing Modal")
    
    # Setup Modal account
    print("\n2. Setting up Modal account...")
    print("💡 If you haven't signed up for Modal yet:")
    print("   Visit: https://modal.com/signup")
    print("   Then run: modal setup")
    
    setup_result = run_command("modal token verify", "Verifying Modal token")
    if not setup_result:
        print("\n🔧 Please run 'modal setup' first to authenticate")
        return
    
    # Create Modal secrets for OpenAI
    print("\n3. Setting up secrets...")
    print("💡 You'll need to create a secret for OpenAI API key")
    print("   Run this command with your actual OpenAI API key:")
    print("   modal secret create openai-secret OPENAI_API_KEY=sk-your-key-here")
    
    # Test the deployment
    print("\n4. Testing Modal app deployment...")
    if run_command("modal app deploy modal_app.py", "Deploying Modal app"):
        print("✅ Modal app deployed successfully!")
    
    # Show next steps
    print("\n" + "=" * 50)
    print("🎉 Setup complete! Next steps:")
    print("\n1. Copy env.example to .env and fill in your credentials:")
    print("   cp env.example .env")
    print("\n2. Create OpenAI secret in Modal:")
    print("   modal secret create openai-secret OPENAI_API_KEY=sk-your-key-here")
    print("\n3. Test the pipeline:")
    print("   modal run modal_app.py::test_pipeline")
    print("\n4. Start the API server:")
    print("   python -m uvicorn api:app --reload")

if __name__ == "__main__":
    main()
