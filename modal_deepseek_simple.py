import modal

# Create Modal app
app = modal.App("deepseek-simple-test")

# Simplified image setup based on Modal's llama.cpp example
def download_model():
    import subprocess
    import os
    
    # Download DeepSeek R1 model (smaller version for testing)
    model_url = "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf"
    model_path = "/models/deepseek-r1.gguf"
    
    os.makedirs("/models", exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"📥 Downloading DeepSeek R1 model...")
        subprocess.run([
            "curl", "-L", "-o", model_path, model_url
        ], check=True)
        print(f"✅ Model downloaded to {model_path}")
    else:
        print(f"✅ Model already exists at {model_path}")

# Build image with proper dependencies
image = (
    modal.Image.debian_slim()
    .apt_install(
        "curl", 
        "build-essential", 
        "cmake", 
        "git",
        "libcurl4-openssl-dev",  # Fix for CURL dependency
        "pkg-config"
    )
    .run_commands(
        # Clone llama.cpp
        "git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp",
        # Build with CMake and proper CURL support
        "cd /llama.cpp && mkdir -p build && cd build",
        "cd /llama.cpp/build && cmake -DLLAMA_CURL=ON .. && make -j$(nproc) llama-cli",
    )
    .run_function(download_model)
    .pip_install(["fastapi"])
)

@app.function(
    image=image,
    timeout=300,
)
def test_deepseek_simple(prompt: str):
    """Simple test of DeepSeek R1 LLM"""
    import subprocess
    import os
    
    model_path = "/models/deepseek-r1.gguf"
    llama_cli = "/llama.cpp/build/bin/llama-cli"
    
    # Check if files exist
    if not os.path.exists(model_path):
        return {"success": False, "error": f"Model not found at {model_path}"}
    
    if not os.path.exists(llama_cli):
        return {"success": False, "error": f"llama-cli not found at {llama_cli}"}
    
    try:
        print(f"🤖 Testing DeepSeek R1...")
        print(f"📝 Prompt: {prompt}")
        
        # Simple llama.cpp command
        command = [
            llama_cli,
            "--model", model_path,
            "--prompt", prompt,
            "--n-predict", "100",
            "--temp", "0.7",
            "--threads", "2",
            "--no-display-prompt"
        ]
        
        print(f"🔧 Running command: {' '.join(command[:4])}...")
        
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"LLM failed with code {result.returncode}",
                "stderr": result.stderr,
                "stdout": result.stdout
            }
        
        output = result.stdout.strip()
        print(f"✅ DeepSeek R1 response: {output[:100]}...")
        
        return {
            "success": True,
            "output": output,
            "model": "DeepSeek R1 1.5B",
            "prompt": prompt,
            "response_length": len(output)
        }
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "LLM timeout after 60 seconds"}
    except Exception as e:
        return {"success": False, "error": f"Exception: {str(e)}"}

@modal.web_endpoint(method="POST")
def test_llm(request_data: dict):
    """Test endpoint for DeepSeek R1"""
    try:
        prompt = request_data.get("prompt", "What is 2+2? Answer briefly.")
        
        print(f"🚀 Testing DeepSeek R1 LLM...")
        
        # Call the LLM function
        result = test_deepseek_simple.remote(prompt)
        
        return {
            "success": True,
            "test_result": result,
            "service": "deepseek-r1-test"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "test_result": None
        }

@modal.web_endpoint(method="GET")
def health():
    """Health check"""
    return {"status": "healthy", "service": "deepseek-r1-simple-test"}

if __name__ == "__main__":
    # Local test
    with app.run():
        result = test_deepseek_simple.remote("Hello! What is your name?")
        print(f"Local test result: {result}")
