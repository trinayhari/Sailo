import modal

# Create Modal app
app = modal.App("deepseek-test")

# DeepSeek R1 LLM setup based on Modal documentation
def download_model():
    import subprocess
    import os
    
    # Download DeepSeek R1 model
    model_url = "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf"
    model_path = "/models/deepseek-r1.gguf"
    
    os.makedirs("/models", exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"📥 Downloading DeepSeek R1 model...")
        subprocess.run([
            "wget", "-O", model_path, model_url
        ], check=True)
        print(f"✅ Model downloaded to {model_path}")
    else:
        print(f"✅ Model already exists at {model_path}")

# Build image with llama.cpp and DeepSeek R1
image = (
    modal.Image.debian_slim()
    .apt_install("wget", "build-essential", "cmake", "git")
    .run_commands(
        # Clone and build llama.cpp with CMake (not deprecated Makefile)
        "git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp",
        "cd /llama.cpp && mkdir build && cd build",
        "cd /llama.cpp/build && cmake .. && make -j$(nproc)",
    )
    .run_function(download_model)
    .pip_install(["fastapi"])
)

@app.function(
    image=image,
    timeout=300,
)
def test_deepseek_llm(prompt: str):
    """Test DeepSeek R1 LLM with a simple prompt"""
    import subprocess
    import json
    
    model_path = "/models/deepseek-r1.gguf"
    
    try:
        print(f"🤖 Testing DeepSeek R1 with prompt: {prompt[:100]}...")
        
        # Run llama.cpp with DeepSeek R1
        command = [
            "/llama.cpp/build/bin/llama-cli",
            "--model", model_path,
            "--prompt", prompt,
            "--n-predict", "200",
            "--temp", "0.7",
            "--top-p", "0.9",
            "--threads", "4",
        ]
        
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            timeout=120
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"LLM failed: {result.stderr}",
                "output": None
            }
        
        output = result.stdout.strip()
        print(f"✅ DeepSeek R1 response: {output[:200]}...")
        
        return {
            "success": True,
            "error": None,
            "output": output,
            "model": "DeepSeek R1",
            "prompt_length": len(prompt),
            "response_length": len(output)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "output": None
        }

@modal.web_endpoint(method="POST")
def test_llm_endpoint(request_data: dict):
    """Simple endpoint to test DeepSeek R1 LLM"""
    try:
        prompt = request_data.get("prompt", "Hello, how are you?")
        
        print(f"🚀 Testing LLM with prompt: {prompt}")
        
        # Call the LLM function
        result = test_deepseek_llm.remote(prompt)
        
        return {
            "success": True,
            "llm_result": result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "llm_result": None
        }

@modal.web_endpoint(method="GET")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "deepseek-r1-test"}

if __name__ == "__main__":
    # Test locally
    with app.run():
        result = test_deepseek_llm.remote("What is 2+2?")
        print(f"Test result: {result}")
