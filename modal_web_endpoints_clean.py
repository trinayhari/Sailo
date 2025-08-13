import modal
from typing import Dict, Any
import json
import hashlib
import random
from datetime import datetime

# Create Modal app
app = modal.App("sailo-web-api")

# Define the image with required dependencies
image = modal.Image.debian_slim().pip_install([
    "pandas",
    "numpy", 
    "requests",
    "supabase",
    "fastapi"
])

# Real LLM image with llama.cpp CUDA support
MINUTES = 60
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

llama_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands("git clone https://github.com/ggerganov/llama.cpp")
    .workdir("/llama.cpp")
    .run_commands(
        "make GGML_CUDA=1 CUDA_DOCKER_ARCH=all",
        "curl -L -o models/Llama-3.2-3B-Instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    )
    .pip_install(["pandas", "numpy", "requests", "supabase"])
)

# Model cache volume
model_cache = modal.Volume.from_name("llama-models", create_if_missing=True)
cache_dir = "/cache"

@app.function(
    image=llama_image,
    volumes={cache_dir: model_cache},
    timeout=600,
    secrets=[modal.Secret.from_name("supabase-secret")]
)
def analyze_data_with_llm(analysis_prompt: str, model_name: str = "llama3.2"):
    """Perform comprehensive data analysis using REAL LLM (llama.cpp)"""
    import subprocess
    import json
    import os
    
    cache_dir = "/cache"
    
    # Model configurations
    model_configs = {
        "llama3.2": {
            "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "description": "Llama 3.2 3B Instruct",
            "context_size": 8192
        }
    }
    
    model_config = model_configs.get(model_name, model_configs["llama3.2"])
    
    try:
        # Run the model with llama.cpp
        model_file = f"{cache_dir}/{model_config['filename']}"
        command = [
            "/llama.cpp/llama-cli",
            "--model", model_file,
            "--prompt", analysis_prompt,
            "--n-predict", "1000",  # More tokens for comprehensive analysis
            "--threads", "4",
            "-no-cnv",
            "--ctx-size", str(model_config['context_size']),
            "--temp", "0.3",  # Slightly higher temperature for creative analysis
        ]
        
        print(f"🤖 Running {model_config['description']} for data analysis...")
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise Exception(f"{model_name} failed: {result.stderr}")
        
        # Extract JSON from response
        response_text = result.stdout.strip()
        print(f"🤖 Raw {model_name} response: {response_text[:300]}...")
        
        # Try to find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in response")
        
        json_text = response_text[json_start:json_end]
        analysis_result = json.loads(json_text)
        
        print(f"✅ {model_name} analysis completed successfully")
        return json.dumps(analysis_result, indent=2)
        
    except Exception as e:
        print(f"❌ {model_name} analysis failed: {str(e)}")
        return None

# Universal analysis function that uses the real LLM
def universal_analysis(analysis_prompt: str, user_query: str, unique_id: str) -> Dict[str, Any]:
    """Universal analysis function that calls the real LLM for all analysis"""
    
    analysis_result = {
        "type": "real_llm_analysis",
        "query": user_query,
        "domain_detected": "universal",
        "interpretation": "LLM-native analysis",
        "analysis_performed": "Real LLM data analysis",
        "total_anomalies": 0,
        "message": f"Real LLM analysis completed for: {user_query}",
        "summary": "",
        "anomalies": []
    }
    
    # TRUE LLM-NATIVE ANALYSIS - NO HARDCODED LOGIC
    # Pass the full prompt and data to LLM and return LLM output as-is
    try:
        print(f"🤖 Calling real LLM for analysis: {user_query}")
        
        # Call the real LLM function with the full analysis prompt
        llm_result = analyze_data_with_llm.remote(analysis_prompt)
        
        if llm_result:
            # Parse LLM result and return as-is
            try:
                llm_analysis = json.loads(llm_result)
                
                # Use LLM output directly - no modifications
                analysis_result["summary"] = llm_analysis.get("summary", "LLM analysis completed")
                analysis_result["total_anomalies"] = llm_analysis.get("total_anomalies", 0)
                analysis_result["anomalies"] = llm_analysis.get("anomalies", [])
                
                print(f"✅ Real LLM analysis successful: {len(analysis_result['anomalies'])} insights")
                
            except json.JSONDecodeError as e:
                print(f"⚠️ LLM returned non-JSON result: {e}")
                # If LLM returns non-JSON, treat as text summary
                analysis_result["summary"] = str(llm_result)
                analysis_result["total_anomalies"] = 1
                analysis_result["anomalies"] = [{
                    "symbol": "LLM Analysis",
                    "value": "Text analysis",
                    "threshold": "LLM determined",
                    "severity": "INFO",
                    "details": str(llm_result),
                    "action_required": "Review LLM analysis",
                    "business_impact": "LLM-determined insights",
                    "reason": "Direct LLM output"
                }]
        else:
            print("⚠️ LLM returned no result - using fallback")
            analysis_result["summary"] = "LLM analysis failed - no result returned"
            analysis_result["total_anomalies"] = 0
            analysis_result["anomalies"] = []
            
    except Exception as e:
        print(f"❌ LLM analysis failed: {e}")
        analysis_result["summary"] = f"LLM analysis error: {str(e)}"
        analysis_result["total_anomalies"] = 0
        analysis_result["anomalies"] = []
    
    return analysis_result

@app.function(image=image)
def fetch_supabase_data(table_name: str, limit: int = 5):
    """Fetch sample data from Supabase for analysis"""
    import os
    from supabase import create_client, Client
    
    try:
        # Get Supabase credentials from Modal secrets
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("Missing Supabase credentials")
        
        # Create Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Fetch sample data
        response = supabase.table(table_name).select("*").limit(limit).execute()
        
        if response.data:
            print(f"✅ Fetched {len(response.data)} records from {table_name}")
            return response.data
        else:
            print(f"⚠️ No data found in table {table_name}")
            return []
            
    except Exception as e:
        print(f"❌ Error fetching data from {table_name}: {e}")
        return []

@modal.web_endpoint(method="POST")
def run_monitoring_scenario(request_data: Dict[str, Any]):
    """Universal LLM-native data analysis endpoint"""
    try:
        query = request_data.get("query", "analyze data")
        table_name = request_data.get("table_name", "options_trades")
        
        # Generate unique ID for this analysis
        unique_id = hashlib.md5(f"{query}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        
        print(f"🚀 Starting LLM-native analysis (ID: {unique_id})")
        print(f"📝 User query: {query}")
        print(f"📊 Table: {table_name}")
        
        # Fetch sample data from Supabase
        sample_data = fetch_supabase_data.remote(table_name, 5)
        
        if not sample_data:
            return {
                "success": False,
                "error": f"No data available in table {table_name}",
                "results": None
            }
        
        # Build analysis prompt for LLM
        analysis_prompt = f"""
You are a universal data analysis AI. Analyze the following data and user query to provide actionable business insights.

USER QUERY: {query}

TABLE: {table_name}
TOTAL RECORDS: {len(sample_data)}
TOTAL COLUMNS: {len(sample_data[0].keys()) if sample_data else 0}

COLUMN NAMES: {list(sample_data[0].keys()) if sample_data else []}

SAMPLE DATA (first 5 records):
{json.dumps(sample_data, indent=2)}

INSTRUCTIONS:
1. Analyze the data in the context of the user's query
2. Provide specific, actionable recommendations based on the actual data values
3. Return your analysis in the following JSON format:

{{
    "summary": "Brief summary of your analysis",
    "total_anomalies": <number of insights/recommendations>,
    "anomalies": [
        {{
            "symbol": "Specific data identifier (e.g., 'AAPL CALL', 'Record_1')",
            "value": "Specific data values (e.g., 'Strike $150, Premium $5.25')",
            "threshold": "Your analysis criteria",
            "severity": "HIGH/MEDIUM/LOW",
            "details": "Detailed explanation of your finding",
            "action_required": "Specific action recommendation (e.g., 'BUY AAPL $150 CALL')",
            "business_impact": "Expected business impact",
            "reason": "Why this recommendation makes sense"
        }}
    ]
}}

Focus on providing specific, data-driven recommendations based on the actual values in the sample data.
"""
        
        # Call universal LLM analysis
        results = universal_analysis(analysis_prompt, query, unique_id)
        
        # Add data info to response
        data_info = {
            "table": table_name,
            "rows": len(sample_data),
            "columns": len(sample_data[0].keys()) if sample_data else 0,
            "column_names": list(sample_data[0].keys()) if sample_data else []
        }
        
        return {
            "success": True,
            "results": results,
            "data_info": data_info
        }
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": None
        }

@modal.web_endpoint(method="GET")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "universal-llm-native-analysis"}

@modal.web_endpoint(method="POST")
def test_custom_analysis(request_data: Dict[str, Any]):
    """Test endpoint for custom LLM analysis"""
    try:
        query = request_data.get("query", "test analysis")
        
        # Simple test prompt
        test_prompt = f"""
Analyze this query: {query}

Return a JSON response with:
{{
    "summary": "Test analysis completed",
    "total_anomalies": 1,
    "anomalies": [
        {{
            "symbol": "Test",
            "value": "LLM working",
            "threshold": "Test threshold",
            "severity": "INFO",
            "details": "This is a test of the LLM analysis system",
            "action_required": "Verify LLM is working",
            "business_impact": "System validation",
            "reason": "Testing LLM integration"
        }}
    ]
}}
"""
        
        # Call LLM directly
        llm_result = analyze_data_with_llm.remote(test_prompt)
        
        if llm_result:
            try:
                analysis = json.loads(llm_result)
                return {"success": True, "llm_result": analysis}
            except json.JSONDecodeError:
                return {"success": True, "llm_result": {"raw_output": str(llm_result)}}
        else:
            return {"success": False, "error": "LLM returned no result"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}
