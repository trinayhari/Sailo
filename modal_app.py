import os
import json
from typing import Dict, Any, Optional, List, Literal
import modal

# Create Modal app
app = modal.App("supabase-agent-mvp")

# Define the image with all dependencies
image = modal.Image.debian_slim().pip_install([
    "pandas",
    "numpy", 
    "sqlalchemy",
    "psycopg[binary]",
    "matplotlib",
    "requests",
    "pydantic",
    "openai"  # For LLM integration
])

# Simplified LLM Infrastructure using llama.cpp (CPU-only for Phi-4)
llama_image = (
    modal.Image.debian_slim()
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands("git clone https://github.com/ggerganov/llama.cpp")
    .run_commands(
        "cmake llama.cpp -B llama.cpp/build "
        "-DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=ON "  # Removed CUDA for CPU-only
    )
    .run_commands(
        "cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli"
    )
    .run_commands("cp llama.cpp/build/bin/llama-* llama.cpp")
    .entrypoint([])
    .pip_install("huggingface_hub[hf_transfer]==0.26.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

model_cache = modal.Volume.from_name("llamacpp-cache", create_if_missing=True)
cache_dir = "/root/.cache/llama.cpp"

# Model configurations - easy to switch!
AVAILABLE_MODELS = {
    "phi-4": {
        "repo_id": "unsloth/phi-4-GGUF",
        "pattern": "*Q2_K*",
        "filename": "phi-4-Q2_K.gguf",
        "context_size": 4096,
        "description": "Microsoft Phi-4 (14B) - Great for structured tasks"
    },
    "phi-3.5-mini": {
        "repo_id": "microsoft/Phi-3.5-mini-instruct-gguf",
        "pattern": "*Q4_K_M*",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf", 
        "context_size": 8192,
        "description": "Phi-3.5 Mini (3.8B) - Very fast and capable"
    },
    "qwen2.5-7b": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "pattern": "*q4_k_m*",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "context_size": 8192,
        "description": "Qwen2.5 (7B) - Excellent reasoning"
    },
    "llama-3.2-3b": {
        "repo_id": "huggingface/Llama-3.2-3B-Instruct-GGUF",
        "pattern": "*Q4_K_M*",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "context_size": 8192,
        "description": "Llama 3.2 (3B) - Very fast"
    },
    "mistral-7b": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "pattern": "*q4_k_m*",
        "filename": "mistral-7b-instruct-v0.1.q4_k_m.gguf",
        "context_size": 8192,
        "description": "Mistral 7B - Fast and reliable"
    }
}

@app.function(
    image=llama_image,
    volumes={cache_dir: model_cache},
    timeout=15 * 60  # 15 minutes for larger models
)
def download_model(model_name: str = "phi-4"):
    """Download any supported model for planning tasks."""
    from huggingface_hub import snapshot_download
    
    if model_name not in AVAILABLE_MODELS:
        available = ", ".join(AVAILABLE_MODELS.keys())
        raise ValueError(f"Model '{model_name}' not supported. Available: {available}")
    
    model_config = AVAILABLE_MODELS[model_name]
    repo_id = model_config["repo_id"]
    pattern = model_config["pattern"]
    
    print(f"🤖 Downloading {model_config['description']}...")
    print(f"   Repo: {repo_id}")
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=cache_dir,
        allow_patterns=[pattern],
    )
    model_cache.commit()
    print(f"🤖 {model_name} model downloaded successfully!")

@app.function(
    image=llama_image,
    volumes={cache_dir: model_cache},
    timeout=600,
    secrets=[modal.Secret.from_name("supabase-secret")]
)
def analyze_data_with_llm(analysis_prompt: str, model_name: str = "llama3.2"):
    """Perform comprehensive data analysis using LLM"""
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

@app.function(
    image=llama_image,
    volumes={cache_dir: model_cache},
    timeout=5 * 60,  # 5 minutes for planning
    cpu=4  # Most models can run on CPU
)
def plan_with_llm(schema_info: Dict[str, Any], goal: str, slack_webhook: Optional[str] = None, model_name: str = "phi-4") -> Dict[str, Any]:
    """Use any supported LLM via llama.cpp to generate a monitoring plan, with fallback."""
    import subprocess
    import json
    
    try:
        # Validate model
        if model_name not in AVAILABLE_MODELS:
            print(f"🤖 Unknown model '{model_name}', falling back to phi-4")
            model_name = "phi-4"
        
        model_config = AVAILABLE_MODELS[model_name]
        
        # Ensure model is downloaded
        download_model.remote(model_name)
        
        # Prepare context for the LLM
        columns_info = []
        for col in schema_info.get("columns", []):
            col_desc = f"- {col['name']} ({col['dtype']})"
            if col.get('is_datetime'):
                col_desc += " [TIMESTAMP]"
            elif col.get('is_numeric'):
                col_desc += " [NUMERIC]"
            columns_info.append(col_desc)
        
        # Get sample data for context
        sample_data = schema_info.get("sample", [])[:3]  # First 3 rows to keep prompt manageable
        
        # Build the prompt (optimized for different models)
        if "llama" in model_name.lower():
            # Llama models prefer this format
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a database monitoring expert. Generate ONLY valid JSON for a monitoring plan.<|eot_id|><|start_header_id|>user<|end_header_id|>

DATABASE SCHEMA:
{chr(10).join(columns_info)}

SAMPLE DATA:
{json.dumps(sample_data, indent=2, default=str)}

USER GOAL: {goal}

Generate a JSON monitoring plan with this exact schema:
{{
  "metric": "column_name",           // must be an existing NUMERIC column
  "timestamp_col": "column_name",    // must be an existing TIMESTAMP column  
  "method": "zscore",
  "threshold": 3.0,
  "ew_span": 24,
  "schedule_minutes": 15,
  "action": "slack",
  "action_config": {{}}
}}

Return ONLY the JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            # Standard format for other models
            prompt = f"""You are a database monitoring expert. Generate ONLY valid JSON for a monitoring plan.

DATABASE SCHEMA:
{chr(10).join(columns_info)}

SAMPLE DATA:
{json.dumps(sample_data, indent=2, default=str)}

USER GOAL: {goal}

REQUIRED JSON SCHEMA:
{{
  "metric": "column_name",           // must be an existing NUMERIC column from above
  "timestamp_col": "column_name",    // must be an existing TIMESTAMP column from above  
  "method": "zscore",                // always use "zscore"
  "threshold": 3.0,                  // z-score threshold (2.0-4.0)
  "ew_span": 24,                     // exponential window span (12-168)
  "schedule_minutes": 15,            // monitoring frequency (5-60)
  "action": "slack",                 // always "slack"
  "action_config": {{}}              // empty for now
}}

RULES:
- Pick existing columns only
- Use conservative thresholds (3.0 for most cases)
- Return ONLY the JSON, no explanations or markdown

JSON:"""

        # Run the model with llama.cpp
        model_file = f"{cache_dir}/{model_config['filename']}"
        command = [
            "/llama.cpp/llama-cli",
            "--model", model_file,
            "--prompt", prompt,
            "--n-predict", "200",  # Limit tokens for JSON response
            "--threads", "4",
            "-no-cnv",
            "--ctx-size", str(model_config['context_size']),
            "--temp", "0.1",  # Low temperature for structured output
        ]
        
        print(f"🤖 Running {model_config['description']} for plan generation...")
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise Exception(f"{model_name} failed: {result.stderr}")
        
        # Extract JSON from response
        response_text = result.stdout.strip()
        print(f"🤖 Raw {model_name} response: {response_text[:200]}...")
        
        # Try to find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in response")
        
        json_text = response_text[json_start:json_end]
        plan_dict = json.loads(json_text)
        
        # Validate with Pydantic
        from pydantic import BaseModel, Field
        
        class Plan(BaseModel):
            metric: str
            timestamp_col: str
            group_by: Optional[str] = None
            method: Literal["zscore", "ewm"] = "zscore"
            threshold: float = Field(3.0, ge=0)
            ew_span: int = Field(24, ge=3, le=336)
            schedule_minutes: int = Field(15, ge=1)
            action: Literal["slack", "webhook"] = "slack"
            action_config: Dict[str, Any] = {}
        
        plan = Plan(**plan_dict)
        
        # Add Slack webhook if provided
        if slack_webhook and plan.action == "slack":
            plan.action_config["webhook_url"] = slack_webhook
        
        print(f"🤖 Generated plan using {model_name}: {plan.model_dump()}")
        return plan.model_dump()
        
    except Exception as e:
        print(f"🤖 {model_name} planning failed: {e}")
        # Fallback to deterministic plan
        return create_fallback_plan(schema_info, slack_webhook)

@app.function(image=image)
def create_fallback_plan(schema_info: Dict[str, Any], slack_webhook: Optional[str] = None) -> Dict[str, Any]:
    """Create a safe fallback plan when LLM fails."""
    
    # Find timestamp column
    timestamp_col = None
    for col in schema_info.get("columns", []):
        if col.get("is_datetime") or any(keyword in col["name"].lower() for keyword in ["ts", "time", "date", "created"]):
            timestamp_col = col["name"]
            break
    
    # Find numeric column  
    metric_col = None
    for col in schema_info.get("columns", []):
        if col.get("is_numeric"):
            metric_col = col["name"]
            break
    
    if not timestamp_col or not metric_col:
        raise ValueError("Cannot create fallback plan: missing timestamp or numeric columns")
    
    fallback_plan = {
        "metric": metric_col,
        "timestamp_col": timestamp_col,
        "method": "zscore",
        "threshold": 3.0,
        "ew_span": 24,
        "schedule_minutes": 15,
        "action": "slack",
        "action_config": {"webhook_url": slack_webhook} if slack_webhook else {}
    }
    
    return fallback_plan

@app.function(image=image)
def fetch_db_data(pg_url: str, sql: str):
    """Fetch data from Supabase and normalize columns."""
    import pandas as pd
    from sqlalchemy import create_engine, text
    
    try:
        engine = create_engine(pg_url)
        df = pd.read_sql(text(sql), engine)
        
        # Basic data cleaning
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'ts' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col])
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna()
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise

@app.function(image=image)
def inspect_database(pg_url: str, sql: str) -> Dict[str, Any]:
    """Inspect database schema and data for LLM planning."""
    import pandas as pd
    
    try:
        df = fetch_db_data.call(pg_url, sql)
        
        # Analyze columns
        columns = []
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
                "is_datetime": pd.api.types.is_datetime64_any_dtype(df[col]) or 'ts' in col.lower() or 'time' in col.lower(),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique())
            }
            columns.append(col_info)
        
        # Get numeric statistics
        numeric_stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats = df[col].describe()
            numeric_stats[col] = {
                "mean": float(stats['mean']),
                "std": float(stats['std']),
                "min": float(stats['min']),
                "max": float(stats['max'])
            }
        
        # Sample data for context
        sample = df.head(10).to_dict('records')
        for row in sample:
            for key, value in row.items():
                if pd.api.types.is_datetime64_any_dtype(type(value)):
                    row[key] = str(value)
                elif isinstance(value, (np.integer, np.floating)):
                    if np.isnan(value):
                        row[key] = None
                    else:
                        row[key] = float(value)
        
        return {
            "columns": columns,
            "numeric_stats": numeric_stats,
            "sample": sample,
            "total_rows": len(df)
        }
        
    except Exception as e:
        print(f"Error inspecting database: {e}")
        raise

@app.function(image=image)
def detect_anomalies(df, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Detect anomalies using the generated plan."""
    import pandas as pd
    import numpy as np
    
    try:
        # Extract plan parameters
        metric_col = plan["metric"]
        timestamp_col = plan["timestamp_col"]
        method = plan.get("method", "zscore")
        threshold = plan.get("threshold", 3.0)
        ew_span = plan.get("ew_span", 24)
        
        # Prepare data
        if metric_col not in df.columns or timestamp_col not in df.columns:
            raise ValueError(f"Columns {metric_col} or {timestamp_col} not found in data")
        
        work_df = df[[timestamp_col, metric_col]].copy()
        work_df.columns = ['ts', 'value']
        work_df = work_df.sort_values('ts')
        
        if len(work_df) < 2:
            return {"count": 0, "latest": None, "series": []}
        
        # Calculate exponentially weighted mean and std
        ewm_mean = work_df['value'].ewm(span=ew_span).mean()
        ewm_std = work_df['value'].ewm(span=ew_span).std()
        
        # Calculate z-scores
        z_scores = (work_df['value'] - ewm_mean) / ewm_std
        anomalies = np.abs(z_scores) > threshold
        
        # Prepare results
        series_data = []
        for i, (_, row) in enumerate(work_df.iterrows()):
            point = {
                "ts": row['ts'].isoformat(),
                "value": float(row['value']),
                "ewm_mean": float(ewm_mean.iloc[i]) if not pd.isna(ewm_mean.iloc[i]) else None,
                "z_score": float(z_scores.iloc[i]) if not pd.isna(z_scores.iloc[i]) else None,
                "is_anomaly": bool(anomalies.iloc[i]) if not pd.isna(anomalies.iloc[i]) else False
            }
            series_data.append(point)
        
        latest = series_data[-1] if series_data else None
        
        return {
            "count": int(anomalies.sum()),
            "latest": latest,
            "series": series_data,
            "plan_used": plan
        }
        
    except Exception as e:
        print(f"Error detecting anomalies: {e}")
        raise

@app.function(image=image)
def create_alert_chart(anomaly_data: Dict[str, Any]) -> str:
    """Create matplotlib chart and return as base64 data URI."""
    try:
        series = anomaly_data.get('series', [])
        if not series:
            return ""
        
        # Extract data
        timestamps = [pd.to_datetime(point['ts']) for point in series]
        values = [point['value'] for point in series]
        ewm_means = [point.get('ewm_mean') for point in series if point.get('ewm_mean') is not None]
        anomalies = [(pd.to_datetime(point['ts']), point['value']) for point in series if point.get('is_anomaly')]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, values, 'b-', label='Actual Values', alpha=0.7)
        
        if ewm_means and len(ewm_means) == len(timestamps):
            plt.plot(timestamps, ewm_means, 'g--', label='EWM Mean', alpha=0.8)
        
        if anomalies:
            anomaly_times, anomaly_values = zip(*anomalies)
            plt.scatter(anomaly_times, anomaly_values, color='red', s=50, label='Anomalies', zorder=5)
        
        plt.title('Database Monitoring: Anomaly Detection')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        print(f"Error creating chart: {e}")
        return ""

@app.function(image=image)
def send_slack_alert(webhook_url: str, summary: str, chart_uri: Optional[str] = None) -> bool:
    """Send alert to Slack with optional chart."""
    try:
        blocks = [{
            "type": "section",
            "text": {"type": "mrkdwn", "text": summary}
        }]
        
        if chart_uri:
            blocks.append({
                "type": "image",
                "image_url": chart_uri,
                "alt_text": "Anomaly Detection Chart"
            })
        
        response = requests.post(webhook_url, json={"blocks": blocks})
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error sending Slack alert: {e}")
        return False

@app.function(image=image)
def run_monitoring_pipeline(pg_url: str, base_sql: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete monitoring pipeline: fetch → detect → alert."""
    try:
        # Fetch data
        df = fetch_db_data.call(pg_url, base_sql)
        
        # Detect anomalies
        anomaly_result = detect_anomalies.call(df, plan)
        
        # Create chart
        chart_uri = create_alert_chart.call(anomaly_result)
        
        # Prepare summary
        count = anomaly_result.get("count", 0)
        latest = anomaly_result.get("latest", {})
        
        if count > 0:
            summary = f"🚨 *Anomaly Alert!*\nFound {count} anomalies.\n"
            if latest:
                summary += f"Latest: {latest.get('value', 'N/A')} (z-score: {latest.get('z_score', 0):.2f})"
        else:
            summary = f"✅ *All Clear*\nNo anomalies detected.\n"
            if latest:
                summary += f"Latest: {latest.get('value', 'N/A')}"
        
        # Send Slack alert if configured
        slack_sent = False
        if plan.get("action") == "slack":
            webhook_url = plan.get("action_config", {}).get("webhook_url")
            if webhook_url:
                slack_sent = send_slack_alert.call(webhook_url, summary, chart_uri)
        
        return {
            "anomalies": anomaly_result,
            "slack_sent": slack_sent,
            "chart_uri": chart_uri,
            "summary": summary,
            "plan_used": plan
        }
        
    except Exception as e:
        print(f"Error in monitoring pipeline: {e}")
        raise

# Local entrypoint for testing
@app.function(image=llama_image, volumes={cache_dir: model_cache})
def list_available_models():
    """List all available models with their descriptions."""
    print("🤖 Available Models for AI Planning:\n")
    for name, config in AVAILABLE_MODELS.items():
        print(f"• {name}: {config['description']}")
        print(f"  └─ Size: ~{config['filename'].split('-')[-1].replace('.gguf', '').upper()}")
    print("\nTo use a model: modal run modal_app.py::test_model --model='model-name'")

@app.local_entrypoint()
def test_model(model: str = "phi-4"):
    """Test a specific model by downloading and running a simple prompt."""
    if model not in AVAILABLE_MODELS:
        print(f"❌ Model '{model}' not available. Use list_models to see options.")
        return
    
    print(f"🤖 Testing {AVAILABLE_MODELS[model]['description']}...")
    
    # Test schema info
    test_schema = {
        "columns": [
            {"name": "ts", "dtype": "datetime64[ns]", "is_datetime": True, "is_numeric": False},
            {"name": "cpu_usage", "dtype": "float64", "is_datetime": False, "is_numeric": True},
            {"name": "memory_usage", "dtype": "float64", "is_datetime": False, "is_numeric": True}
        ],
        "sample": [
            {"ts": "2024-01-01 10:00:00", "cpu_usage": 45.2, "memory_usage": 67.8},
            {"ts": "2024-01-01 10:05:00", "cpu_usage": 52.1, "memory_usage": 71.2},
            {"ts": "2024-01-01 10:10:00", "cpu_usage": 48.7, "memory_usage": 69.1}
        ]
    }
    
    result = plan_with_llm.call(test_schema, "monitor CPU spikes over 80%", None, model)
    print(f"✅ {model} generated plan successfully:")
    print(f"   Monitoring: {result['metric']}")
    print(f"   Threshold: {result['threshold']}")
    print(f"   Method: {result['method']}")

@app.local_entrypoint()
def test_pipeline(model: str = "phi-4"):
    """Test the pipeline locally with environment variables."""
    from dotenv import load_dotenv
    load_dotenv()
    
    pg_url = os.getenv("POSTGRES_URL")
    base_sql = os.getenv("BASE_SQL")
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    
    if not pg_url or not base_sql:
        print("Set POSTGRES_URL and BASE_SQL in .env file")
        return
    
    print(f"Testing pipeline with {AVAILABLE_MODELS.get(model, {}).get('description', model)}...")
    
    print("1. Inspecting database...")
    schema_info = inspect_database.call(pg_url, base_sql)
    print(f"Found {len(schema_info['columns'])} columns, {schema_info['total_rows']} rows")
    
    print("2. Generating monitoring plan...")
    plan = plan_with_llm.call(schema_info, "detect spikes and unusual patterns", slack_webhook, model)
    print(f"Plan: monitoring {plan['metric']} with threshold {plan['threshold']}")
    
    print("3. Running monitoring pipeline...")
    result = run_monitoring_pipeline.call(pg_url, base_sql, plan)
    print(f"Found {result['anomalies']['count']} anomalies")
    print(f"Slack sent: {result['slack_sent']}")

@app.local_entrypoint()
def list_models():
    """List all available models."""
    print("🤖 Available Models for AI Planning:\n")
    for name, config in AVAILABLE_MODELS.items():
        print(f"• {name}: {config['description']}")
        print(f"  └─ Size: ~{config['filename'].split('-')[-1].replace('.gguf', '').upper()}")
    print("\nTo use a model: modal run modal_app.py::test_model --model='model-name'")
    print("To test with your database: modal run modal_app.py::test_pipeline --model='model-name'")
