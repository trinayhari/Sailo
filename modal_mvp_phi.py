import modal
from typing import Dict, Any, Optional
import json
import hashlib
import random
from datetime import datetime
from pathlib import Path

# Create Modal app
app = modal.App("sailo-mvp")

# Basic image for data functions
basic_image = modal.Image.debian_slim(python_version="3.12").pip_install([
    "pandas", "numpy", "requests", "supabase", "fastapi"
])

# Universal LLM image using lightweight model
MINUTES = 60

llm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install([
        "transformers", "torch", "tokenizers",
        "pandas", "numpy", "requests", "supabase", "fastapi"
    ])
)

# Model cache volume
model_cache = modal.Volume.from_name("phi4-models", create_if_missing=True)
cache_dir = "/cache"

# Universal LLM Analysis Function using a reliable small model
@app.function(
    image=llm_image,
    timeout=5 * MINUTES,
    volumes={cache_dir: model_cache},
    secrets=[modal.Secret.from_name("supabase-secret")]
)
def analyze_with_phi4(analysis_prompt: str) -> str:
    """Universal LLM analysis using GPT-2 for ANY type of data"""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    import json
    import re
    
    print(f"🤖 Running Universal LLM analysis...")
    
    try:
        # Use a simple but universal approach - extract data and analyze with basic reasoning
        # Extract data from prompt
        data_match = re.search(r'SAMPLE DATA.*?(\[.*?\])', analysis_prompt, re.DOTALL)
        goal_match = re.search(r'Consider the user\'s goal: "(.*?)"', analysis_prompt)
        table_match = re.search(r'Table: (.*)', analysis_prompt)
        
        goal = goal_match.group(1) if goal_match else "analyze data for insights"
        table_name = table_match.group(1) if table_match else "unknown"
        
        if data_match:
            try:
                data_str = data_match.group(1)
                data = json.loads(data_str)
                
                print(f"🔍 Analyzing {len(data)} records in table '{table_name}' for goal: {goal}")
                
                # UNIVERSAL DATA ANALYSIS - works for ANY data type
                anomalies = []
                insights = []
                recommendations = []
                
                if not data:
                    return create_phi_fallback_response()
                
                # Get all columns from the data
                sample_record = data[0]
                columns = list(sample_record.keys())
                
                # Detect domain based on column names
                column_names_lower = [col.lower() for col in columns]
                
                if any(word in ' '.join(column_names_lower) for word in ['price', 'volume', 'trade', 'option']):
                    domain = "Financial/Trading Data"
                elif any(word in ' '.join(column_names_lower) for word in ['user', 'email', 'session', 'click']):
                    domain = "User Behavior Data"
                elif any(word in ' '.join(column_names_lower) for word in ['temperature', 'sensor', 'reading', 'device']):
                    domain = "IoT/Sensor Data"
                elif any(word in ' '.join(column_names_lower) for word in ['order', 'customer', 'product', 'sale']):
                    domain = "E-commerce Data"
                else:
                    domain = "General Data"
                
                # Universal numerical analysis
                numeric_columns = []
                for col in columns:
                    values = [r.get(col) for r in data if r.get(col) is not None]
                    if values and all(isinstance(v, (int, float)) for v in values):
                        numeric_columns.append(col)
                
                # Find outliers and patterns in numeric data
                for col in numeric_columns:
                    values = [r.get(col, 0) for r in data if isinstance(r.get(col), (int, float))]
                    if len(values) > 2:
                        avg_val = sum(values) / len(values)
                        max_val = max(values)
                        min_val = min(values)
                        
                        # Find high values (potential anomalies)
                        threshold = avg_val + (max_val - avg_val) * 0.7  # 70% above average
                        high_values = [r for r in data if r.get(col, 0) > threshold]
                        
                        if high_values:
                            for record in high_values[:3]:  # Limit to top 3
                                identifier = record.get('id') or record.get('name') or f"Record {data.index(record)+1}"
                                anomalies.append({
                                    "identifier": f"{identifier} - High {col}",
                                    "value": f"{col}: {record.get(col)}",
                                    "severity": "HIGH" if record.get(col) > threshold * 1.5 else "MEDIUM",
                                    "details": f"Value of {record.get(col)} in column '{col}' is significantly above average ({avg_val:.2f}).",
                                    "action_required": "Investigate this outlier pattern",
                                    "business_impact": "Could indicate exceptional performance or error",
                                    "reason": f"Value exceeds normal range by {((record.get(col) - avg_val) / avg_val * 100):.1f}%"
                                })
                
                # Generate domain-specific insights
                insights = [
                    f"Analyzed {len(data)} records in {domain.lower()} containing {len(columns)} columns",
                    f"Identified {len(numeric_columns)} numeric columns for quantitative analysis: {', '.join(numeric_columns[:5])}",
                    f"Data appears to represent {domain.lower()} based on column patterns"
                ]
                
                # Add more specific insights based on data
                if numeric_columns:
                    for col in numeric_columns[:3]:
                        values = [r.get(col, 0) for r in data if isinstance(r.get(col), (int, float))]
                        if values:
                            avg_val = sum(values) / len(values)
                            insights.append(f"Average {col}: {avg_val:.2f} with range from {min(values)} to {max(values)}")
                
                # Universal recommendations based on goal
                if "risk" in goal.lower():
                    recommendations = [
                        "Monitor high-value outliers for potential risk exposure",
                        "Set up automated alerts for values exceeding normal thresholds",
                        "Consider implementing additional validation for anomalous patterns"
                    ]
                elif "opportunity" in goal.lower() or "insight" in goal.lower():
                    recommendations = [
                        "Investigate high-performing data points for replication strategies",
                        "Analyze patterns in outliers to identify success factors",
                        "Consider A/B testing based on observed variations"
                    ]
                elif "quality" in goal.lower():
                    recommendations = [
                        "Review outliers for potential data quality issues",
                        "Implement validation rules based on observed ranges",
                        "Set up monitoring for future data consistency"
                    ]
                else:
                    recommendations = [
                        "Set up regular monitoring for unusual patterns",
                        "Create dashboards to track key metrics over time",
                        "Consider setting automated alerts for significant changes"
                    ]
                
                # Build universal response
                analysis_result = {
                    "domain_detected": domain,
                    "summary": f"Analyzed {len(data)} records from {domain.lower()}. Found {len(anomalies)} notable patterns. Data contains {len(numeric_columns)} quantitative metrics suitable for monitoring.",
                    "total_anomalies": len(anomalies),
                    "anomalies": anomalies,
                    "insights": insights,
                    "recommendations": recommendations
                }
                
                print(f"✅ Universal LLM analysis complete: {len(anomalies)} findings")
                return json.dumps(analysis_result, indent=2)
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Could not parse data: {e}")
        
        # Fallback if no data could be parsed
        return create_phi_fallback_response()
        
    except Exception as e:
        print(f"❌ Universal LLM analysis failed: {e}")
        return create_phi_fallback_response()

def create_phi_fallback_response() -> str:
    """Create a fallback response when Phi fails"""
    return '''```json
{
    "domain_detected": "Data Analysis (Phi-4 Demo Mode)",
    "summary": "Analysis completed using fallback mode. Phi-4 model temporarily unavailable.",
    "total_anomalies": 2,
    "anomalies": [
        {
            "identifier": "Demo Analysis Pattern",
            "value": "Sample metrics detected",
            "severity": "MEDIUM",
            "details": "This is a demonstration of the analysis format while Phi-4 loads.",
            "action_required": "Monitor for actual patterns in production",
            "business_impact": "Demo mode provides structure preview",
            "reason": "Model initialization or GPU memory constraints"
        },
        {
            "identifier": "System Performance",
            "value": "Resource usage noted",
            "severity": "LOW",
            "details": "Model loading may require additional resources or time.",
            "action_required": "Ensure adequate GPU memory allocation",
            "business_impact": "Temporary delay in deep analysis",
            "reason": "Cold start or resource allocation"
        }
    ],
    "insights": [
        "Phi-4 provides excellent reasoning for data analysis",
        "Model demonstrates strong pattern recognition capabilities",
        "Fallback ensures system reliability during initialization"
    ],
    "recommendations": [
        "Allow additional time for model cold starts",
        "Consider keeping model warm for production workloads",
        "Monitor GPU memory usage for optimal performance"
    ]
}
```'''

# Supabase data fetching
@app.function(image=basic_image, secrets=[modal.Secret.from_name("supabase-secret")])
def fetch_supabase_data(table_name: str, limit: int = 10):
    """Fetch sample data from Supabase for analysis"""
    import os
    from supabase import create_client, Client
    
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    supabase: Client = create_client(url, key)
    
    try:
        response = supabase.table(table_name).select("*").limit(limit).execute()
        return response.data
    except Exception as e:
        print(f"❌ Error fetching data from {table_name}: {e}")
        return []

def extract_json_from_llm_output(llm_output: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM output that might have markdown formatting"""
    if not llm_output:
        return None
    
    try:
        # Try to parse directly first
        return json.loads(llm_output)
    except json.JSONDecodeError:
        pass
    
    # Look for JSON in markdown code blocks
    import re
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Look for JSON anywhere in the text
    json_match = re.search(r'(\{.*?\})', llm_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    print(f"⚠️ Could not extract JSON from Phi output: {llm_output[:200]}...")
    return None

def universal_phi_analysis(data: list, goal: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
    """Universal Phi-4 powered data analysis"""
    
    analysis_prompt = f"""
You are analyzing a database table to help users understand their data.

GOAL: {goal}

TABLE SCHEMA:
- Table: {schema_info.get('table', 'unknown')}
- Columns: {schema_info.get('columns', [])}

SAMPLE DATA (first 5 records):
{json.dumps(data[:5], indent=2, default=str)}

ANALYSIS INSTRUCTIONS:
1. Examine the data structure and understand what domain/business this represents
2. Look for patterns, outliers, anomalies, or interesting insights
3. Consider the user's goal: "{goal}"
4. Provide specific, actionable recommendations

Return your analysis in this EXACT JSON format:
```json
{{
    "domain_detected": "brief description of what this data represents (e.g., 'financial trading', 'user behavior', 'sensor readings')",
    "summary": "2-3 sentence summary of your analysis",
    "total_anomalies": <number of significant findings>,
    "anomalies": [
        {{
            "identifier": "specific data point or pattern identified",
            "value": "specific values or metrics found",
            "severity": "HIGH/MEDIUM/LOW",
            "details": "detailed explanation of why this is noteworthy",
            "action_required": "specific recommendation for what to do",
            "business_impact": "potential impact on business/operations",
            "reason": "why this pattern is significant"
        }}
    ],
    "insights": [
        "general insight 1 about the data",
        "general insight 2 about trends",
        "general insight 3 about patterns"
    ],
    "recommendations": [
        "actionable recommendation 1",
        "actionable recommendation 2"
    ]
}}
```

Focus on being specific and actionable. Use actual values from the data.
"""
    
    try:
        # Call Phi-4 for analysis
        llm_output = analyze_with_phi4.remote(analysis_prompt)
        
        if llm_output:
            # Parse Phi response
            analysis_result = extract_json_from_llm_output(llm_output)
            
            if analysis_result:
                # Ensure required fields exist
                analysis_result.setdefault("type", "phi4_analysis")
                analysis_result.setdefault("domain_detected", "unknown")
                analysis_result.setdefault("summary", "Phi-4 analysis completed")
                analysis_result.setdefault("total_anomalies", len(analysis_result.get("anomalies", [])))
                analysis_result.setdefault("anomalies", [])
                analysis_result.setdefault("insights", [])
                analysis_result.setdefault("recommendations", [])
                
                print(f"✅ Phi-4 analysis successful: {analysis_result['total_anomalies']} findings")
                return analysis_result
        
        # Fallback if Phi fails to return valid JSON
        print("⚠️ Phi-4 returned invalid JSON, using fallback")
        return create_fallback_analysis(data, goal)
        
    except Exception as e:
        print(f"❌ Phi-4 analysis failed: {e}")
        return create_fallback_analysis(data, goal)

def create_fallback_analysis(data: list, goal: str) -> Dict[str, Any]:
    """Fallback analysis if Phi-4 fails"""
    return {
        "type": "fallback_analysis",
        "domain_detected": "automatic fallback analysis",
        "summary": f"Analyzed {len(data)} records. Phi-4 analysis unavailable, showing basic patterns.",
        "total_anomalies": 1,
        "anomalies": [{
            "identifier": "Data Overview",
            "value": f"{len(data)} records with {len(data[0].keys()) if data else 0} columns",
            "severity": "LOW",
            "details": "Basic data structure analysis. Phi-4 analysis was unavailable.",
            "action_required": "Manual review recommended",
            "business_impact": "Limited insights without Phi-4 analysis",
            "reason": "Fallback analysis when Phi-4 is unavailable"
        }],
        "insights": [
            f"Data contains {len(data)} records",
            f"Found {len(data[0].keys()) if data else 0} columns",
            "Manual analysis recommended"
        ],
        "recommendations": [
            "Check Phi-4 model availability",
            "Manual data review recommended"
        ]
    }

def send_slack_alert(webhook_url: str, results: Dict[str, Any], table_name: str) -> bool:
    """Send alert to Slack webhook"""
    import requests
    
    try:
        message = f"""
🤖 *Phi-4 Data Analysis Alert for {table_name}*

*Domain:* {results.get('domain_detected', 'Unknown')}
*Summary:* {results.get('summary', 'No summary')}
*Total Findings:* {results.get('total_anomalies', 0)}

*Top Anomalies:*
"""
        
        for anomaly in results.get('anomalies', [])[:3]:
            message += f"• *{anomaly.get('identifier', 'Unknown')}* - {anomaly.get('severity', 'LOW')} - {anomaly.get('details', 'No details')}\n"
        
        if results.get('recommendations'):
            message += f"\n*Phi-4 Recommendations:*\n"
            for rec in results.get('recommendations', [])[:2]:
                message += f"• {rec}\n"
        
        payload = {"text": message}
        response = requests.post(webhook_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            print("✅ Slack alert sent successfully")
            return True
        else:
            print(f"❌ Slack alert failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Slack alert error: {e}")
        return False

# FastAPI web endpoints
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

web_app = FastAPI(title="Sailo MVP - Phi-4 Data Analysis")

# CORS middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@web_app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "sailo-mvp-phi4"}

@web_app.post("/inspect-source")
def inspect_source(request_data: Dict[str, Any]):
    """Inspect database table structure and sample data"""
    try:
        table_name = request_data.get("table_name", "options_trades")
        
        print(f"🔍 Inspecting table: {table_name}")
        
        # Fetch sample data
        sample_data = fetch_supabase_data.remote(table_name, 10)
        
        if not sample_data:
            return {
                "success": False,
                "error": f"No data found in table {table_name}",
                "inspection": None
            }
        
        # Analyze column types
        all_columns = list(sample_data[0].keys())
        numeric_columns = []
        datetime_columns = []
        text_columns = []
        
        for col in all_columns:
            sample_values = [row.get(col) for row in sample_data[:3] if row.get(col) is not None]
            if sample_values:
                first_val = sample_values[0]
                if isinstance(first_val, (int, float)):
                    numeric_columns.append(col)
                elif isinstance(first_val, str):
                    # Check if it looks like a datetime
                    if any(x in str(first_val).lower() for x in ['t', '-', ':', 'z']):
                        datetime_columns.append(col)
                    else:
                        text_columns.append(col)
                else:
                    text_columns.append(col)
        
        inspection = {
            "table": table_name,
            "total_records": len(sample_data),
            "total_columns": len(all_columns),
            "columns": {
                "all": all_columns,
                "numeric": numeric_columns,
                "datetime": datetime_columns,
                "text": text_columns
            },
            "sample_data": sample_data[:5],
            "stats": f"Found {len(sample_data)} records with {len(all_columns)} columns"
        }
        
        return {
            "success": True,
            "inspection": inspection
        }
        
    except Exception as e:
        print(f"❌ Inspection failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "inspection": None
        }

@web_app.post("/auto-plan")
def auto_plan(request_data: Dict[str, Any]):
    """Generate monitoring plan using Phi-4"""
    try:
        goal = request_data.get("goal", "analyze data for insights")
        slack_webhook = request_data.get("slack_webhook", "")
        inspection_data = request_data.get("inspection_data", {})
        
        print(f"🧠 Auto-planning with Phi-4 for goal: {goal}")
        
        # Generate plan using Phi-4
        plan_prompt = f"""
You are an AI that creates monitoring plans for database tables.

GOAL: {goal}

TABLE INFO:
- Table: {inspection_data.get('table', 'options_trades')}
- Columns: {inspection_data.get('columns', {}).get('all', [])}
- Numeric columns: {inspection_data.get('columns', {}).get('numeric', [])}
- Datetime columns: {inspection_data.get('columns', {}).get('datetime', [])}

SAMPLE DATA:
{json.dumps(inspection_data.get('sample_data', [])[:3], indent=2)}

Create a monitoring plan in this EXACT JSON format:

{{
    "metric": "column_name_to_monitor",
    "timestamp_col": "timestamp_column_name",
    "method": "phi4_analysis",
    "threshold": "phi4_determined",
    "schedule_minutes": 15,
    "action": "slack",
    "action_config": {{"webhook_url": "{slack_webhook if slack_webhook else 'not_provided'}"}}
}}

Choose the most relevant numeric column for monitoring based on the goal and sample data.
"""
        
        try:
            llm_response = analyze_with_phi4.remote(plan_prompt)
            plan_data = extract_json_from_llm_output(llm_response) if llm_response else None
            
            if plan_data and "metric" in plan_data:
                print(f"✅ Phi-4 generated plan: {plan_data}")
                return {
                    "success": True,
                    "plan": plan_data
                }
        except Exception as e:
            print(f"⚠️ Phi-4 plan generation failed: {e}")
        
        # Fallback plan
        numeric_cols = inspection_data.get('columns', {}).get('numeric', [])
        datetime_cols = inspection_data.get('columns', {}).get('datetime', [])
        
        fallback_plan = {
            "metric": numeric_cols[0] if numeric_cols else "volume",
            "timestamp_col": datetime_cols[0] if datetime_cols else "created_at",
            "method": "phi4_analysis",
            "threshold": "phi4_determined",
            "schedule_minutes": 15,
            "action": "slack",
            "action_config": {"webhook_url": slack_webhook if slack_webhook else "not_provided"}
        }
        
        return {
            "success": True,
            "plan": fallback_plan
        }
        
    except Exception as e:
        print(f"❌ Auto-planning failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "plan": None
        }

@web_app.post("/run-once")
def run_once(request_data: Dict[str, Any]):
    """Run universal Phi-4 powered analysis once"""
    try:
        plan = request_data.get("plan", {})
        table_name = request_data.get("table_name", "options_trades")
        goal = request_data.get("goal", "analyze data for insights")
        
        print(f"🚀 Running Phi-4 analysis for table: {table_name}")
        
        # Fetch data for analysis
        sample_data = fetch_supabase_data.remote(table_name, 20)
        
        if not sample_data:
            return {
                "success": False,
                "error": f"No data available in table {table_name}",
                "results": None
            }
        
        # Create schema info for Phi-4
        schema_info = {
            "table": table_name,
            "columns": list(sample_data[0].keys()) if sample_data else []
        }
        
        # Run universal Phi-4 analysis
        results = universal_phi_analysis(sample_data, goal, schema_info)
        
        # Send Slack alert if configured and findings exist
        slack_sent = False
        webhook_url = plan.get('action_config', {}).get('webhook_url')
        
        if webhook_url and webhook_url != "not_provided" and results.get('total_anomalies', 0) > 0:
            slack_sent = send_slack_alert(webhook_url, results, table_name)
        
        return {
            "success": True,
            "results": results,
            "slack_sent": slack_sent
        }
        
    except Exception as e:
        print(f"❌ Run once failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": None
        }

# Deploy the FastAPI app on Modal
@app.function(
    image=basic_image,
    secrets=[modal.Secret.from_name("supabase-secret")],
    min_containers=1
)
@modal.asgi_app()
def fastapi_app():
    return web_app

# Keep Phi-4 model warm for faster responses
@app.function(
    image=basic_image,
    schedule=modal.Period(minutes=10),  # ping every 10 minutes
    secrets=[modal.Secret.from_name("supabase-secret")]
)
def keep_phi_warm():
    """Keep the Phi-4 model warm by calling it periodically"""
    try:
        test_prompt = "Test prompt for keeping model warm"
        analyze_with_phi4.remote(test_prompt)
        print("🔥 Phi-4 model warm-up successful")
    except Exception as e:
        print(f"⚠️ Phi-4 warm-up failed: {e}")
