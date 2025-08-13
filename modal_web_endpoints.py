import modal
from typing import Dict, Any
import json

# Create Modal app
app = modal.App("sailo-web-api")

# Define the image with required dependencies
image = modal.Image.debian_slim().pip_install([
    "pandas",
    "numpy", 
    "requests",
    "supabase",
    "fastapi"  # For web endpoints
])

@app.function(image=image)
@modal.web_endpoint(method="POST")
def run_monitoring_scenario(item: Dict[str, Any]):
    """Web endpoint for running AI monitoring scenarios from React frontend"""
    import pandas as pd
    import numpy as np
    from supabase import create_client, Client
    import json
    
    try:
        # Extract parameters from request
        scenario_type = item.get('scenario', 'volatility_spike')
        custom_query = item.get('customQuery', '')
        slack_webhook = item.get('slackWebhook', '')
        
        # Your Supabase credentials
        supabase_url = "https://xcwdavnejsnkddaroaaf.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhjd2Rhdm5lanNua2RkYXJvYWFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwNTY1MDQsImV4cCI6MjA3MDYzMjUwNH0.2q_9k1D26H9EFh2OBdEqVnqnMAEGHcErEFz36n9TgVY"
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        print(f"🤖 Running AI monitoring scenario: {scenario_type}")
        
        # Fetch options data
        response = supabase.table('options_trades').select(
            'symbol, contract_type, strike_price, premium, implied_volatility, volume, delta, gamma, theta, vega, created_at'
        ).order('created_at', desc=True).execute()
        
        if not response.data:
            return {
                "success": False,
                "error": "No options data found in database",
                "suggestion": "Please run the SQL setup script to add sample data"
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(response.data)
        
        # Clean numeric columns
        numeric_cols = ['strike_price', 'premium', 'implied_volatility', 'volume', 'delta', 'gamma', 'theta', 'vega']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"📊 Analyzing {len(df)} options records for {scenario_type}...")
        
        # Run scenario-specific analysis
        results = {}
        
        if scenario_type == 'volatility_spike':
            results = analyze_volatility_spikes(df)
        elif scenario_type == 'unusual_volume':
            results = analyze_unusual_volume(df)
        elif scenario_type == 'premium_anomaly':
            results = analyze_premium_anomalies(df)
        elif scenario_type == 'custom' and custom_query:
            results = analyze_custom_scenario(df, custom_query)
        else:
            # Run comprehensive analysis
            results = run_comprehensive_analysis(df)
        
        # Send Slack alert if webhook provided and anomalies found
        slack_sent = False
        if slack_webhook and results.get('total_anomalies', 0) > 0:
            slack_sent = send_slack_alert(slack_webhook, results)
        
        return {
            "success": True,
            "scenario": scenario_type,
            "total_records": len(df),
            "results": results,
            "slack_sent": slack_sent,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error in monitoring scenario: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "scenario": scenario_type if 'scenario_type' in locals() else "unknown"
        }

def analyze_volatility_spikes(df):
    """Analyze implied volatility spikes"""
    import numpy as np
    import pandas as pd
    
    if 'implied_volatility' not in df.columns:
        return {"error": "No implied volatility data available"}
    
    iv_data = df['implied_volatility'].dropna()
    if len(iv_data) == 0:
        return {"error": "No valid implied volatility data"}
    
    iv_mean = iv_data.mean()
    iv_std = iv_data.std()
    
    if iv_std == 0:
        return {"message": "No volatility variation detected", "anomalies": []}
    
    # Z-score anomaly detection (lowered threshold for demo)
    df['iv_zscore'] = np.abs((df['implied_volatility'] - iv_mean) / iv_std)
    anomalies = df[df['iv_zscore'] > 1.0].copy()  # 1+ standard deviations (more sensitive)
    
    anomaly_list = []
    for _, row in anomalies.nlargest(5, 'iv_zscore').iterrows():
        anomaly_list.append({
            'symbol': row['symbol'],
            'contract_type': row['contract_type'],
            'strike_price': float(row['strike_price']) if pd.notna(row['strike_price']) else 0,
            'implied_volatility': float(row['implied_volatility']) if pd.notna(row['implied_volatility']) else 0,
            'z_score': float(row['iv_zscore']) if pd.notna(row['iv_zscore']) else 0,
            'severity': 'HIGH' if row['iv_zscore'] > 3.0 else 'MEDIUM'
        })
    
    return {
        "type": "volatility_spikes",
        "total_anomalies": len(anomalies),
        "threshold": 1.0,
        "mean_iv": float(iv_mean),
        "std_iv": float(iv_std),
        "anomalies": anomaly_list,
        "message": f"Found {len(anomalies)} volatility spike anomalies"
    }

def analyze_unusual_volume(df):
    """Analyze unusual trading volume"""
    import numpy as np
    import pandas as pd
    
    if 'volume' not in df.columns:
        return {"error": "No volume data available"}
    
    volume_data = df['volume'].dropna()
    if len(volume_data) == 0:
        return {"error": "No valid volume data"}
    
    volume_mean = volume_data.mean()
    volume_std = volume_data.std()
    
    if volume_std == 0:
        return {"message": "No volume variation detected", "anomalies": []}
    
    # Volume spike detection (more than 1.5x average - more sensitive)
    volume_threshold = volume_mean + (1.5 * volume_std)
    anomalies = df[df['volume'] > volume_threshold].copy()
    
    anomaly_list = []
    for _, row in anomalies.nlargest(5, 'volume').iterrows():
        anomaly_list.append({
            'symbol': row['symbol'],
            'contract_type': row['contract_type'],
            'volume': int(row['volume']) if pd.notna(row['volume']) else 0,
            'volume_ratio': float(row['volume'] / volume_mean) if volume_mean > 0 and pd.notna(row['volume']) else 0,
            'severity': 'CRITICAL' if row['volume'] > volume_threshold * 2 else 'HIGH'
        })
    
    return {
        "type": "unusual_volume",
        "total_anomalies": len(anomalies),
        "threshold": float(volume_threshold),
        "mean_volume": float(volume_mean),
        "anomalies": anomaly_list,
        "message": f"Found {len(anomalies)} unusual volume anomalies"
    }

def analyze_premium_anomalies(df):
    """Analyze premium-to-strike ratio anomalies"""
    import numpy as np
    import pandas as pd
    
    if 'premium' not in df.columns or 'strike_price' not in df.columns:
        return {"error": "No premium or strike price data available"}
    
    # Filter valid rows
    valid_rows = df[(df['premium'].notna()) & (df['strike_price'].notna()) & (df['strike_price'] > 0)].copy()
    
    if len(valid_rows) == 0:
        return {"error": "No valid premium/strike price data"}
    
    valid_rows['premium_ratio'] = valid_rows['premium'] / valid_rows['strike_price']
    pr_data = valid_rows['premium_ratio'].dropna()
    
    if len(pr_data) == 0:
        return {"error": "No valid premium ratio data"}
    
    pr_mean = pr_data.mean()
    pr_std = pr_data.std()
    
    if pr_std == 0:
        return {"message": "No premium ratio variation detected", "anomalies": []}
    
    valid_rows['pr_zscore'] = np.abs((valid_rows['premium_ratio'] - pr_mean) / pr_std)
    anomalies = valid_rows[valid_rows['pr_zscore'] > 1.0].copy()
    
    anomaly_list = []
    for _, row in anomalies.nlargest(5, 'pr_zscore').iterrows():
        anomaly_list.append({
            'symbol': row['symbol'],
            'contract_type': row['contract_type'],
            'premium': float(row['premium']) if pd.notna(row['premium']) else 0,
            'strike_price': float(row['strike_price']) if pd.notna(row['strike_price']) else 0,
            'premium_ratio': float(row['premium_ratio']) if pd.notna(row['premium_ratio']) else 0,
            'z_score': float(row['pr_zscore']) if pd.notna(row['pr_zscore']) else 0,
            'severity': 'HIGH' if row['pr_zscore'] > 2.5 else 'MEDIUM'
        })
    
    return {
        "type": "premium_anomalies",
        "total_anomalies": len(anomalies),
        "threshold": 1.0,
        "mean_premium_ratio": float(pr_mean),
        "anomalies": anomaly_list,
        "message": f"Found {len(anomalies)} premium anomalies"
    }

def analyze_custom_scenario(df, query):
    """Analyze custom user-defined scenario"""
    # For demo purposes, return a simple analysis
    # In production, this could use LLM to interpret the query
    return {
        "type": "custom",
        "query": query,
        "total_anomalies": 0,
        "message": f"Custom analysis for: {query}",
        "anomalies": []
    }

def run_comprehensive_analysis(df):
    """Run all analysis types"""
    vol_results = analyze_volatility_spikes(df)
    volume_results = analyze_unusual_volume(df)
    premium_results = analyze_premium_anomalies(df)
    
    total_anomalies = (
        vol_results.get('total_anomalies', 0) + 
        volume_results.get('total_anomalies', 0) + 
        premium_results.get('total_anomalies', 0)
    )
    
    return {
        "type": "comprehensive",
        "total_anomalies": total_anomalies,
        "volatility_spikes": vol_results,
        "unusual_volume": volume_results,
        "premium_anomalies": premium_results,
        "message": f"Comprehensive analysis found {total_anomalies} total anomalies"
    }

def send_slack_alert(webhook_url, results):
    """Send Slack alert with results"""
    import requests
    import pandas as pd
    
    try:
        message = f"🚨 *Sailo AI Alert*\n"
        message += f"📊 Analysis Type: {results.get('type', 'unknown')}\n"
        message += f"🔍 Total Anomalies: {results.get('total_anomalies', 0)}\n"
        message += f"⏰ Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if results.get('anomalies'):
            message += f"\n*Top Anomalies:*\n"
            for i, anomaly in enumerate(results['anomalies'][:3], 1):
                if 'implied_volatility' in anomaly:
                    message += f"{i}. {anomaly['symbol']} {anomaly['contract_type']} - IV: {anomaly['implied_volatility']:.4f} [{anomaly['severity']}]\n"
                elif 'volume' in anomaly:
                    message += f"{i}. {anomaly['symbol']} {anomaly['contract_type']} - Volume: {anomaly['volume']:,} [{anomaly['severity']}]\n"
        
        payload = {"text": message}
        response = requests.post(webhook_url, json=payload, timeout=10)
        return response.status_code == 200
        
    except Exception as e:
        print(f"Failed to send Slack alert: {e}")
        return False

@app.function(image=image)
@modal.web_endpoint(method="GET")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "sailo-ai-etl-pipeline"}

@app.function(image=image)
@modal.web_endpoint(method="GET")
def get_table_info():
    """Get information about the options_trades table"""
    from supabase import create_client, Client
    
    try:
        supabase_url = "https://xcwdavnejsnkddaroaaf.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhjd2Rhdm5lanNua2RkYXJvYWFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwNTY1MDQsImV4cCI6MjA3MDYzMjUwNH0.2q_9k1D26H9EFh2OBdEqVnqnMAEGHcErEFz36n9TgVY"
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Get record count
        count_response = supabase.table('options_trades').select('*', count='exact').execute()
        record_count = count_response.count
        
        # Get sample data
        sample_response = supabase.table('options_trades').select('*').limit(3).execute()
        sample_data = sample_response.data
        
        return {
            "success": True,
            "table_exists": True,
            "record_count": record_count,
            "sample_data": sample_data
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "table_exists": False
        }

# Deploy the app
if __name__ == "__main__":
    print("Deploying Sailo AI ETL Web API...")
    print("Web endpoints will be available at:")
    print("- POST /run_monitoring_scenario")
    print("- GET /health_check") 
    print("- GET /get_table_info")
