import os
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from typing import Literal
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
    "openai",
    "python-dotenv"
])

# Pydantic schema for monitoring plans
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

@app.function(image=image)
def test_supabase_connection(supabase_url: str, supabase_key: str) -> Dict[str, Any]:
    """Test connection to Supabase and list available tables"""
    import pandas as pd
    from sqlalchemy import create_engine, text
    
    try:
        # Convert Supabase URL to PostgreSQL connection string
        project_ref = supabase_url.replace('https://', '').replace('.supabase.co', '')
        
        # For testing, we'll use the anon key approach (limited permissions)
        # In production, you'd use the service role key or database password
        pg_url = f"postgresql://postgres.{project_ref}:{supabase_key}@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
        
        engine = create_engine(pg_url)
        
        # Test connection by listing tables
        tables_query = """
        SELECT table_name, column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
        ORDER BY table_name, ordinal_position
        """
        
        df = pd.read_sql(text(tables_query), engine)
        
        # Group by table
        tables_info = {}
        for _, row in df.iterrows():
            table_name = row['table_name']
            if table_name not in tables_info:
                tables_info[table_name] = []
            tables_info[table_name].append({
                'column': row['column_name'],
                'type': row['data_type']
            })
        
        return {
            "status": "success",
            "tables_found": list(tables_info.keys()),
            "tables_info": tables_info,
            "connection_url": pg_url.replace(supabase_key, "***")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "connection_url": pg_url.replace(supabase_key, "***") if 'pg_url' in locals() else "N/A"
        }

@app.function(image=image)
def fetch_options_data(supabase_url: str, supabase_key: str, limit: int = 10) -> Dict[str, Any]:
    """Fetch sample options trading data from Supabase"""
    import pandas as pd
    from sqlalchemy import create_engine, text
    
    try:
        project_ref = supabase_url.replace('https://', '').replace('.supabase.co', '')
        pg_url = f"postgresql://postgres.{project_ref}:{supabase_key}@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
        
        engine = create_engine(pg_url)
        
        # Fetch options trading data
        query = f"""
        SELECT 
            id,
            symbol,
            contract_type,
            strike_price,
            premium,
            implied_volatility,
            volume,
            trade_timestamp,
            created_at
        FROM options_trades 
        ORDER BY created_at DESC 
        LIMIT {limit}
        """
        
        df = pd.read_sql(text(query), engine)
        
        # Convert to JSON-serializable format
        data = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    record[col] = None
                elif isinstance(value, pd.Timestamp):
                    record[col] = value.isoformat()
                else:
                    record[col] = value
            data.append(record)
        
        return {
            "status": "success",
            "count": len(data),
            "data": data,
            "columns": list(df.columns)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": []
        }

@app.function(image=image)
def simple_anomaly_detection(supabase_url: str, supabase_key: str, metric: str = "implied_volatility") -> Dict[str, Any]:
    """Simple anomaly detection on options trading data"""
    import pandas as pd
    import numpy as np
    from sqlalchemy import create_engine, text
    
    try:
        project_ref = supabase_url.replace('https://', '').replace('.supabase.co', '')
        pg_url = f"postgresql://postgres.{project_ref}:{supabase_key}@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
        
        engine = create_engine(pg_url)
        
        # Fetch recent data for analysis
        query = f"""
        SELECT 
            trade_timestamp,
            symbol,
            {metric},
            volume,
            premium
        FROM options_trades 
        WHERE {metric} IS NOT NULL
        AND trade_timestamp >= NOW() - INTERVAL '7 days'
        ORDER BY trade_timestamp DESC
        """
        
        df = pd.read_sql(text(query), engine)
        
        if df.empty:
            return {
                "status": "no_data",
                "message": "No data found for analysis",
                "anomalies": []
            }
        
        # Simple z-score anomaly detection
        df[f'{metric}_zscore'] = np.abs((df[metric] - df[metric].mean()) / df[metric].std())
        
        # Find anomalies (z-score > 2)
        anomalies = df[df[f'{metric}_zscore'] > 2.0].copy()
        
        # Convert to JSON-serializable format
        anomaly_records = []
        for _, row in anomalies.iterrows():
            record = {
                'timestamp': row['trade_timestamp'].isoformat() if pd.notna(row['trade_timestamp']) else None,
                'symbol': row['symbol'],
                'metric_value': float(row[metric]) if pd.notna(row[metric]) else None,
                'z_score': float(row[f'{metric}_zscore']) if pd.notna(row[f'{metric}_zscore']) else None,
                'volume': int(row['volume']) if pd.notna(row['volume']) else None,
                'premium': float(row['premium']) if pd.notna(row['premium']) else None
            }
            anomaly_records.append(record)
        
        # Summary statistics
        stats = {
            'total_records': len(df),
            'anomalies_found': len(anomalies),
            'metric_mean': float(df[metric].mean()),
            'metric_std': float(df[metric].std()),
            'threshold_used': 2.0
        }
        
        return {
            "status": "success",
            "metric_analyzed": metric,
            "statistics": stats,
            "anomalies": anomaly_records[:10],  # Return top 10 anomalies
            "summary": f"Found {len(anomalies)} anomalies in {metric} out of {len(df)} records"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "anomalies": []
        }

# Test functions that can be run locally
@app.local_entrypoint()
def test_connection():
    """Test Supabase connection"""
    supabase_url = "https://xcwdavnejsnkddaroaaf.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhjd2Rhdm5lanNua2RkYXJvYWFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwNTY1MDQsImV4cCI6MjA3MDYzMjUwNH0.2q_9k1D26H9EFh2OBdEqVnqnMAEGHcErEFz36n9TgVY"
    
    print("🔍 Testing Supabase connection...")
    result = test_supabase_connection.remote(supabase_url, supabase_key)
    
    if result["status"] == "success":
        print(f"✅ Connection successful!")
        print(f"📊 Tables found: {', '.join(result['tables_found'])}")
        
        if 'options_trades' in result['tables_found']:
            print("\n🎯 options_trades table found! Testing data fetch...")
            data_result = fetch_options_data.remote(supabase_url, supabase_key, 5)
            
            if data_result["status"] == "success":
                print(f"✅ Fetched {data_result['count']} records")
                print(f"📈 Columns: {', '.join(data_result['columns'])}")
                
                print("\n🤖 Running anomaly detection...")
                anomaly_result = simple_anomaly_detection.remote(supabase_url, supabase_key)
                
                if anomaly_result["status"] == "success":
                    print(f"✅ {anomaly_result['summary']}")
                    if anomaly_result['anomalies']:
                        print("\n🚨 Top anomalies found:")
                        for i, anomaly in enumerate(anomaly_result['anomalies'][:3], 1):
                            print(f"  {i}. {anomaly['symbol']}: {anomaly['metric_value']:.4f} (z-score: {anomaly['z_score']:.2f})")
                else:
                    print(f"❌ Anomaly detection failed: {anomaly_result['error']}")
            else:
                print(f"❌ Data fetch failed: {data_result['error']}")
        else:
            print("⚠️  options_trades table not found. Please run the SQL setup first.")
    else:
        print(f"❌ Connection failed: {result['error']}")
    
    return result

@app.local_entrypoint()
def test_anomaly_detection(metric: str = "implied_volatility"):
    """Test anomaly detection on a specific metric"""
    supabase_url = "https://xcwdavnejsnkddaroaaf.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhjd2Rhdm5lanNua2RkYXJvYWFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwNTY1MDQsImV4cCI6MjA3MDYzMjUwNH0.2q_9k1D26H9EFh2OBdEqVnqnMAEGHcErEFz36n9TgVY"
    
    print(f"🤖 Testing anomaly detection for metric: {metric}")
    result = simple_anomaly_detection.remote(supabase_url, supabase_key, metric)
    
    if result["status"] == "success":
        print(f"✅ Analysis complete: {result['summary']}")
        print(f"📊 Statistics:")
        stats = result['statistics']
        print(f"   • Total records: {stats['total_records']}")
        print(f"   • Anomalies found: {stats['anomalies_found']}")
        print(f"   • Mean {metric}: {stats['metric_mean']:.4f}")
        print(f"   • Std {metric}: {stats['metric_std']:.4f}")
        
        if result['anomalies']:
            print(f"\n🚨 Anomalies detected:")
            for i, anomaly in enumerate(result['anomalies'], 1):
                print(f"   {i}. {anomaly['symbol']} at {anomaly['timestamp']}")
                print(f"      Value: {anomaly['metric_value']:.4f} (z-score: {anomaly['z_score']:.2f})")
    else:
        print(f"❌ Analysis failed: {result['error']}")
    
    return result

if __name__ == "__main__":
    # For local testing
    test_connection()
