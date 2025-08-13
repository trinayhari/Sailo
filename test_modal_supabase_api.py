import modal

# Create Modal app
app = modal.App("sailo-etl-pipeline")

# Define the image with required dependencies
image = modal.Image.debian_slim().pip_install([
    "pandas",
    "numpy", 
    "requests",
    "supabase"  # Use the official Supabase Python client
])

@app.function(image=image)
def test_supabase_api_connection():
    """Test connection to your Supabase database using REST API"""
    import pandas as pd
    from supabase import create_client, Client
    import requests
    
    try:
        # Your Supabase credentials
        supabase_url = "https://xcwdavnejsnkddaroaaf.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhjd2Rhdm5lanNua2RkYXJvYWFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwNTY1MDQsImV4cCI6MjA3MDYzMjUwNH0.2q_9k1D26H9EFh2OBdEqVnqnMAEGHcErEFz36n9TgVY"
        
        print(f"🔍 Connecting to Supabase via REST API...")
        
        # Create Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        
        print(f"✅ Supabase client created successfully!")
        
        # Test connection by checking if options_trades table exists
        try:
            # Try to fetch a small sample to test connection
            response = supabase.table('options_trades').select('*').limit(1).execute()
            
            if response.data:
                print(f"✅ options_trades table found with data!")
                
                # Get table count
                count_response = supabase.table('options_trades').select('*', count='exact').execute()
                record_count = count_response.count
                
                print(f"📊 Found {record_count} records in options_trades")
                
                # Get sample data for analysis
                sample_response = supabase.table('options_trades').select(
                    'symbol, contract_type, strike_price, premium, volume, implied_volatility, created_at'
                ).order('created_at', desc=True).limit(10).execute()
                
                sample_data = sample_response.data
                
                print("📋 Sample options data:")
                for i, row in enumerate(sample_data[:5], 1):
                    iv = f"IV: {row.get('implied_volatility', 'N/A')}"
                    if isinstance(row.get('implied_volatility'), (int, float)):
                        iv = f"IV: {row['implied_volatility']:.4f}"
                    print(f"   {i}. {row['symbol']} {row['contract_type']} ${row['strike_price']} - Premium: ${row['premium']} - {iv}")
                
                return {
                    "status": "success",
                    "connection_method": "supabase_rest_api",
                    "options_records": record_count,
                    "sample_data": sample_data,
                    "message": f"Successfully connected via REST API! Found {record_count} options records."
                }
                
            else:
                print("⚠️ options_trades table exists but has no data")
                return {
                    "status": "success_no_data",
                    "connection_method": "supabase_rest_api",
                    "message": "Connected successfully but options_trades table has no data. Please run the SQL setup script."
                }
                
        except Exception as table_error:
            # Table might not exist
            if "relation" in str(table_error).lower() and "does not exist" in str(table_error).lower():
                print("⚠️ options_trades table does not exist")
                return {
                    "status": "success_no_table",
                    "connection_method": "supabase_rest_api",
                    "message": "Connected successfully but options_trades table not found. Please run the SQL setup script."
                }
            else:
                raise table_error
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Connection error: {error_msg}")
        
        # Provide helpful error messages
        if "unauthorized" in error_msg.lower() or "invalid" in error_msg.lower():
            return {
                "status": "auth_error",
                "error": "Authentication failed with the provided API key.",
                "suggestion": "Verify the Supabase URL and API key are correct."
            }
        elif "not found" in error_msg.lower():
            return {
                "status": "connection_error", 
                "error": "Supabase project not found.",
                "suggestion": "Check if the Supabase project URL is correct and the project is active."
            }
        else:
            return {
                "status": "error",
                "error": error_msg,
                "suggestion": "Check the connection details and try again."
            }

@app.function(image=image)
def run_ai_anomaly_detection_api():
    """Run AI-powered anomaly detection using Supabase REST API"""
    import pandas as pd
    import numpy as np
    from supabase import create_client, Client
    
    try:
        # Connection setup
        supabase_url = "https://xcwdavnejsnkddaroaaf.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhjd2Rhdm5lanNua2RkYXJvYWFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwNTY1MDQsImV4cCI6MjA3MDYzMjUwNH0.2q_9k1D26H9EFh2OBdEqVnqnMAEGHcErEFz36n9TgVY"
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        print("🤖 Starting AI anomaly detection on options data...")
        
        # Fetch comprehensive options data for analysis
        response = supabase.table('options_trades').select(
            'symbol, contract_type, strike_price, premium, implied_volatility, volume, delta, gamma, theta, vega, created_at'
        ).order('created_at', desc=True).execute()
        
        if not response.data:
            return {
                "status": "no_data",
                "message": "No options data found for analysis"
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(response.data)
        
        # Clean numeric columns
        numeric_cols = ['strike_price', 'premium', 'implied_volatility', 'volume', 'delta', 'gamma', 'theta', 'vega']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"📊 Analyzing {len(df)} options records...")
        
        # AI-powered anomaly detection scenarios
        anomalies_detected = {}
        
        # 1. Implied Volatility Spike Detection
        if 'implied_volatility' in df.columns:
            iv_data = df['implied_volatility'].dropna()
            if len(iv_data) > 0:
                iv_mean = iv_data.mean()
                iv_std = iv_data.std()
                
                if iv_std > 0:  # Avoid division by zero
                    # Z-score anomaly detection
                    df['iv_zscore'] = np.abs((df['implied_volatility'] - iv_mean) / iv_std)
                    iv_anomalies = df[df['iv_zscore'] > 2.5].copy()  # More than 2.5 standard deviations
                    
                    anomalies_detected['volatility_spikes'] = {
                        'count': len(iv_anomalies),
                        'threshold': 2.5,
                        'mean_iv': float(iv_mean),
                        'std_iv': float(iv_std),
                        'top_anomalies': []
                    }
                    
                    # Get top 3 volatility anomalies
                    for _, row in iv_anomalies.nlargest(3, 'iv_zscore').iterrows():
                        anomalies_detected['volatility_spikes']['top_anomalies'].append({
                            'symbol': row['symbol'],
                            'contract_type': row['contract_type'],
                            'strike_price': float(row['strike_price']) if pd.notna(row['strike_price']) else 0,
                            'implied_volatility': float(row['implied_volatility']) if pd.notna(row['implied_volatility']) else 0,
                            'z_score': float(row['iv_zscore']) if pd.notna(row['iv_zscore']) else 0,
                            'severity': 'HIGH' if row['iv_zscore'] > 3.0 else 'MEDIUM'
                        })
        
        # 2. Unusual Volume Detection
        if 'volume' in df.columns:
            volume_data = df['volume'].dropna()
            if len(volume_data) > 0:
                volume_mean = volume_data.mean()
                volume_std = volume_data.std()
                
                if volume_std > 0:
                    # Volume spike detection (more than 3x average)
                    volume_threshold = volume_mean + (3 * volume_std)
                    volume_anomalies = df[df['volume'] > volume_threshold].copy()
                    
                    anomalies_detected['unusual_volume'] = {
                        'count': len(volume_anomalies),
                        'threshold': float(volume_threshold),
                        'mean_volume': float(volume_mean),
                        'top_anomalies': []
                    }
                    
                    # Get top 3 volume anomalies
                    for _, row in volume_anomalies.nlargest(3, 'volume').iterrows():
                        anomalies_detected['unusual_volume']['top_anomalies'].append({
                            'symbol': row['symbol'],
                            'contract_type': row['contract_type'],
                            'volume': int(row['volume']) if pd.notna(row['volume']) else 0,
                            'volume_ratio': float(row['volume'] / volume_mean) if volume_mean > 0 and pd.notna(row['volume']) else 0,
                            'severity': 'CRITICAL' if row['volume'] > volume_threshold * 2 else 'HIGH'
                        })
        
        # 3. Premium Anomaly Detection (compared to strike price)
        if 'premium' in df.columns and 'strike_price' in df.columns:
            # Filter out rows with valid premium and strike price
            valid_rows = df[(df['premium'].notna()) & (df['strike_price'].notna()) & (df['strike_price'] > 0)]
            
            if len(valid_rows) > 0:
                valid_rows = valid_rows.copy()
                valid_rows['premium_ratio'] = valid_rows['premium'] / valid_rows['strike_price']
                premium_ratio_data = valid_rows['premium_ratio'].dropna()
                
                if len(premium_ratio_data) > 0:
                    pr_mean = premium_ratio_data.mean()
                    pr_std = premium_ratio_data.std()
                    
                    if pr_std > 0:
                        valid_rows['pr_zscore'] = np.abs((valid_rows['premium_ratio'] - pr_mean) / pr_std)
                        premium_anomalies = valid_rows[valid_rows['pr_zscore'] > 2.0].copy()
                        
                        anomalies_detected['premium_anomalies'] = {
                            'count': len(premium_anomalies),
                            'threshold': 2.0,
                            'mean_premium_ratio': float(pr_mean),
                            'top_anomalies': []
                        }
                        
                        # Get top 3 premium anomalies
                        for _, row in premium_anomalies.nlargest(3, 'pr_zscore').iterrows():
                            anomalies_detected['premium_anomalies']['top_anomalies'].append({
                                'symbol': row['symbol'],
                                'contract_type': row['contract_type'],
                                'premium': float(row['premium']) if pd.notna(row['premium']) else 0,
                                'strike_price': float(row['strike_price']) if pd.notna(row['strike_price']) else 0,
                                'premium_ratio': float(row['premium_ratio']) if pd.notna(row['premium_ratio']) else 0,
                                'z_score': float(row['pr_zscore']) if pd.notna(row['pr_zscore']) else 0,
                                'severity': 'HIGH' if row['pr_zscore'] > 2.5 else 'MEDIUM'
                            })
        
        # Generate AI summary
        total_anomalies = sum(scenario['count'] for scenario in anomalies_detected.values())
        
        # Create alert message
        alert_message = f"🚨 AI Anomaly Detection Complete!\n"
        alert_message += f"📊 Analyzed {len(df)} options records\n"
        alert_message += f"🔍 Detected {total_anomalies} total anomalies\n\n"
        
        for scenario_name, data in anomalies_detected.items():
            if data['count'] > 0:
                alert_message += f"• {scenario_name.replace('_', ' ').title()}: {data['count']} anomalies\n"
        
        return {
            "status": "success",
            "analysis_type": "ai_powered_anomaly_detection_via_api",
            "total_records": len(df),
            "total_anomalies": total_anomalies,
            "anomalies_by_type": anomalies_detected,
            "alert_message": alert_message,
            "summary": f"AI detected {total_anomalies} anomalies across {len(anomalies_detected)} scenarios from {len(df)} options records"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "AI anomaly detection failed"
        }

# Local entrypoint for testing
@app.local_entrypoint()
def main():
    """Main test function for the AI ETL Pipeline using Supabase REST API"""
    print("🚀 Testing Sailo AI ETL Pipeline with Modal + Supabase REST API")
    print("=" * 60)
    
    print("\n1️⃣ Testing Supabase REST API Connection...")
    connection_result = test_supabase_api_connection.remote()
    
    if connection_result["status"] == "success":
        print(f"✅ Supabase REST API connection successful!")
        print(f"🎯 Options records: {connection_result['options_records']}")
        
        print("\n2️⃣ Running AI-Powered Anomaly Detection...")
        anomaly_result = run_ai_anomaly_detection_api.remote()
        
        if anomaly_result["status"] == "success":
            print(f"✅ {anomaly_result['summary']}")
            print(f"\n📈 AI Analysis Results:")
            print(anomaly_result['alert_message'])
            
            # Show detailed anomalies
            for scenario_name, data in anomaly_result['anomalies_by_type'].items():
                if data['count'] > 0 and data['top_anomalies']:
                    print(f"\n🔍 {scenario_name.replace('_', ' ').title()}:")
                    for i, anomaly in enumerate(data['top_anomalies'], 1):
                        if 'implied_volatility' in anomaly:
                            print(f"   {i}. {anomaly['symbol']} {anomaly['contract_type']} - IV: {anomaly['implied_volatility']:.4f} (z-score: {anomaly['z_score']:.2f}) [{anomaly['severity']}]")
                        elif 'volume' in anomaly:
                            print(f"   {i}. {anomaly['symbol']} {anomaly['contract_type']} - Volume: {anomaly['volume']:,} ({anomaly['volume_ratio']:.1f}x avg) [{anomaly['severity']}]")
                        elif 'premium_ratio' in anomaly:
                            print(f"   {i}. {anomaly['symbol']} {anomaly['contract_type']} - Premium/Strike: {anomaly['premium_ratio']:.4f} (z-score: {anomaly['z_score']:.2f}) [{anomaly['severity']}]")
            
            print(f"\n🎉 AI ETL Pipeline Test Complete!")
            print(f"✅ Successfully processed {anomaly_result['total_records']} records via REST API")
            print(f"🤖 AI detected {anomaly_result['total_anomalies']} anomalies")
            print(f"🎯 Ready for hackathon demo!")
            
        else:
            print(f"⚠️ AI analysis issue: {anomaly_result.get('message', anomaly_result.get('error'))}")
    
    elif connection_result["status"] in ["success_no_data", "success_no_table"]:
        print(f"⚠️ {connection_result['message']}")
        print("💡 Next step: Run the SQL setup script in your Supabase dashboard")
        print("   1. Go to: https://supabase.com/dashboard/project/xcwdavnejsnkddaroaaf/sql")
        print("   2. Copy and paste the contents of 'quick_setup.sql'")
        print("   3. Click 'Run' to create the options_trades table with sample data")
        
    else:
        print(f"❌ Connection failed: {connection_result.get('message', connection_result.get('error'))}")
        if 'suggestion' in connection_result:
            print(f"💡 {connection_result['suggestion']}")
    
    return connection_result

if __name__ == "__main__":
    main()
