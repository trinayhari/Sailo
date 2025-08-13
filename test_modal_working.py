import modal

# Create Modal app
app = modal.App("sailo-etl-pipeline")

# Define the image with correct PostgreSQL dependencies
image = modal.Image.debian_slim().pip_install([
    "pandas",
    "numpy", 
    "sqlalchemy",
    "psycopg2-binary",  # This is the correct package for Modal
    "requests"
])

@app.function(image=image)
def test_supabase_connection():
    """Test connection to your Supabase database"""
    import pandas as pd
    from sqlalchemy import create_engine, text
    
    try:
        # Your Supabase credentials
        supabase_url = "https://xcwdavnejsnkddaroaaf.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhjd2Rhdm5lanNua2RkYXJvYWFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwNTY1MDQsImV4cCI6MjA3MDYzMjUwNH0.2q_9k1D26H9EFh2OBdEqVnqnMAEGHcErEFz36n9TgVY"
        
        # Convert to PostgreSQL connection string
        project_ref = supabase_url.replace('https://', '').replace('.supabase.co', '')
        
        # Use the direct database connection (this should work with anon key for read operations)
        pg_url = f"postgresql://postgres:{supabase_key}@db.{project_ref}.supabase.co:5432/postgres"
        
        print(f"🔍 Connecting to Supabase database...")
        engine = create_engine(pg_url)
        
        # Test with a simple query first
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            test_value = result.fetchone()[0]
            
            if test_value == 1:
                print(f"✅ Database connection successful!")
                
                # List all tables
                tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
                """
                
                df = pd.read_sql(text(tables_query), engine)
                tables = df['table_name'].tolist()
                
                print(f"📊 Found {len(tables)} tables: {', '.join(tables)}")
                
                # Check if options_trades exists
                if 'options_trades' in tables:
                    print("🎯 options_trades table found! Testing data fetch...")
                    
                    # Get table info
                    count_query = "SELECT COUNT(*) as record_count FROM options_trades"
                    count_df = pd.read_sql(text(count_query), engine)
                    record_count = count_df['record_count'].iloc[0]
                    
                    print(f"📈 options_trades has {record_count} records")
                    
                    if record_count > 0:
                        # Fetch sample data
                        sample_query = """
                        SELECT symbol, contract_type, strike_price, premium, volume, implied_volatility
                        FROM options_trades 
                        ORDER BY created_at DESC
                        LIMIT 5
                        """
                        
                        sample_df = pd.read_sql(text(sample_query), engine)
                        print("📋 Sample options data:")
                        for _, row in sample_df.iterrows():
                            iv = f"IV: {row['implied_volatility']:.4f}" if pd.notna(row['implied_volatility']) else "IV: N/A"
                            print(f"   {row['symbol']} {row['contract_type']} ${row['strike_price']} - Premium: ${row['premium']} - {iv}")
                        
                        return {
                            "status": "success",
                            "tables_found": tables,
                            "options_records": int(record_count),
                            "sample_data": sample_df.to_dict('records'),
                            "message": f"Successfully connected! Found {record_count} options records."
                        }
                    else:
                        return {
                            "status": "success_no_data",
                            "tables_found": tables,
                            "message": "options_trades table exists but has no data. Please run the SQL setup script in Supabase."
                        }
                else:
                    return {
                        "status": "success_no_table",
                        "tables_found": tables,
                        "message": "Connected successfully but options_trades table not found. Please run the SQL setup script in Supabase."
                    }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Connection error: {error_msg}")
        
        # Provide helpful error messages
        if "password authentication failed" in error_msg.lower():
            return {
                "status": "auth_error",
                "error": "Authentication failed. The anon key might not have sufficient permissions.",
                "suggestion": "Try using the service role key or database password instead."
            }
        elif "could not connect" in error_msg.lower():
            return {
                "status": "connection_error", 
                "error": "Could not connect to database server.",
                "suggestion": "Check if the Supabase project is active and the connection string is correct."
            }
        else:
            return {
                "status": "error",
                "error": error_msg,
                "suggestion": "Check the connection details and try again."
            }

@app.function(image=image)
def run_ai_anomaly_detection():
    """Run AI-powered anomaly detection on options trading data"""
    import pandas as pd
    import numpy as np
    from sqlalchemy import create_engine, text
    
    try:
        # Connection setup
        supabase_url = "https://xcwdavnejsnkddaroaaf.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhjd2Rhdm5lanNua2RkYXJvYWFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwNTY1MDQsImV4cCI6MjA3MDYzMjUwNH0.2q_9k1D26H9EFh2OBdEqVnqnMAEGHcErEFz36n9TgVY"
        
        project_ref = supabase_url.replace('https://', '').replace('.supabase.co', '')
        pg_url = f"postgresql://postgres:{supabase_key}@db.{project_ref}.supabase.co:5432/postgres"
        
        engine = create_engine(pg_url)
        
        print("🤖 Starting AI anomaly detection on options data...")
        
        # Fetch comprehensive options data for analysis
        query = """
        SELECT 
            symbol,
            contract_type,
            strike_price,
            premium,
            implied_volatility,
            volume,
            delta,
            gamma,
            theta,
            vega,
            created_at
        FROM options_trades 
        WHERE implied_volatility IS NOT NULL
        ORDER BY created_at DESC
        """
        
        df = pd.read_sql(text(query), engine)
        
        if df.empty:
            return {
                "status": "no_data",
                "message": "No options data found for analysis"
            }
        
        print(f"📊 Analyzing {len(df)} options records...")
        
        # AI-powered anomaly detection scenarios
        anomalies_detected = {}
        
        # 1. Implied Volatility Spike Detection
        if 'implied_volatility' in df.columns:
            iv_data = df['implied_volatility'].dropna()
            if len(iv_data) > 0:
                iv_mean = iv_data.mean()
                iv_std = iv_data.std()
                
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
                        'strike_price': float(row['strike_price']),
                        'implied_volatility': float(row['implied_volatility']),
                        'z_score': float(row['iv_zscore']),
                        'severity': 'HIGH' if row['iv_zscore'] > 3.0 else 'MEDIUM'
                    })
        
        # 2. Unusual Volume Detection
        if 'volume' in df.columns:
            volume_data = df['volume'].dropna()
            if len(volume_data) > 0:
                volume_mean = volume_data.mean()
                volume_std = volume_data.std()
                
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
                        'volume': int(row['volume']),
                        'volume_ratio': float(row['volume'] / volume_mean),
                        'severity': 'CRITICAL' if row['volume'] > volume_threshold * 2 else 'HIGH'
                    })
        
        # 3. Premium Anomaly Detection (compared to strike price)
        if 'premium' in df.columns and 'strike_price' in df.columns:
            df['premium_ratio'] = df['premium'] / df['strike_price']
            premium_ratio_data = df['premium_ratio'].dropna()
            
            if len(premium_ratio_data) > 0:
                pr_mean = premium_ratio_data.mean()
                pr_std = premium_ratio_data.std()
                
                df['pr_zscore'] = np.abs((df['premium_ratio'] - pr_mean) / pr_std)
                premium_anomalies = df[df['pr_zscore'] > 2.0].copy()
                
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
                        'premium': float(row['premium']),
                        'strike_price': float(row['strike_price']),
                        'premium_ratio': float(row['premium_ratio']),
                        'z_score': float(row['pr_zscore']),
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
            "analysis_type": "ai_powered_anomaly_detection",
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
    """Main test function for the AI ETL Pipeline"""
    print("🚀 Testing Sailo AI ETL Pipeline with Modal")
    print("=" * 50)
    
    print("\n1️⃣ Testing Supabase Connection...")
    connection_result = test_supabase_connection.remote()
    
    if connection_result["status"] == "success":
        print(f"✅ Database connection successful!")
        print(f"📊 Found {len(connection_result['tables_found'])} tables")
        print(f"🎯 Options records: {connection_result['options_records']}")
        
        print("\n2️⃣ Running AI-Powered Anomaly Detection...")
        anomaly_result = run_ai_anomaly_detection.remote()
        
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
            print(f"✅ Successfully processed {anomaly_result['total_records']} records")
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
