import modal

# Create Modal app
app = modal.App("sailo-test")

# Define the image with all dependencies
image = modal.Image.debian_slim().pip_install([
    "pandas",
    "numpy", 
    "sqlalchemy",
    "psycopg[binary]",
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
        
        # Try different connection approaches
        connection_strings = [
            f"postgresql://postgres.{project_ref}:{supabase_key}@aws-0-us-west-1.pooler.supabase.com:6543/postgres",
            f"postgresql://postgres:{supabase_key}@db.{project_ref}.supabase.co:5432/postgres"
        ]
        
        for i, pg_url in enumerate(connection_strings, 1):
            try:
                print(f"🔍 Trying connection method {i}...")
                engine = create_engine(pg_url)
                
                # Test with a simple query first
                result = engine.execute(text("SELECT 1 as test"))
                test_value = result.fetchone()[0]
                
                if test_value == 1:
                    print(f"✅ Connection method {i} successful!")
                    
                    # Now try to list tables
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
                        
                        data_query = """
                        SELECT COUNT(*) as record_count 
                        FROM options_trades
                        """
                        
                        count_df = pd.read_sql(text(data_query), engine)
                        record_count = count_df['record_count'].iloc[0]
                        
                        print(f"📈 options_trades has {record_count} records")
                        
                        if record_count > 0:
                            # Fetch sample data
                            sample_query = """
                            SELECT symbol, contract_type, strike_price, premium, volume
                            FROM options_trades 
                            LIMIT 3
                            """
                            
                            sample_df = pd.read_sql(text(sample_query), engine)
                            print("📋 Sample data:")
                            for _, row in sample_df.iterrows():
                                print(f"   {row['symbol']} {row['contract_type']} ${row['strike_price']} - Premium: ${row['premium']}")
                            
                            return {
                                "status": "success",
                                "connection_method": i,
                                "tables_found": tables,
                                "options_records": int(record_count),
                                "sample_data": sample_df.to_dict('records')
                            }
                        else:
                            return {
                                "status": "success_no_data",
                                "connection_method": i,
                                "tables_found": tables,
                                "message": "options_trades table exists but has no data. Run the SQL setup script."
                            }
                    else:
                        return {
                            "status": "success_no_table",
                            "connection_method": i,
                            "tables_found": tables,
                            "message": "Connected but options_trades table not found. Run the SQL setup script."
                        }
                        
            except Exception as conn_error:
                print(f"❌ Connection method {i} failed: {str(conn_error)}")
                continue
        
        # If we get here, all connection methods failed
        return {
            "status": "error",
            "message": "All connection methods failed",
            "error": "Could not establish database connection"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": "Unexpected error",
            "error": str(e)
        }

@app.function(image=image)
def run_simple_anomaly_detection():
    """Run simple anomaly detection on options data"""
    import pandas as pd
    import numpy as np
    from sqlalchemy import create_engine, text
    
    try:
        # Use the working connection from previous test
        supabase_url = "https://xcwdavnejsnkddaroaaf.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhjd2Rhdm5lanNua2RkYXJvYWFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwNTY1MDQsImV4cCI6MjA3MDYzMjUwNH0.2q_9k1D26H9EFh2OBdEqVnqnMAEGHcErEFz36n9TgVY"
        
        project_ref = supabase_url.replace('https://', '').replace('.supabase.co', '')
        pg_url = f"postgresql://postgres.{project_ref}:{supabase_key}@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
        
        engine = create_engine(pg_url)
        
        # Fetch options data for analysis
        query = """
        SELECT 
            symbol,
            contract_type,
            strike_price,
            premium,
            implied_volatility,
            volume,
            created_at
        FROM options_trades 
        ORDER BY created_at DESC
        """
        
        df = pd.read_sql(text(query), engine)
        
        if df.empty:
            return {
                "status": "no_data",
                "message": "No data found in options_trades table"
            }
        
        print(f"📊 Analyzing {len(df)} options records...")
        
        # Simple anomaly detection on implied volatility
        if 'implied_volatility' in df.columns and df['implied_volatility'].notna().any():
            iv_data = df['implied_volatility'].dropna()
            iv_mean = iv_data.mean()
            iv_std = iv_data.std()
            
            # Find high volatility anomalies (z-score > 2)
            df['iv_zscore'] = np.abs((df['implied_volatility'] - iv_mean) / iv_std)
            anomalies = df[df['iv_zscore'] > 2.0].copy()
            
            print(f"🚨 Found {len(anomalies)} volatility anomalies")
            
            # Prepare results
            anomaly_list = []
            for _, row in anomalies.head(5).iterrows():  # Top 5 anomalies
                anomaly_list.append({
                    'symbol': row['symbol'],
                    'contract_type': row['contract_type'],
                    'strike_price': float(row['strike_price']),
                    'implied_volatility': float(row['implied_volatility']),
                    'z_score': float(row['iv_zscore']),
                    'volume': int(row['volume']) if pd.notna(row['volume']) else 0
                })
            
            return {
                "status": "success",
                "analysis_type": "implied_volatility_anomalies",
                "total_records": len(df),
                "anomalies_found": len(anomalies),
                "iv_mean": float(iv_mean),
                "iv_std": float(iv_std),
                "top_anomalies": anomaly_list,
                "summary": f"Detected {len(anomalies)} volatility anomalies out of {len(df)} records"
            }
        else:
            return {
                "status": "no_iv_data",
                "message": "No implied volatility data available for analysis",
                "total_records": len(df)
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Local entrypoint for testing
@app.local_entrypoint()
def main():
    """Main test function"""
    print("🚀 Testing Sailo AI ETL Pipeline with Modal")
    print("=" * 50)
    
    print("\n1️⃣ Testing Supabase Connection...")
    connection_result = test_supabase_connection.remote()
    
    if connection_result["status"] == "success":
        print(f"✅ Database connection successful!")
        print(f"📊 Found {len(connection_result['tables_found'])} tables")
        print(f"🎯 Options records: {connection_result['options_records']}")
        
        print("\n2️⃣ Running Anomaly Detection...")
        anomaly_result = run_simple_anomaly_detection.remote()
        
        if anomaly_result["status"] == "success":
            print(f"✅ {anomaly_result['summary']}")
            print(f"📈 IV Mean: {anomaly_result['iv_mean']:.4f}")
            print(f"📊 IV Std: {anomaly_result['iv_std']:.4f}")
            
            if anomaly_result['top_anomalies']:
                print("\n🚨 Top Anomalies:")
                for i, anomaly in enumerate(anomaly_result['top_anomalies'], 1):
                    print(f"   {i}. {anomaly['symbol']} {anomaly['contract_type']} ${anomaly['strike_price']}")
                    print(f"      IV: {anomaly['implied_volatility']:.4f} (z-score: {anomaly['z_score']:.2f})")
            
            print(f"\n🎉 AI ETL Pipeline Test Complete!")
            print(f"✅ Successfully processed {anomaly_result['total_records']} records")
            print(f"🤖 AI detected {anomaly_result['anomalies_found']} anomalies")
            
        else:
            print(f"⚠️ Anomaly detection issue: {anomaly_result.get('message', anomaly_result.get('error'))}")
    
    elif connection_result["status"] == "success_no_data":
        print(f"⚠️ {connection_result['message']}")
        print("💡 Next step: Run the SQL setup script in your Supabase dashboard")
        
    elif connection_result["status"] == "success_no_table":
        print(f"⚠️ {connection_result['message']}")
        print("💡 Next step: Run the SQL setup script in your Supabase dashboard")
        
    else:
        print(f"❌ Connection failed: {connection_result.get('message', connection_result.get('error'))}")
        print("💡 Check your Supabase credentials and database setup")
    
    return connection_result

if __name__ == "__main__":
    main()
