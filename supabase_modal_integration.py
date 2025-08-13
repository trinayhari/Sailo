"""
Supabase-Modal Integration for AI ETL Pipeline
Connects your React Supabase data retrieval with Modal's AI processing
"""

import os
import json
from typing import Dict, Any, List, Optional
import modal
from modal_app import app, run_monitoring_pipeline, plan_with_llm, inspect_database

# Integration functions to bridge Supabase data with Modal AI pipeline

class SupabaseModalBridge:
    """Bridge class to connect Supabase data retrieval with Modal AI ETL pipeline"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        # Convert Supabase URL to PostgreSQL connection string
        self.pg_url = self._convert_supabase_to_pg_url(supabase_url, supabase_key)
    
    def _convert_supabase_to_pg_url(self, supabase_url: str, supabase_key: str) -> str:
        """Convert Supabase URL to PostgreSQL connection string"""
        # Extract project reference from Supabase URL
        project_ref = supabase_url.replace('https://', '').replace('.supabase.co', '')
        
        # Note: For production, you'd need the database password
        # For now, we'll use the service role key approach
        return f"postgresql://postgres:[YOUR_DB_PASSWORD]@db.{project_ref}.supabase.co:5432/postgres"
    
    def create_options_monitoring_plan(self, goal: str, slack_webhook: Optional[str] = None) -> Dict[str, Any]:
        """Create AI monitoring plan specifically for options trading data"""
        
        # Define options trading SQL queries for different monitoring scenarios
        monitoring_queries = {
            "volatility_spikes": """
                SELECT 
                    trade_timestamp as timestamp,
                    symbol,
                    implied_volatility,
                    volume,
                    premium,
                    delta,
                    gamma
                FROM options_trades 
                WHERE trade_timestamp >= NOW() - INTERVAL '7 days'
                ORDER BY trade_timestamp DESC
            """,
            
            "unusual_volume": """
                SELECT 
                    trade_timestamp as timestamp,
                    symbol,
                    contract_type,
                    volume,
                    open_interest,
                    premium,
                    strike_price
                FROM options_trades 
                WHERE trade_timestamp >= NOW() - INTERVAL '24 hours'
                ORDER BY trade_timestamp DESC
            """,
            
            "risk_metrics": """
                SELECT 
                    calculated_at as timestamp,
                    portfolio_id,
                    var_1d,
                    var_5d,
                    total_exposure,
                    delta_exposure,
                    concentration_risk
                FROM risk_metrics 
                WHERE calculated_at >= NOW() - INTERVAL '7 days'
                ORDER BY calculated_at DESC
            """,
            
            "active_alerts": """
                SELECT 
                    created_at as timestamp,
                    alert_type,
                    severity,
                    symbol,
                    message,
                    details
                FROM trading_alerts 
                WHERE status = 'ACTIVE' 
                AND created_at >= NOW() - INTERVAL '24 hours'
                ORDER BY created_at DESC
            """
        }
        
        # Determine which query to use based on the goal
        if "volatility" in goal.lower() or "iv" in goal.lower():
            sql_query = monitoring_queries["volatility_spikes"]
        elif "volume" in goal.lower():
            sql_query = monitoring_queries["unusual_volume"]
        elif "risk" in goal.lower() or "exposure" in goal.lower():
            sql_query = monitoring_queries["risk_metrics"]
        elif "alert" in goal.lower():
            sql_query = monitoring_queries["active_alerts"]
        else:
            # Default to volatility monitoring
            sql_query = monitoring_queries["volatility_spikes"]
        
        # Use Modal's AI planning function
        schema_info = inspect_database.remote(self.pg_url, sql_query)
        plan = plan_with_llm.remote(schema_info, goal, slack_webhook)
        
        return {
            "plan": plan,
            "sql_query": sql_query,
            "schema_info": schema_info
        }
    
    def run_options_monitoring(self, goal: str, slack_webhook: Optional[str] = None) -> Dict[str, Any]:
        """Run complete options trading monitoring pipeline"""
        
        # Create monitoring plan
        monitoring_setup = self.create_options_monitoring_plan(goal, slack_webhook)
        plan = monitoring_setup["plan"]
        sql_query = monitoring_setup["sql_query"]
        
        # Run the monitoring pipeline
        result = run_monitoring_pipeline.remote(self.pg_url, sql_query, plan)
        
        return {
            "monitoring_result": result,
            "plan_used": plan,
            "sql_query": sql_query,
            "goal": goal
        }

# Pre-defined monitoring scenarios for options trading
TRADING_SCENARIOS = {
    "volatility_spike": {
        "goal": "detect implied volatility spikes above 50% increase in the last hour",
        "description": "Monitor for sudden volatility increases that might indicate market stress"
    },
    
    "unusual_volume": {
        "goal": "detect options volume 300% above average for specific strikes",
        "description": "Identify unusual trading activity that might indicate insider information"
    },
    
    "risk_exposure": {
        "goal": "monitor portfolio delta exposure exceeding risk limits",
        "description": "Track portfolio risk metrics to prevent excessive exposure"
    },
    
    "price_anomaly": {
        "goal": "detect options pricing anomalies compared to theoretical values",
        "description": "Find mispriced options that might present arbitrage opportunities"
    },
    
    "gamma_squeeze": {
        "goal": "detect high gamma exposure that might cause price acceleration",
        "description": "Monitor for gamma squeeze conditions in popular strikes"
    }
}

@app.function()
def run_predefined_scenario(scenario_name: str, supabase_url: str, supabase_key: str, slack_webhook: Optional[str] = None):
    """Run a predefined trading monitoring scenario"""
    
    if scenario_name not in TRADING_SCENARIOS:
        available = ", ".join(TRADING_SCENARIOS.keys())
        raise ValueError(f"Scenario '{scenario_name}' not found. Available: {available}")
    
    scenario = TRADING_SCENARIOS[scenario_name]
    bridge = SupabaseModalBridge(supabase_url, supabase_key)
    
    print(f"🎯 Running scenario: {scenario_name}")
    print(f"📋 Description: {scenario['description']}")
    
    result = bridge.run_options_monitoring(scenario["goal"], slack_webhook)
    
    return {
        "scenario": scenario_name,
        "description": scenario["description"],
        "result": result
    }

@app.local_entrypoint()
def test_integration():
    """Test the Supabase-Modal integration with your options trading data"""
    
    # Load environment variables
    supabase_url = os.getenv("VITE_SUPABASE_URL", "https://xcwdavnejsnkddaroaaf.supabase.co")
    supabase_key = os.getenv("VITE_SUPABASE_ANON_KEY")
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    
    if not supabase_key:
        print("❌ Please set VITE_SUPABASE_ANON_KEY environment variable")
        return
    
    print("🚀 Testing Supabase-Modal Integration for Options Trading")
    print(f"📊 Supabase URL: {supabase_url}")
    
    # Test volatility spike detection
    print("\n🎯 Testing volatility spike detection...")
    result = run_predefined_scenario.remote(
        "volatility_spike", 
        supabase_url, 
        supabase_key, 
        slack_webhook
    )
    
    print(f"✅ Found {result['result']['monitoring_result']['anomalies']['count']} anomalies")
    print(f"📱 Slack alert sent: {result['result']['monitoring_result']['slack_sent']}")
    
    return result

if __name__ == "__main__":
    # For local testing
    test_integration()
