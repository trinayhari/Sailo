#!/usr/bin/env python3
"""Test the run-once endpoint with LLM"""

import requests
import json

def test_run_once():
    url = "https://sailo--sailo-mvp-fastapi-app.modal.run/run-once"
    
    payload = {
        "plan": {
            "metric": "volume",
            "timestamp_col": "trade_timestamp", 
            "method": "ai_analysis",
            "threshold": "ai_determined",
            "schedule_minutes": 15,
            "action": "slack",
            "action_config": {"webhook_url": "not_provided"}
        },
        "table_name": "options_trades",
        "goal": "find unusual patterns in my options trading data"
    }
    
    print("🧪 Testing run-once endpoint with LLM...")
    print(f"Goal: {payload['goal']}")
    
    try:
        response = requests.post(url, json=payload, timeout=120)  # 2 min timeout
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Response received!")
            print(f"Success: {result.get('success')}")
            
            if result.get('success'):
                results = result.get('results', {})
                print(f"\nDomain: {results.get('domain_detected')}")
                print(f"Type: {results.get('type')}")
                print(f"Summary: {results.get('summary')}")
                print(f"Anomalies: {results.get('total_anomalies')}")
                
                if results.get('anomalies'):
                    print("\nFirst anomaly:")
                    print(json.dumps(results['anomalies'][0], indent=2))
                    
            else:
                print(f"❌ Error: {result.get('error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - LLM might be taking too long")
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_run_once()
