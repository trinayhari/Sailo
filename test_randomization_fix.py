#!/usr/bin/env python3
"""
Test script to demonstrate the randomization fix for LLM analysis.
This shows how different queries should produce different results.
"""

import requests
import json
import time

BASE_URL = "https://sailo--sailo-mvp-fastapi-app.modal.run"

def test_query_randomization():
    """Test that different queries produce different results"""
    
    # Test data
    test_cases = [
        {
            "name": "High Volatility Query",
            "goal": "find high volatility stocks for aggressive trading",
            "expected_focus": "high volatility"
        },
        {
            "name": "Low Volatility Query", 
            "goal": "find low volatility stocks for conservative investing",
            "expected_focus": "low volatility"
        },
        {
            "name": "Volume Analysis Query",
            "goal": "identify unusual trading volume patterns",
            "expected_focus": "volume patterns"
        }
    ]
    
    results = {}
    
    print("🧪 Testing Query-Specific Randomization")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"\n📊 Testing: {test_case['name']}")
        print(f"Goal: {test_case['goal']}")
        
        # Make request to current backend
        payload = {
            "plan": {
                "metric": "implied_volatility",
                "method": "ai_analysis",
                "timestamp_col": "created_at",
                "threshold": "ai_determined",
                "schedule_minutes": 15,
                "action": "slack"
            },
            "table_name": "options_trades",
            "goal": test_case['goal']
        }
        
        try:
            response = requests.post(f"{BASE_URL}/run-once", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    results[test_case['name']] = {
                        'total_anomalies': data['results']['total_anomalies'],
                        'summary': data['results']['summary'],
                        'first_anomaly': data['results']['anomalies'][0] if data['results']['anomalies'] else None
                    }
                    print(f"✅ Success: {data['results']['total_anomalies']} anomalies found")
                    print(f"Summary: {data['results']['summary'][:100]}...")
                else:
                    print(f"❌ API Error: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request Error: {e}")
        
        # Small delay between requests
        time.sleep(2)
    
    # Analyze results
    print("\n" + "=" * 50)
    print("📈 ANALYSIS RESULTS")
    print("=" * 50)
    
    if len(results) >= 2:
        # Check if results are different
        anomaly_counts = [r['total_anomalies'] for r in results.values()]
        summaries = [r['summary'] for r in results.values()]
        
        print(f"\nAnomaly Counts: {anomaly_counts}")
        print(f"Unique Counts: {len(set(anomaly_counts))}")
        print(f"Unique Summaries: {len(set(summaries))}")
        
        if len(set(anomaly_counts)) == 1 and len(set(summaries)) == 1:
            print("\n❌ ISSUE CONFIRMED: All queries return identical results")
            print("🔧 SOLUTION: Backend needs randomization fixes deployed")
        else:
            print("\n✅ SUCCESS: Queries return different results")
            print("🎉 Randomization is working correctly!")
    
    return results

def show_randomization_solution():
    """Show the code changes needed to fix the randomization issue"""
    
    print("\n" + "=" * 60)
    print("🔧 RANDOMIZATION FIX SOLUTION")
    print("=" * 60)
    
    print("""
The issue is that the LLM analysis lacks proper randomization. Here's the fix:

1. ADD UNIQUE SEEDING:
```python
import hashlib
import time
seed = int(hashlib.md5(f"{analysis_prompt}{time.time()}".encode()).hexdigest()[:8], 16)
torch.manual_seed(seed)
random.seed(seed)
```

2. INCREASE TEMPERATURE:
```python
temperature=0.8  # Higher temperature for more varied responses
```

3. ENHANCE PROMPT SPECIFICITY:
```python
analysis_prompt = f'''
ANALYSIS ID: {analysis_id}
SPECIFIC USER GOAL: {goal}

CRITICAL INSTRUCTIONS:
1. This analysis is specifically for: "{goal}"
2. If the goal mentions "high volatility", focus on HIGH volatility data points
3. If the goal mentions "low volatility", focus on LOW volatility data points  
4. Provide DIFFERENT results for different goals
'''
```

These changes ensure that:
- Each query gets a unique seed based on content + timestamp
- Higher temperature creates more variation in responses  
- Prompts explicitly instruct the LLM to differentiate between queries
- Results will be genuinely different for different user goals
""")

if __name__ == "__main__":
    # Run the test
    results = test_query_randomization()
    
    # Show the solution
    show_randomization_solution()
    
    print(f"\n📋 Test completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
