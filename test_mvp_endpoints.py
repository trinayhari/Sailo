#!/usr/bin/env python3
"""
Test script for MVP endpoints
Run this to verify all three endpoints are working correctly
"""

import requests
import json
import time

# Modal endpoint base URL
BASE_URL = "https://sailo--sailo-mvp"

def test_endpoint(endpoint_name, endpoint_url, payload, description):
    """Test a single endpoint"""
    print(f"\n{'='*50}")
    print(f"🧪 Testing {endpoint_name}")
    print(f"📝 {description}")
    print(f"🌐 URL: {endpoint_url}")
    print(f"{'='*50}")
    
    try:
        # Make the request
        print("⏳ Sending request...")
        start_time = time.time()
        
        response = requests.post(
            endpoint_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️  Response time: {duration:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            
            # Pretty print the response
            if isinstance(result, dict):
                if result.get('success'):
                    print("🎉 Endpoint returned success=True")
                    
                    # Print specific fields based on endpoint
                    if 'inspection' in result:
                        inspection = result['inspection']
                        print(f"📋 Table: {inspection.get('table')}")
                        print(f"📊 Records: {inspection.get('total_records')}")
                        print(f"🔢 Columns: {inspection.get('total_columns')}")
                        print(f"🔢 Numeric cols: {inspection.get('columns', {}).get('numeric', [])}")
                        
                    elif 'plan' in result:
                        plan = result['plan']
                        print(f"📈 Monitoring metric: {plan.get('metric')}")
                        print(f"⏰ Timestamp column: {plan.get('timestamp_col')}")
                        print(f"🎯 Method: {plan.get('method')}")
                        print(f"🔔 Action: {plan.get('action')}")
                        
                    elif 'results' in result:
                        results = result['results']
                        print(f"📊 Analysis type: {results.get('type', 'N/A')}")
                        print(f"🚨 Anomalies found: {results.get('total_anomalies', 0)}")
                        print(f"📨 Slack sent: {result.get('slack_sent', False)}")
                        if results.get('summary'):
                            print(f"📝 Summary: {results['summary'][:100]}...")
                            
                else:
                    print(f"❌ Endpoint returned success=False")
                    print(f"🔥 Error: {result.get('error', 'Unknown error')}")
            
            return result
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"🔥 Response: {response.text[:200]}...")
            return None
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (60s)")
        return None
    except requests.exceptions.RequestException as e:
        print(f"🌐 Network error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"📄 JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return None

def main():
    print("🚀 Testing Sailo MVP Endpoints")
    print("🔗 Modal App: sailo-web-api")
    
    # Test 1: Inspect Source
    inspect_result = test_endpoint(
        "Inspect Source",
        f"{BASE_URL}-inspect-source.modal.run",
        {"table_name": "options_trades"},
        "Analyze database schema and sample data"
    )
    
    if not inspect_result or not inspect_result.get('success'):
        print("\n💀 Inspect failed - stopping tests")
        return
    
    # Test 2: Auto Plan (using inspect result)
    plan_payload = {
        "goal": "detect spikes in options volume and alert team",
        "inspection_data": inspect_result.get('inspection', {}),
        "slack_webhook": "https://hooks.slack.com/services/TEST/DEMO/WEBHOOK"
    }
    
    plan_result = test_endpoint(
        "Auto Plan",
        f"{BASE_URL}-auto-plan.modal.run", 
        plan_payload,
        "AI-generate monitoring plan"
    )
    
    if not plan_result or not plan_result.get('success'):
        print("\n💀 Auto Plan failed - stopping tests")
        return
    
    # Test 3: Run Once (using plan result)
    run_payload = {
        "plan": plan_result.get('plan', {}),
        "table_name": "options_trades"
    }
    
    run_result = test_endpoint(
        "Run Once",
        f"{BASE_URL}-run-once.modal.run",
        run_payload, 
        "Execute monitoring pipeline"
    )
    
    # Final summary
    print(f"\n{'='*50}")
    print("🎯 TEST SUMMARY")
    print(f"{'='*50}")
    
    tests = [
        ("Inspect Source", inspect_result),
        ("Auto Plan", plan_result), 
        ("Run Once", run_result)
    ]
    
    all_passed = True
    for test_name, result in tests:
        if result and result.get('success'):
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! MVP is ready for demo.")
        print("\n🎬 Demo steps:")
        print("1. Open http://localhost:5173")
        print("2. Click 'MVP Demo' tab")
        print("3. Follow the 3-button flow")
    else:
        print("\n💥 Some tests failed. Check Modal deployment and Supabase setup.")
        print("\n🔧 Troubleshooting:")
        print("- Ensure Modal app is deployed: modal deploy modal_web_endpoints_clean.py")
        print("- Check Supabase secret exists: modal secret list")
        print("- Verify options_trades table has data in Supabase")

if __name__ == "__main__":
    main()
