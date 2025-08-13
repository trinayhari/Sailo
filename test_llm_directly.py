#!/usr/bin/env python3
"""Test LLM function directly to debug the issue"""

import modal

# Connect to the deployed app
app = modal.App.lookup("sailo-mvp", create_if_missing=False)

def test_llm():
    """Test the LLM analysis function directly"""
    
    # Simple test prompt
    test_prompt = """
    You are analyzing data. Please respond with this exact JSON format:

    ```json
    {
        "domain_detected": "test data",
        "summary": "This is a test",
        "total_anomalies": 1,
        "anomalies": [
            {
                "identifier": "test",
                "value": "test_value",
                "severity": "LOW",
                "details": "This is a test anomaly",
                "action_required": "No action needed",
                "business_impact": "None",
                "reason": "This is just a test"
            }
        ],
        "insights": ["Test insight 1", "Test insight 2"],
        "recommendations": ["Test recommendation 1"]
    }
    ```
    """
    
    print("🧪 Testing LLM function directly...")
    
    try:
        # Get the function from the deployed app
        analyze_with_llm = app.analyze_with_llm
        
        print("🔄 Calling LLM function...")
        # Call it
        result = analyze_with_llm.remote(test_prompt)
        
        print(f"✅ LLM Response ({len(result) if result else 0} chars):")
        print(result)
        
        return result
        
    except Exception as e:
        print(f"❌ LLM Test Failed: {e}")
        return None

if __name__ == "__main__":
    test_llm()
