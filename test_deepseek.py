#!/usr/bin/env python3

import modal

def test_deepseek_llm():
    """Test the DeepSeek R1 LLM to verify it's working"""
    
    # Connect to the deployed app
    app = modal.App.lookup("deepseek-simple-test", create_if_missing=False)
    
    # Get the test function
    test_func = modal.Function.lookup("deepseek-simple-test", "test_deepseek_simple")
    
    print("🚀 Testing DeepSeek R1 LLM...")
    
    # Test with a simple prompt
    test_prompt = "What is 2+2? Answer in one sentence."
    
    try:
        # Call the LLM function
        result = test_func.remote(test_prompt)
        
        print(f"✅ LLM Test Result:")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Model: {result.get('model', 'Unknown')}")
        print(f"   Prompt: {result.get('prompt', 'N/A')}")
        print(f"   Output: {result.get('output', 'No output')[:200]}...")
        
        if result.get('success'):
            print("🎉 DeepSeek R1 LLM is working!")
            return True
        else:
            print(f"❌ LLM Test Failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing LLM: {e}")
        return False

if __name__ == "__main__":
    test_deepseek_llm()
