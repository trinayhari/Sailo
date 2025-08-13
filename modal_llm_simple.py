import modal
import json

# Create Modal app for LLM analysis
app = modal.App("sailo-llm-analysis")

@app.function(
    timeout=600,
    image=modal.Image.debian_slim().pip_install("openai")
)
def analyze_data_with_llm(analysis_prompt: str, unique_id: str = None, model_name: str = "mock"):
    """Perform comprehensive data analysis - uses mock responses for testing"""
    import json
    import hashlib
    import random
    import re
    
    print(f"🤖 Processing analysis request (ID: {unique_id})...")
    
    # Create truly dynamic mock responses based on the query
    query_hash = hashlib.md5(f"{analysis_prompt}{unique_id}".encode()).hexdigest()
    random.seed(query_hash)  # Consistent but unique responses per query
    
    # Extract the actual user query from the prompt
    query_match = re.search(r'USER QUERY: "([^"]+)"', analysis_prompt)
    user_query = query_match.group(1) if query_match else "analyze data"
    query_text = user_query.lower()
    
    print(f"🔍 Analyzing query: '{user_query}'")
    
    # Dynamic response based on query content
    if "watch out" in query_text or "warning" in query_text or "risk" in query_text:
        focus = "risk_analysis"
        message = f"⚠️ Risk Analysis Complete - Found {random.randint(2,5)} high-priority warnings requiring immediate attention"
        severity_options = ["HIGH", "CRITICAL", "MEDIUM"]
        action_prefix = "URGENT"
        interpretation = "User is asking for risk assessment and potential warnings in the data"
    elif "opportunity" in query_text or "undervalued" in query_text or "buy" in query_text:
        focus = "opportunity_analysis" 
        message = f"💰 Opportunity Analysis Complete - Identified {random.randint(1,4)} profitable opportunities"
        severity_options = ["MEDIUM", "LOW", "INFO"]
        action_prefix = "CONSIDER"
        interpretation = "User is looking for investment opportunities and undervalued positions"
    elif "suspicious" in query_text or "anomaly" in query_text or "outlier" in query_text:
        focus = "anomaly_detection"
        message = f"🔍 Anomaly Detection Complete - Detected {random.randint(3,7)} suspicious patterns"
        severity_options = ["HIGH", "MEDIUM", "CRITICAL"]
        action_prefix = "INVESTIGATE"
        interpretation = "User wants to identify suspicious patterns and anomalies in the data"
    else:
        focus = "general_analysis"
        message = f"📊 General Analysis Complete - Found {random.randint(2,6)} actionable insights"
        severity_options = ["MEDIUM", "LOW", "INFO"]
        action_prefix = "REVIEW"
        interpretation = "User is requesting a general analysis of the data"
        
    # Generate dynamic anomalies based on query focus
    anomalies = []
    num_anomalies = random.randint(2, 4)
    
    for i in range(num_anomalies):
        symbol = f"{random.choice(['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'NVDA'])}_{random.randint(100,500)}"
        severity = random.choice(severity_options)
        
        if focus == "risk_analysis":
            anomalies.append({
                "symbol": symbol,
                "value": f"Risk Score: {random.uniform(0.7, 0.95):.3f}",
                "threshold": "Risk threshold: 0.6",
                "severity": severity,
                "details": f"High volatility detected with unusual trading patterns in {symbol}",
                "action_required": f"{action_prefix}: Monitor position closely and consider hedging strategy",
                "business_impact": "Potential significant losses if market moves adversely - immediate risk management required",
                "reason": f"Risk metrics exceed normal parameters for {symbol} - volatility spike detected"
            })
        elif focus == "opportunity_analysis":
            anomalies.append({
                "symbol": symbol,
                "value": f"Opportunity Score: {random.uniform(0.6, 0.9):.3f}",
                "threshold": "Opportunity threshold: 0.5",
                "severity": severity,
                "details": f"Undervalued position detected in {symbol} with strong fundamentals",
                "action_required": f"{action_prefix}: Increase position size or initiate new position in {symbol}",
                "business_impact": "Potential profit opportunity if market corrects - could generate significant returns",
                "reason": f"Technical indicators suggest {symbol} is undervalued relative to market conditions"
            })
        else:
            anomalies.append({
                "symbol": symbol,
                "value": f"Anomaly Score: {random.uniform(0.5, 0.8):.3f}",
                "threshold": "Anomaly threshold: 0.4",
                "severity": severity,
                "details": f"Unusual pattern detected in {symbol} requiring investigation",
                "action_required": f"{action_prefix}: Analyze underlying causes and take appropriate action for {symbol}",
                "business_impact": "Potential operational or strategic implications - requires immediate attention",
                "reason": f"Statistical deviation detected in {symbol} metrics - pattern analysis needed"
            })
    
    result = {
        "type": "actionable_llm_analysis",
        "query": user_query,
        "domain_detected": "options_trading",
        "interpretation": interpretation,
        "analysis_performed": f"Query-specific {focus.replace('_', ' ')} with {num_anomalies} findings",
        "total_anomalies": num_anomalies,
        "message": message,
        "summary": f"Performed {focus.replace('_', ' ')} based on your specific query '{user_query}'. Each finding includes actionable recommendations tailored to your request.",
        "anomalies": anomalies
    }
    
    print(f"✅ Analysis completed: {num_anomalies} findings for '{user_query}'")
    return json.dumps(result, indent=2)
