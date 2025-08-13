#!/usr/bin/env python3
"""
Demo script showing the complete Supabase Agent MVP flow.
This simulates the experience a user would have.
"""

def demo_narrative():
    """Show the complete demo flow with sample data."""
    
    print("🎪 SUPABASE AGENT MVP - DEMO FLOW")
    print("=" * 50)
    
    print("\n📋 SCENARIO:")
    print("• You're a startup founder with a Supabase database")
    print("• You have metrics flowing in but no monitoring")
    print("• You need alerts but don't want to build ETL pipelines")
    print("• Solution: AI-powered monitoring that sets itself up!")
    
    print("\n1️⃣ INSPECT DATABASE")
    print("   → Connect to Supabase Postgres")
    print("   → Analyze schema automatically")
    print("   → Find timestamp + numeric columns")
    print("   → Calculate statistics")
    
    sample_schema = {
        "columns": [
            {"name": "ts", "dtype": "timestamptz", "is_datetime": True},
            {"name": "cpu_usage", "dtype": "float", "is_numeric": True}, 
            {"name": "memory_usage", "dtype": "float", "is_numeric": True},
            {"name": "response_time", "dtype": "float", "is_numeric": True}
        ],
        "sample": [
            {"ts": "2024-01-15 10:00:00", "cpu_usage": 45.2, "memory_usage": 67.8, "response_time": 120},
            {"ts": "2024-01-15 10:05:00", "cpu_usage": 52.1, "memory_usage": 71.2, "response_time": 95},
            {"ts": "2024-01-15 10:10:00", "cpu_usage": 89.4, "memory_usage": 85.1, "response_time": 450}  # SPIKE!
        ]
    }
    
    print(f"   ✅ Found {len(sample_schema['columns'])} columns")
    print(f"   ✅ Detected {len([c for c in sample_schema['columns'] if c['is_numeric']])} metrics to monitor")
    
    print("\n2️⃣ AI AUTO-PLANNING")
    print("   → User goal: 'Alert me when CPU usage spikes'")
    print("   → AI (Phi-4) analyzes schema + goal")
    print("   → Generates structured monitoring plan")
    
    ai_plan = {
        "metric": "cpu_usage",
        "timestamp_col": "ts",
        "method": "zscore", 
        "threshold": 3.0,
        "ew_span": 24,
        "schedule_minutes": 15,
        "action": "slack"
    }
    
    print(f"   ✅ AI chose to monitor: {ai_plan['metric']}")
    print(f"   ✅ Detection method: {ai_plan['method']} with threshold {ai_plan['threshold']}")
    print(f"   ✅ Check every {ai_plan['schedule_minutes']} minutes")
    
    print("\n3️⃣ ANOMALY DETECTION")
    print("   → Fetch recent data from Supabase")
    print("   → Calculate exponential weighted mean")
    print("   → Compute z-scores for each point")
    print("   → Identify anomalies above threshold")
    
    detection_result = {
        "count": 1,
        "latest": {
            "ts": "2024-01-15 10:10:00",
            "value": 89.4,
            "z_score": 4.2,
            "is_anomaly": True
        }
    }
    
    print(f"   🚨 Found {detection_result['count']} anomalies!")
    print(f"   🚨 Latest spike: {detection_result['latest']['value']}% CPU (z-score: {detection_result['latest']['z_score']})")
    
    print("\n4️⃣ SLACK ALERT")
    print("   → Generate matplotlib chart")
    print("   → Create formatted message")
    print("   → Post to Slack with context")
    
    slack_message = """🚨 *CPU Usage Alert!* 🚨

Found 1 anomaly in the last monitoring window.
Latest data point: 89.4% (z-score: 4.20)
Time: 2024-01-15 10:10:00

Monitoring plan: cpu_usage with threshold z=3.0

[Chart showing CPU timeline with red spike at 10:10]"""
    
    print("   ✅ Slack message sent:")
    for line in slack_message.split('\n'):
        print(f"      {line}")
    
    print("\n🎯 DEMO COMPLETE - FROM DATABASE TO ACTION IN SECONDS!")
    print("\n🚀 KEY BENEFITS:")
    print("   • Zero ETL pipeline building")
    print("   • AI understands your data automatically") 
    print("   • Runs serverlessly on Modal (no ops)")
    print("   • Scales from 1 to 1000 databases")
    print("   • No external API costs (local LLMs)")
    print("   • Always has fallback plans (never breaks)")

def modal_value_prop():
    """Explain specifically how Modal enables this solution."""
    
    print("\n" + "=" * 50)
    print("🚀 HOW MODAL MAKES THIS POSSIBLE")
    print("=" * 50)
    
    print("\n🔧 TRADITIONAL APPROACH (Without Modal):")
    print("   ❌ Set up Kubernetes cluster")
    print("   ❌ Configure auto-scaling")
    print("   ❌ Manage model serving infrastructure") 
    print("   ❌ Handle deployment pipelines")
    print("   ❌ Pay $100s/month for 24/7 servers")
    print("   ❌ Weeks of DevOps work")
    print("   ❌ $1000s in OpenAI API costs")
    
    print("\n✅ MODAL APPROACH (Our Solution):")
    print("   🚀 Deploy: `modal deploy modal_app.py` (30 seconds)")
    print("   🚀 Scale: Automatic (0 to 1000 concurrent functions)")
    print("   🚀 Cost: Pay-per-second usage only")
    print("   🚀 LLMs: Run Phi-4/Qwen locally (no API costs)")
    print("   🚀 Ops: Zero (Modal handles everything)")
    print("   🚀 Reliability: Built-in retries + fallbacks")
    
    print("\n💰 COST COMPARISON:")
    print("   Traditional: $500-2000/month (servers + APIs)")
    print("   Modal: $10-50/month (usage-based)")
    
    print("\n⚡ SPEED COMPARISON:")
    print("   Traditional: 2-4 weeks to build + deploy")
    print("   Modal: 2-4 hours to build + deploy")
    
    print("\n🎯 MODAL'S SECRET SAUCE:")
    print("   • Serverless functions that start in milliseconds")
    print("   • Built-in GPU/CPU auto-scaling")  
    print("   • Model weights cached across runs")
    print("   • Container image optimization")
    print("   • Persistent volumes for model storage")
    print("   • Built-in secrets management")

if __name__ == "__main__":
    demo_narrative()
    modal_value_prop()
