# 🚀 Sailo MVP - AI Database Monitoring Pipeline

**Turn raw database tables into self-running AI pipelines in 3 clicks**

## 🎯 What This MVP Does

1. **Inspect**: AI examines your Supabase database schema and sample data
2. **Auto-Plan**: AI creates a monitoring plan based on your goal
3. **Run Once**: AI analyzes data for anomalies and sends Slack alerts

## 🏁 Quick Start

### 1. Prerequisites

- Supabase project with options trading data
- Modal account (free tier works)
- Slack webhook URL (optional)

### 2. Database Setup

Run this in your Supabase SQL Editor to create sample data:

```sql
-- Copy the contents from web/setup_options_trading_db.sql
-- This creates options_trades table with sample data
```

Add anomalous data for demo:

```sql
-- Copy the contents from web/demo_anomaly_data_fixed.sql
-- This inserts data with high volatility, unusual volume, etc.
```

### 3. Environment Setup

Create `web/.env.local`:

```env
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key-here
```

### 4. Deploy Modal Backend

```bash
# Install Modal
pip install modal

# Set up Modal account
modal setup

# Create Supabase secret in Modal
modal secret create supabase-secret \
  SUPABASE_URL=https://your-project.supabase.co \
  SUPABASE_KEY=your-anon-key-here

# Deploy the backend
modal deploy modal_web_endpoints_clean.py
```

### 5. Start Frontend

```bash
cd web
npm install
npm run dev
```

Visit `http://localhost:5173` and click "🚀 MVP Demo"

## 🎬 Demo Flow

### Step 1: Inspect

- Click "🔍 1. Inspect Database"
- AI analyzes your `options_trades` table
- Shows columns, data types, and sample records

### Step 2: Auto-Plan

- Enter your goal: "watch for spikes in options volume and alert the team"
- Optionally add Slack webhook URL
- Click "🤖 2. Auto-Plan Pipeline"
- AI creates a monitoring plan with metric selection and thresholds

### Step 3: Run Once

- Click "⚡ 3. Run Once"
- AI analyzes recent data for anomalies
- Sends alert to Slack if anomalies found
- Shows results in the UI

## 📊 Sample Goals to Try

- "watch for spikes in options volume and alert the team"
- "detect high volatility options that need attention"
- "find options with unusual premium pricing"
- "monitor for gamma squeeze conditions"
- "alert on high-risk positions"

## 🔧 Modal Endpoints

The MVP creates these endpoints:

- `POST /inspect-source` - Analyze database schema and sample data
- `POST /auto-plan` - AI-generated monitoring plan
- `POST /run-once` - Execute monitoring pipeline once

## 🚨 Slack Integration

1. Create a Slack webhook:

   - Go to https://api.slack.com/messaging/webhooks
   - Create a new webhook for your channel
   - Copy the webhook URL

2. Add the webhook URL in the MVP interface

3. When anomalies are detected, you'll get a Slack message with:
   - Number of anomalies found
   - Specific details for each anomaly
   - Recommended actions

## 🎯 What Makes This Special

### Traditional Approach:

- ❌ Weeks to build ETL pipelines
- ❌ Manual threshold setting
- ❌ Expensive infrastructure ($100s/month)
- ❌ Complex rule maintenance

### Sailo MVP:

- ✅ 3 clicks to working pipeline
- ✅ AI determines thresholds
- ✅ Pay-per-use serverless ($10s/month)
- ✅ Self-adapting monitoring

## 🏗️ Architecture

```
Supabase DB → Modal (AI Analysis) → Slack Alerts
     ↑              ↑                    ↑
  Raw Data     LLM Processing      Actionable Insights
```

- **Frontend**: React + TypeScript
- **Backend**: Modal serverless functions
- **AI**: Llama 3.2 via llama.cpp (no OpenAI costs!)
- **Database**: Supabase PostgreSQL
- **Alerts**: Slack webhooks

## 🚀 Demo Script (15 seconds)

> "We inspect your schema, the AI auto-plans a monitoring pipeline, and Modal runs it serverlessly — turning raw Supabase data into action (Slack/webhooks) in seconds. Today you saw postgres→anomaly→slack; next we add multi-step branching and more actions, all with zero ops."

## 🔍 Technical Details

- **No ML expertise needed**: AI handles all analysis
- **Serverless**: Modal scales from 0 to 1000+ concurrent functions
- **Cost effective**: Only pay when pipeline runs
- **Reliable**: Built-in fallbacks if AI fails
- **Real-time**: Results in seconds, not minutes

## 🎪 Next Steps

After the demo:

1. Add scheduled runs (Modal cron)
2. Multiple data sources
3. Custom action types (email, webhooks, API calls)
4. Multi-step pipelines
5. Dashboard for monitoring multiple pipelines

---

**Built with Modal's serverless platform - making AI monitoring pipelines accessible to everyone.**
