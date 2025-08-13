# 🚀 Sailo MVP - Quick Start Guide

## ✅ What's Ready

You now have a **clean, focused MVP** with only the essentials:

### 📁 Files Structure

```
Sailo/
├── modal_mvp.py           # Clean Modal backend (3 endpoints)
├── requirements.txt       # Minimal dependencies
├── test_mvp_endpoints.py  # Test script
└── web/
    ├── src/App.tsx        # Simple app wrapper
    └── components/
        ├── MVPDemo.tsx    # 3-button demo interface
        └── MVPDemo.css    # Clean styling
```

### 🎯 What It Does

1. **🔍 Inspect** - AI analyzes your database schema
2. **🤖 Auto-Plan** - AI creates monitoring plan
3. **⚡ Run Once** - AI finds anomalies + sends Slack alerts

## 🏁 Quick Deploy & Test

### Step 1: Setup Environment

```bash
# Make sure you have your Supabase credentials
# Create web/.env.local:
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

### Step 2: Deploy Modal Backend

```bash
# Install Modal if needed
pip install modal

# Set up Modal account
modal setup

# Create Supabase secret
modal secret create supabase-secret \
  SUPABASE_URL=https://your-project.supabase.co \
  SUPABASE_KEY=your-anon-key

# Deploy the MVP backend
modal deploy modal_mvp.py
```

### Step 3: Test Frontend (Already Running!)

The frontend should be running at: `http://localhost:5173`

**What you'll see:**

- Clean interface with goal pre-filled
- 3-button flow: Inspect → Auto-Plan → Run Once
- Progress bar showing current step
- Results display with anomaly details

### Step 4: Test the Flow

1. Open `http://localhost:5173`
2. The goal is pre-filled: "watch for spikes in options volume and alert the team"
3. Click **🔍 1. Inspect Database** (should work with your Supabase data)
4. Click **🤖 2. Auto-Plan Pipeline** (AI creates plan)
5. Click **⚡ 3. Run Once** (AI finds anomalies)

## 🧪 Test Endpoints Manually

```bash
# Test all endpoints
python test_mvp_endpoints.py
```

Expected output:

- ✅ Inspect Source: PASSED
- ✅ Auto Plan: PASSED
- ✅ Run Once: PASSED

## 🎬 Demo Ready!

Your MVP is now:

- **Simplified** - Only essential files
- **Clean** - No syntax errors or unused code
- **Focused** - 3-button demo flow
- **Fast** - Minimal dependencies
- **Reliable** - Fallback plans if AI fails

## 🚨 If Something Breaks

### Frontend won't load:

```bash
cd web
npm install
npm run dev
```

### Modal deployment fails:

- Check: `modal setup` completed
- Check: Supabase secret exists: `modal secret list`
- Try: `modal app list` to see deployed apps

### No data in database:

```bash
# Run in Supabase SQL Editor:
# 1. Copy contents from web/setup_options_trading_db.sql
# 2. Copy contents from web/demo_anomaly_data_fixed.sql
```

## 🎯 Success Criteria

✅ Frontend loads cleanly at `http://localhost:5173`  
✅ All 3 buttons work without errors  
✅ Inspect shows real database data  
✅ Auto-plan generates JSON config  
✅ Run once finds 2-5 anomalies with details

**You're ready to demo! 🎉**
