# ✅ Sailo MVP Deployment Checklist

## 🎯 Pre-Demo Setup

### 1. Database Setup

- [ ] Supabase project created
- [ ] Run `web/setup_options_trading_db.sql` in Supabase SQL Editor
- [ ] Run `web/demo_anomaly_data_fixed.sql` for anomaly data
- [ ] Verify data exists: `SELECT COUNT(*) FROM options_trades;`

### 2. Modal Deployment

- [ ] Modal account set up (`modal setup`)
- [ ] Create Supabase secret:
  ```bash
  modal secret create supabase-secret \
    SUPABASE_URL=https://your-project.supabase.co \
    SUPABASE_KEY=your-anon-key
  ```
- [ ] Deploy backend: `modal deploy modal_web_endpoints_clean.py`
- [ ] Verify deployment: Check Modal dashboard for "sailo-web-api"

### 3. Frontend Setup

- [ ] Create `web/.env.local`:
  ```env
  VITE_SUPABASE_URL=https://your-project.supabase.co
  VITE_SUPABASE_ANON_KEY=your-anon-key-here
  ```
- [ ] Install dependencies: `cd web && npm install`
- [ ] Start dev server: `npm run dev`
- [ ] Verify app loads at `http://localhost:5173`

### 4. Slack Setup (Optional)

- [ ] Create Slack webhook at https://api.slack.com/messaging/webhooks
- [ ] Test webhook with curl or Postman
- [ ] Copy webhook URL for demo

## 🧪 Testing

### Automated Test

```bash
python test_mvp_endpoints.py
```

Expected output:

- ✅ Inspect Source: PASSED
- ✅ Auto Plan: PASSED
- ✅ Run Once: PASSED

### Manual Test

1. Open `http://localhost:5173`
2. Click "🚀 MVP Demo" tab
3. Test the 3-button flow:
   - [ ] Step 1: Inspect works, shows table data
   - [ ] Step 2: Auto-plan works, generates JSON plan
   - [ ] Step 3: Run once works, finds anomalies

### Slack Test (if webhook configured)

- [ ] Add webhook URL in MVP interface
- [ ] Run pipeline and verify Slack message received
- [ ] Check message formatting and anomaly details

## 🎬 Demo Preparation

### Performance Check

- [ ] Inspect response < 5 seconds
- [ ] Auto-plan response < 10 seconds
- [ ] Run once response < 15 seconds
- [ ] Frontend loads quickly

### Content Check

- [ ] Sample goal pre-filled: "watch for spikes in options volume and alert the team"
- [ ] Inspect shows realistic data (20+ records)
- [ ] Auto-plan generates sensible monitoring config
- [ ] Run once finds 2-5 anomalies with details

### Backup Plan

- [ ] Screenshots of successful run saved
- [ ] Alternative demo data ready
- [ ] Offline version works if needed
- [ ] Explanation ready if live demo fails

## 🎯 Demo Day

### 5 Minutes Before

- [ ] Close all unnecessary browser tabs
- [ ] Test complete flow once more
- [ ] Check Modal endpoints are responsive
- [ ] Verify Slack webhook if using

### During Demo

- [ ] Start with problem statement (30s)
- [ ] Show 3-button flow (90s)
- [ ] Explain technology (30s)
- [ ] Close with vision (15s)

### Key Talking Points

1. **3 clicks** - emphasize simplicity
2. **AI-powered** - highlight intelligence
3. **Serverless** - mention cost/scale benefits
4. **Zero ops** - no maintenance needed

## 🔧 Troubleshooting

### Common Issues

**Frontend won't load:**

- Check `.env.local` file exists and has correct Supabase URLs
- Verify `npm install` ran successfully
- Try `npm run dev` again

**Inspect endpoint fails:**

- Check Modal deployment: `modal app list`
- Verify Supabase secret: `modal secret list`
- Test database connection directly in Supabase

**Auto-plan takes too long:**

- LLM might be starting up (first call can be slow)
- Check Modal logs for errors
- Fallback plan should trigger after timeout

**Run once finds no anomalies:**

- Verify anomaly data was inserted: Check `demo_anomaly_data_fixed.sql`
- Try different goal text
- Check LLM is processing data correctly

### Emergency Fallbacks

1. Use screenshots instead of live demo
2. Show code and explain architecture
3. Focus on problem/solution rather than demo
4. Use existing AI Monitoring component instead

---

## 🎉 Success Criteria

✅ **Demo works end-to-end**
✅ **Response times under 15 seconds total**  
✅ **Finds 2-5 meaningful anomalies**
✅ **Explanation resonates with audience**
✅ **Backup plan ready if needed**

**Remember: The goal is to show the magic of AI + serverless for database monitoring!**
