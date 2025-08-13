# Slack Integration Setup Guide

Your anomaly detection system now supports Slack notifications! Here's how to set it up:

## 🚀 Quick Setup (Recommended)

### Method 1: Webhook URL (Easiest)

1. **Create a Slack Incoming Webhook:**

   - Go to https://api.slack.com/messaging/webhooks
   - Click "Create your Slack app" → "From scratch"
   - Name your app (e.g., "Anomaly Detector") and select your workspace
   - Go to "Incoming Webhooks" → Toggle "On"
   - Click "Add New Webhook to Workspace"
   - Choose the channel (e.g., #alerts) and click "Allow"
   - Copy the webhook URL (starts with `https://hooks.slack.com/services/...`)

2. **Test the Integration:**

   ```bash
   modal run modal_app_fixed.py::test_slack_integration --webhook-url "YOUR_WEBHOOK_URL"
   ```

3. **Run Anomaly Detection with Slack:**
   ```bash
   modal run modal_app_fixed.py::test_anomaly_with_slack --webhook-url "YOUR_WEBHOOK_URL"
   ```

### Method 2: Bot Token (More Features)

1. **Create a Slack App:**

   - Go to https://api.slack.com/apps → "Create New App" → "From scratch"
   - Name your app and select your workspace

2. **Configure Bot Permissions:**

   - Go to "OAuth & Permissions"
   - Under "Scopes" → "Bot Token Scopes", add:
     - `chat:write`
     - `chat:write.public`
   - Click "Install to Workspace" → "Allow"
   - Copy the "Bot User OAuth Token" (starts with `xoxb-`)

3. **Test the Integration:**
   ```bash
   modal run modal_app_fixed.py::test_slack_integration --bot-token "YOUR_BOT_TOKEN"
   ```

## 🧪 Testing Commands

### Test Slack Connection Only

```bash
# With webhook
modal run modal_app_fixed.py::test_slack_integration --webhook-url "YOUR_WEBHOOK_URL"

# With bot token
modal run modal_app_fixed.py::test_slack_integration --bot-token "YOUR_BOT_TOKEN"

# Custom channel
modal run modal_app_fixed.py::test_slack_integration --webhook-url "YOUR_WEBHOOK_URL" --channel "#custom-channel"
```

### Test Anomaly Detection with Slack

```bash
# Basic test with webhook
modal run modal_app_fixed.py::test_anomaly_with_slack --webhook-url "YOUR_WEBHOOK_URL"

# Test different metric
modal run modal_app_fixed.py::test_anomaly_with_slack --webhook-url "YOUR_WEBHOOK_URL" --metric "volume"

# Custom channel
modal run modal_app_fixed.py::test_anomaly_with_slack --webhook-url "YOUR_WEBHOOK_URL" --channel "#trading-alerts"
```

## 📱 What You'll See

When anomalies are detected, you'll receive Slack messages like:

```
🚨 Anomaly Alert 🚨

Metric: implied_volatility
Anomalies Found: 3 out of 150 records
Threshold: Z-score > 2.0

Top Anomalies:
1. AAPL - Value: 0.8542 (Z-score: 3.21)
2. MSFT - Value: 0.1234 (Z-score: 2.87)
3. GOOGL - Value: 0.9876 (Z-score: 2.45)

Analysis completed at 2024-01-15 14:30:25
```

## 🔧 Configuration Options

The Slack integration supports these parameters:

- **webhook_url**: Slack webhook URL (Method 1)
- **bot_token**: Slack bot token (Method 2)
- **channel**: Target channel (default: #alerts)
- **username**: Bot display name (default: "Anomaly Bot")
- **send_slack_notifications**: Enable/disable notifications (default: false)

## 🔗 Integration in Your Code

To use Slack notifications in your own functions:

```python
# Call the anomaly detection with Slack enabled
result = simple_anomaly_detection.remote(
    supabase_url="your_supabase_url",
    supabase_key="your_supabase_key",
    metric="implied_volatility",
    slack_webhook_url="your_webhook_url",  # or use slack_bot_token
    slack_channel="#alerts",
    send_slack_notifications=True
)

# Check if Slack notification was sent
if 'slack_notification' in result:
    if result['slack_notification']['status'] == 'success':
        print("Slack notification sent!")
    else:
        print(f"Slack failed: {result['slack_notification']['error']}")
```

## 🛠️ Environment Variables

For production use, set these in your environment:

```bash
# In your .env file or environment
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
# OR
SLACK_BOT_TOKEN=xoxb-your-bot-token-here

SLACK_CHANNEL=#alerts
SLACK_USERNAME=Anomaly Bot
```

## 🚨 Troubleshooting

**Common Issues:**

1. **"Invalid webhook URL"** → Check the URL format and ensure it starts with `https://hooks.slack.com/services/`

2. **"Channel not found"** → Make sure the bot is added to the channel or use a public channel

3. **"Not authorized"** → For bot tokens, ensure proper scopes (`chat:write`, `chat:write.public`)

4. **"No anomalies found"** → This is normal! The system only sends notifications when anomalies are detected

**Need Help?**

- Check the Modal logs for detailed error messages
- Test with the `test_slack_integration` function first
- Verify your Slack app permissions and installation

## 🎉 You're All Set!

Your anomaly detection system will now automatically send Slack notifications when it finds unusual patterns in your data. The system is designed to be non-intrusive - it only sends messages when actual anomalies are detected.
