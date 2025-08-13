import { useState } from 'react'

interface MonitoringScenario {
  name: string
  goal: string
  description: string
}

interface MonitoringResult {
  success: boolean
  scenario: string
  total_records: number
  results: {
    type: string
    query?: string
    domain_detected?: string
    interpretation?: string
    analysis_performed?: string
    total_anomalies: number
    message: string
    summary?: string
    anomalies?: Array<{
      symbol: string
      value: string
      threshold: string
      severity: string
      details: string
      action_required?: string
      business_impact?: string
      reason?: string
      timestamp?: string
      contract_type?: string
      [key: string]: any
    }>
  }
  slack_sent: boolean
  timestamp: string
  error?: string
}

// Pre-defined monitoring scenarios for options trading
const TRADING_SCENARIOS: MonitoringScenario[] = [
  {
    name: "volatility_spike",
    goal: "detect implied volatility spikes above 50% increase in the last hour",
    description: "Monitor for sudden volatility increases that might indicate market stress"
  },
  {
    name: "unusual_volume",
    goal: "detect options volume 300% above average for specific strikes",
    description: "Identify unusual trading activity that might indicate insider information"
  },
  {
    name: "risk_exposure",
    goal: "monitor portfolio delta exposure exceeding risk limits",
    description: "Track portfolio risk metrics to prevent excessive exposure"
  },
  {
    name: "price_anomaly",
    goal: "detect options pricing anomalies compared to theoretical values",
    description: "Find mispriced options that might present arbitrage opportunities"
  },
  {
    name: "gamma_squeeze",
    goal: "detect high gamma exposure that might cause price acceleration",
    description: "Monitor for gamma squeeze conditions in popular strikes"
  }
]

export function AIMonitoring() {
  const [selectedScenario, setSelectedScenario] = useState<string>('')
  const [customGoal, setCustomGoal] = useState('')
  const [slackWebhook, setSlackWebhook] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<MonitoringResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const runMonitoring = async (scenarioName?: string, goal?: string) => {
    try {
      setLoading(true)
      setError(null)
      setResult(null)

      // Get Supabase credentials from environment
      const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
      const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY

      if (!supabaseUrl || !supabaseKey) {
        throw new Error('Supabase credentials not configured')
      }

      // Prepare the request payload for Modal endpoint
      const payload = {
        scenario: scenarioName || 'custom',
        customQuery: goal || customGoal,
        slackWebhook: slackWebhook || undefined
      }

      // Call Modal web endpoint
      const response = await fetch('https://sailo--sailo-web-api-run-monitoring-scenario.modal.run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const monitoringResult = await response.json()
      
      if (!monitoringResult.success) {
        throw new Error(monitoringResult.error || 'Monitoring failed')
      }
      
      setResult(monitoringResult)

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
      console.error('Monitoring error:', err)
    } finally {
      setLoading(false)
    }
  }

  const runPredefinedScenario = () => {
    if (!selectedScenario) {
      setError('Please select a monitoring scenario')
      return
    }
    runMonitoring(selectedScenario)
  }

  const runCustomGoal = () => {
    if (!customGoal.trim()) {
      setError('Please enter a custom monitoring goal')
      return
    }
    runMonitoring(undefined, customGoal)
  }

  return (
    <div className="ai-monitoring">
      <h2>🤖 AI-Powered Options Trading Monitoring</h2>
      <p>Use AI to automatically detect patterns and anomalies in your options trading data</p>

      {/* Slack Configuration */}
      <div className="config-section">
        <h3>Configuration</h3>
        <div className="slack-config">
          <label>
            Slack Webhook URL (optional):
            <input
              type="url"
              value={slackWebhook}
              onChange={(e) => setSlackWebhook(e.target.value)}
              placeholder="https://hooks.slack.com/services/..."
            />
          </label>
        </div>
      </div>

      {/* Pre-defined Scenarios */}
      <div className="scenarios-section">
        <h3>📊 Pre-defined Monitoring Scenarios</h3>
        <div className="scenarios-grid">
          {TRADING_SCENARIOS.map((scenario) => (
            <div 
              key={scenario.name}
              className={`scenario-card ${selectedScenario === scenario.name ? 'selected' : ''}`}
              onClick={() => setSelectedScenario(scenario.name)}
            >
              <h4>{scenario.name.replace('_', ' ').toUpperCase()}</h4>
              <p className="scenario-description">{scenario.description}</p>
              <p className="scenario-goal"><strong>Goal:</strong> {scenario.goal}</p>
            </div>
          ))}
        </div>
        
        <button 
          onClick={runPredefinedScenario}
          disabled={loading || !selectedScenario}
          className="run-scenario-btn"
        >
          {loading ? 'Running AI Analysis...' : 'Run Selected Scenario'}
        </button>
      </div>

      {/* Custom Goal Input */}
      <div className="custom-section">
        <h3>🎯 Custom AI Analysis</h3>
        <p>Ask your AI anything about your data - it will automatically detect the domain and provide actionable insights!</p>
        
        <div className="example-queries">
          <h4>Try these example queries:</h4>
          <div className="query-examples">
            <button 
              onClick={() => setCustomGoal("what should I watch out for in my options data?")}
              className="example-query-btn"
            >
              "What should I watch out for?"
            </button>
            <button 
              onClick={() => setCustomGoal("find undervalued opportunities")}
              className="example-query-btn"
            >
              "Find undervalued opportunities"
            </button>
            <button 
              onClick={() => setCustomGoal("show me high-risk positions")}
              className="example-query-btn"
            >
              "Show me high-risk positions"
            </button>
            <button 
              onClick={() => setCustomGoal("what looks suspicious in my data?")}
              className="example-query-btn"
            >
              "What looks suspicious?"
            </button>
          </div>
        </div>
        
        <div className="input-group">
          <textarea
            value={customGoal}
            onChange={(e) => setCustomGoal(e.target.value)}
            placeholder="Ask your AI anything: 'what should I watch out for?', 'find undervalued opportunities', 'show me risks', etc."
            rows={3}
          />
          <button 
            onClick={runCustomGoal} 
            disabled={loading}
            className="run-button custom-button"
          >
            {loading ? '🔄 AI Analyzing...' : '🤖 Ask AI'}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-section">
          <h3>❌ Error</h3>
          <p>{error}</p>
          <details>
            <summary>Troubleshooting</summary>
            <ul>
              <li>Make sure your options_trades table exists in Supabase</li>
              <li>Check that your Supabase credentials are correct</li>
              <li>Ensure Modal deployment is running</li>
              <li>Verify Slack webhook URL format if provided</li>
            </ul>
          </details>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="results-section">
          <h3>🤖 AI Analysis Results</h3>
          
          <div className="result-summary">
            {result.results.query && <h4>Query: "{result.results.query}"</h4>}
            {result.results.domain_detected && (
              <div className="domain-info">
                <p><strong>Domain Detected:</strong> {result.results.domain_detected.replace('_', ' ').toUpperCase()}</p>
                {result.results.interpretation && <p><em>{result.results.interpretation}</em></p>}
              </div>
            )}
            
            <div className="analysis-info">
              <p><strong>Analysis Type:</strong> {result.results.type}</p>
              <p><strong>Records Analyzed:</strong> {result.total_records}</p>
              {result.results.analysis_performed && <p><strong>Method:</strong> {result.results.analysis_performed}</p>}
            </div>
            
            <div className="anomaly-summary">
              <div className="anomaly-count">
                <span className="count">{result.results.total_anomalies}</span>
                <span className="label">Actionable Insights Found</span>
              </div>
              
              {result.results.anomalies && result.results.anomalies.length > 0 && (
                <div className="anomalies-list">
                  <h5>🎯 Actionable Recommendations:</h5>
                  {result.results.anomalies.map((anomaly, index) => (
                    <div key={index} className="anomaly-item actionable">
                      <div className="anomaly-header">
                        <h6><strong>{anomaly.symbol}</strong></h6>
                        <span className={`severity-badge ${anomaly.severity.toLowerCase()}`}>
                          {anomaly.severity}
                        </span>
                      </div>
                      
                      <div className="anomaly-details">
                        <p><strong>Finding:</strong> {anomaly.value}</p>
                        <p><strong>Threshold:</strong> {anomaly.threshold}</p>
                        <p><strong>Details:</strong> {anomaly.details}</p>
                        
                        {anomaly.action_required && (
                          <div className="action-required">
                            <p><strong>🚨 ACTION REQUIRED:</strong></p>
                            <p className="action-text">{anomaly.action_required}</p>
                          </div>
                        )}
                        
                        {anomaly.business_impact && (
                          <div className="business-impact">
                            <p><strong>💼 Business Impact:</strong> {anomaly.business_impact}</p>
                          </div>
                        )}
                        
                        {anomaly.reason && (
                          <div className="reasoning">
                            <p><strong>📊 Analysis:</strong> {anomaly.reason}</p>
                          </div>
                        )}
                        
                        {anomaly.timestamp && (
                          <p className="timestamp"><small>Detected: {new Date(anomaly.timestamp).toLocaleString()}</small></p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="alert-status">
              <p>
                <strong>Slack Alert:</strong> 
                <span className={result.slack_sent ? 'success' : 'warning'}>
                  {result.slack_sent ? ' ✅ Sent' : ' ⚠️ Not sent'}
                </span>
              </p>
            </div>

            <div className="ai-summary">
              <h5>🧠 AI Analysis Summary:</h5>
              <div className="summary-content">
                <p className="main-message">{result.results.message}</p>
                {result.results.summary && (
                  <div className="detailed-summary">
                    <p><strong>Detailed Analysis:</strong></p>
                    <p>{result.results.summary}</p>
                  </div>
                )}
              </div>
            </div>

          </div>

          <details className="technical-details">
            <summary>Technical Details</summary>
            <div className="timestamp-details">
              <h5>Analysis Timestamp:</h5>
              <pre>{new Date(result.timestamp).toLocaleString()}</pre>
            </div>
            <div className="scenario-details">
              <h5>Scenario Configuration:</h5>
              <pre>Type: {result.results.type}
Records: {result.total_records}
Anomalies: {result.results.total_anomalies}</pre>
            </div>
          </details>
        </div>
      )}

      {/* Instructions */}
      <div className="instructions-section">
        <h3>🚀 How It Works</h3>
        <ol>
          <li><strong>Select a scenario</strong> or enter a custom goal</li>
          <li><strong>AI analyzes</strong> your Supabase options trading data</li>
          <li><strong>Detects patterns</strong> and anomalies automatically</li>
          <li><strong>Sends alerts</strong> via Slack (if configured)</li>
          <li><strong>Provides insights</strong> with charts and analysis</li>
        </ol>
        
        <div className="next-steps">
          <h4>Next Steps:</h4>
          <ul>
            <li>Set up automated scheduling in Modal</li>
            <li>Configure multiple Slack channels for different alert types</li>
            <li>Add custom actions (email, API calls, trading halts)</li>
            <li>Integrate with your existing trading systems</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default AIMonitoring
