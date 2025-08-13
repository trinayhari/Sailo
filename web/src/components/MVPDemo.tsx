import { useState } from "react";
import "./MVPDemo.css";

interface InspectionData {
  table: string;
  total_records: number;
  total_columns: number;
  columns: {
    all: string[];
    numeric: string[];
    datetime: string[];
  };
  sample_data: any[];
  stats: string;
}

interface Plan {
  metric: string;
  timestamp_col: string;
  method: string;
  threshold: string;
  schedule_minutes: number;
  action: string;
  action_config: {
    webhook_url: string;
  };
}

interface RunResults {
  type: string;
  domain_detected: string;
  summary: string;
  total_anomalies: number;
  anomalies: Array<{
    identifier: string;
    value: string;
    severity: string;
    details: string;
    action_required: string;
    business_impact: string;
    reason: string;
  }>;
  insights: string[];
  recommendations: string[];
}

export function MVPDemo() {
  // State management
  const [step, setStep] = useState<
    "start" | "inspected" | "planned" | "completed"
  >("start");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Data state
  const [inspectionData, setInspectionData] = useState<InspectionData | null>(
    null
  );
  const [plan, setPlan] = useState<Plan | null>(null);
  const [results, setResults] = useState<RunResults | null>(null);
  const [slackSent, setSlackSent] = useState(false);

  // User inputs
  const [goal, setGoal] = useState(
    "Find profitable investment opportunities in our trading data"
  );
  const [tableName, setTableName] = useState("options_trades");
  const [slackWebhook, setSlackWebhook] = useState("");

  const BASE_URL = "https://sailo--sailo-mvp-fastapi-app.modal.run";

  const handleInspect = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${BASE_URL}/inspect-source`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ table_name: tableName }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || "Inspection failed");
      }

      setInspectionData(result.inspection);
      setStep("inspected");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inspection failed");
    } finally {
      setLoading(false);
    }
  };

  const handleAutoPlan = async () => {
    if (!inspectionData) return;

    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${BASE_URL}/auto-plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          goal,
          inspection_data: inspectionData,
          slack_webhook: slackWebhook || undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || "Auto-planning failed");
      }

      setPlan(result.plan);
      setStep("planned");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Auto-planning failed");
    } finally {
      setLoading(false);
    }
  };

  const handleRunOnce = async () => {
    if (!plan) return;

    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${BASE_URL}/run-once`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          plan,
          table_name: tableName,
          goal: goal,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || "Pipeline run failed");
      }

      setResults(result.results);
      setSlackSent(result.slack_sent);
      setStep("completed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Pipeline run failed");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setStep("start");
    setInspectionData(null);
    setPlan(null);
    setResults(null);
    setSlackSent(false);
    setError(null);
  };

  return (
    <div className="mvp-demo">
      <div className="demo-header">
        <h1>🚀 Sailo MVP Demo</h1>
        <p>Turn raw database tables into self-running AI pipelines</p>

        <div className="progress-bar">
          <div
            className={`step ${
              step === "start" ? "current" : step !== "start" ? "completed" : ""
            }`}
          >
            1. Inspect
          </div>
          <div
            className={`step ${
              step === "inspected"
                ? "current"
                : step === "planned" || step === "completed"
                ? "completed"
                : ""
            }`}
          >
            2. Auto-Plan
          </div>
          <div
            className={`step ${
              step === "planned"
                ? "current"
                : step === "completed"
                ? "completed"
                : ""
            }`}
          >
            3. Run Once
          </div>
        </div>
      </div>

      {/* Configuration Section */}
      <div className="config-section">
        <h3>🎯 Analysis Goal</h3>
        <div className="input-group">
          <label>
            Database Table Name:
            <input
              type="text"
              value={tableName}
              onChange={(e) => setTableName(e.target.value)}
              placeholder="Enter table name (e.g., options_trades, sales_data, user_activity)"
              disabled={step !== "start"}
              className="demo-input"
            />
          </label>
          <label>
            What would you like to analyze?
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="Describe what you want to analyze or monitor in your data..."
              disabled={step !== "start"}
              rows={3}
              className="demo-input"
            />
          </label>
          <small>
            Be specific about what patterns, trends, or issues you want the AI
            to look for
          </small>
        </div>

        {/* Example Goals */}
        <div className="example-goals">
          <h4>💡 Try these example goals:</h4>
          <div className="goal-examples">
            <button
              onClick={() =>
                setGoal(
                  "Find unusual patterns and anomalies in my data that need attention"
                )
              }
              className="example-goal-btn"
              disabled={step !== "start"}
            >
              🔍 Find unusual patterns
            </button>
            <button
              onClick={() =>
                setGoal(
                  "Identify potential business risks and opportunities from this data"
                )
              }
              className="example-goal-btn"
              disabled={step !== "start"}
            >
              📊 Business insights
            </button>
            <button
              onClick={() =>
                setGoal(
                  "Monitor for data quality issues, outliers, and inconsistencies"
                )
              }
              className="example-goal-btn"
              disabled={step !== "start"}
            >
              🔧 Data quality check
            </button>
            <button
              onClick={() =>
                setGoal("Analyze trends and predict what might happen next")
              }
              className="example-goal-btn"
              disabled={step !== "start"}
            >
              📈 Trend analysis
            </button>
            <button
              onClick={() =>
                setGoal("Find high-value opportunities and actionable insights")
              }
              className="example-goal-btn"
              disabled={step !== "start"}
            >
              💎 Value discovery
            </button>
            <button
              onClick={() =>
                setGoal(
                  "Detect security concerns, fraud, or suspicious activity"
                )
              }
              className="example-goal-btn"
              disabled={step !== "start"}
            >
              🛡️ Security analysis
            </button>
          </div>
        </div>
        <div className="input-group">
          <label>
            Slack Webhook (optional):
            <input
              type="url"
              value={slackWebhook}
              onChange={(e) => setSlackWebhook(e.target.value)}
              placeholder="https://hooks.slack.com/services/..."
              disabled={step !== "start" && step !== "inspected"}
            />
          </label>
        </div>
      </div>

      {/* Actions */}
      <div className="actions-section">
        {step === "start" && (
          <button
            onClick={handleInspect}
            disabled={loading}
            className="primary-button inspect-btn"
          >
            {loading ? "🔍 Inspecting..." : "🔍 1. Inspect Database"}
          </button>
        )}

        {step === "inspected" && (
          <button
            onClick={handleAutoPlan}
            disabled={loading}
            className="primary-button plan-btn"
          >
            {loading ? "🤖 Planning..." : "🤖 2. Auto-Plan Pipeline"}
          </button>
        )}

        {step === "planned" && (
          <button
            onClick={handleRunOnce}
            disabled={loading}
            className="primary-button run-btn"
          >
            {loading ? "⚡ Running..." : "⚡ 3. Run Once"}
          </button>
        )}

        {step === "completed" && (
          <div className="completed-actions">
            <button onClick={reset} className="secondary-button reset-btn">
              🔄 Try Different Goal
            </button>
            <button
              onClick={() => {
                setGoal("");
                setStep("start");
              }}
              className="secondary-button clear-btn"
            >
              ✨ Start Fresh
            </button>
          </div>
        )}

        {step !== "start" && step !== "completed" && (
          <button onClick={reset} className="tertiary-button back-btn">
            ← Change Goal
          </button>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-display">
          <h3>❌ Error</h3>
          <p>{error}</p>
        </div>
      )}

      {/* Results Display */}
      {inspectionData && (
        <div className="results-section">
          <h3>📊 Database Inspection</h3>
          <div className="inspection-results">
            <p>
              <strong>Table:</strong> {inspectionData.table}
            </p>
            <p>
              <strong>Records:</strong> {inspectionData.total_records}
            </p>
            <p>
              <strong>Columns:</strong> {inspectionData.total_columns}
            </p>
            <p>
              <strong>Numeric Columns:</strong>{" "}
              {inspectionData.columns.numeric.join(", ")}
            </p>
            <p>
              <strong>Datetime Columns:</strong>{" "}
              {inspectionData.columns.datetime.join(", ")}
            </p>

            <details>
              <summary>Sample Data (first 3 records)</summary>
              <pre>
                {JSON.stringify(
                  inspectionData.sample_data.slice(0, 3),
                  null,
                  2
                )}
              </pre>
            </details>
          </div>
        </div>
      )}

      {plan && (
        <div className="results-section">
          <h3>🤖 AI-Generated Plan</h3>
          <div className="plan-results">
            <div className="plan-grid">
              <div>
                <strong>Metric to Monitor:</strong> {plan.metric}
              </div>
              <div>
                <strong>Timestamp Column:</strong> {plan.timestamp_col}
              </div>
              <div>
                <strong>Method:</strong> {plan.method}
              </div>
              <div>
                <strong>Schedule:</strong> Every {plan.schedule_minutes} minutes
              </div>
              <div>
                <strong>Action:</strong> {plan.action}
              </div>
              <div>
                <strong>Webhook:</strong>{" "}
                {plan.action_config.webhook_url !== "not_provided"
                  ? "✅ Configured"
                  : "❌ Not configured"}
              </div>
            </div>
          </div>
        </div>
      )}

      {results && (
        <div className="results-section">
          <h3>⚡ Analysis Results</h3>
          <div className="analysis-results">
            <div className="results-summary">
              <div className="anomaly-count">
                <span className="count">{results.total_anomalies}</span>
                <span className="label">Insights Found</span>
              </div>
              <div className="slack-status">
                <span className={`status ${slackSent ? "sent" : "not-sent"}`}>
                  Slack: {slackSent ? "✅ Sent" : "❌ Not sent"}
                </span>
              </div>
            </div>

            <div className="domain-detected">
              <h4>🔍 Data Domain Detected</h4>
              <p>
                <strong>{results.domain_detected}</strong>
              </p>
              <p>
                <em>Analysis Type: {results.type}</em>
              </p>
            </div>

            <div className="summary">
              <h4>📊 Analysis Summary</h4>
              <p>{results.summary}</p>
            </div>

            {results.anomalies && results.anomalies.length > 0 && (
              <div className="anomalies">
                <h4>🚨 Detected Anomalies</h4>
                {results.anomalies.map((anomaly, index) => (
                  <div key={index} className="anomaly-card">
                    <div className="anomaly-header">
                      <h5>{anomaly.identifier}</h5>
                      <span
                        className={`severity ${anomaly.severity.toLowerCase()}`}
                      >
                        {anomaly.severity}
                      </span>
                    </div>
                    <p>
                      <strong>Value:</strong> {anomaly.value}
                    </p>
                    <p>
                      <strong>Details:</strong> {anomaly.details}
                    </p>
                    <p>
                      <strong>Action Required:</strong>{" "}
                      {anomaly.action_required}
                    </p>
                    <p>
                      <strong>Business Impact:</strong>{" "}
                      {anomaly.business_impact}
                    </p>
                    <p>
                      <strong>Reason:</strong> {anomaly.reason}
                    </p>
                  </div>
                ))}
              </div>
            )}

            {results.insights && results.insights.length > 0 && (
              <div className="insights">
                <h4>💡 Key Insights</h4>
                <ul>
                  {results.insights.map((insight, index) => (
                    <li key={index}>{insight}</li>
                  ))}
                </ul>
              </div>
            )}

            {results.recommendations && results.recommendations.length > 0 && (
              <div className="recommendations">
                <h4>🎯 Recommendations</h4>
                <ul>
                  {results.recommendations.map((rec, index) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Info Section */}
      <div className="info-section">
        <h3>💡 How It Works</h3>
        <div className="info-grid">
          <div className="info-card">
            <h4>1. Inspect</h4>
            <p>
              AI examines your database schema, identifies column types, and
              samples your data to understand the domain
            </p>
          </div>
          <div className="info-card">
            <h4>2. Auto-Plan</h4>
            <p>
              LLM intelligently creates a monitoring plan based on your goal and
              data type (works with ANY data domain)
            </p>
          </div>
          <div className="info-card">
            <h4>3. Run Once</h4>
            <p>
              LLM analyzes patterns in your data and provides contextual
              insights, recommendations, and alerts
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MVPDemo;
