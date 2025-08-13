-- Options Trading Database Schema for Hackathon MVP
-- Run this in your Supabase SQL Editor

-- 1. Options Trades Table
CREATE TABLE IF NOT EXISTS options_trades (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    contract_type VARCHAR(4) CHECK (contract_type IN ('CALL', 'PUT')),
    strike_price DECIMAL(10,2) NOT NULL,
    expiration_date DATE NOT NULL,
    premium DECIMAL(8,4) NOT NULL,
    implied_volatility DECIMAL(6,4),
    delta DECIMAL(6,4),
    gamma DECIMAL(6,4),
    theta DECIMAL(6,4),
    vega DECIMAL(6,4),
    volume INTEGER DEFAULT 0,
    open_interest INTEGER DEFAULT 0,
    bid_price DECIMAL(8,4),
    ask_price DECIMAL(8,4),
    last_price DECIMAL(8,4),
    underlying_price DECIMAL(10,2),
    trade_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    trader_id VARCHAR(50),
    exchange VARCHAR(10) DEFAULT 'CBOE',
    status VARCHAR(20) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'EXPIRED', 'EXERCISED', 'CLOSED')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Market Data Table (for underlying stocks)
CREATE TABLE IF NOT EXISTS market_data (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    volume BIGINT DEFAULT 0,
    market_cap BIGINT,
    pe_ratio DECIMAL(8,2),
    volatility_30d DECIMAL(6,4),
    beta DECIMAL(6,4),
    sector VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Risk Metrics Table (for monitoring)
CREATE TABLE IF NOT EXISTS risk_metrics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    portfolio_id VARCHAR(50),
    var_1d DECIMAL(12,2), -- Value at Risk 1 day
    var_5d DECIMAL(12,2), -- Value at Risk 5 day
    max_drawdown DECIMAL(8,4),
    sharpe_ratio DECIMAL(6,4),
    total_exposure DECIMAL(15,2),
    delta_exposure DECIMAL(15,2),
    gamma_exposure DECIMAL(15,2),
    vega_exposure DECIMAL(15,2),
    theta_exposure DECIMAL(15,2),
    concentration_risk DECIMAL(6,4),
    liquidity_risk VARCHAR(20) CHECK (liquidity_risk IN ('LOW', 'MEDIUM', 'HIGH')),
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Alerts Table (for AI-generated alerts)
CREATE TABLE IF NOT EXISTS trading_alerts (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    symbol VARCHAR(10),
    message TEXT NOT NULL,
    details JSONB,
    triggered_by VARCHAR(100), -- What rule/condition triggered this
    status VARCHAR(20) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'ACKNOWLEDGED', 'RESOLVED')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Insert Sample Options Trading Data
INSERT INTO options_trades (symbol, contract_type, strike_price, expiration_date, premium, implied_volatility, delta, gamma, theta, vega, volume, open_interest, bid_price, ask_price, last_price, underlying_price, trader_id, exchange) VALUES
('AAPL', 'CALL', 150.00, '2024-03-15', 5.25, 0.2850, 0.6500, 0.0125, -0.0450, 0.1250, 1250, 5680, 5.20, 5.30, 5.25, 148.75, 'TRADER_001', 'CBOE'),
('AAPL', 'PUT', 145.00, '2024-03-15', 2.80, 0.3100, -0.3500, 0.0135, -0.0380, 0.1180, 890, 3420, 2.75, 2.85, 2.80, 148.75, 'TRADER_002', 'CBOE'),
('TSLA', 'CALL', 200.00, '2024-04-19', 12.50, 0.4500, 0.7200, 0.0095, -0.0650, 0.2150, 2100, 8950, 12.40, 12.60, 12.50, 195.30, 'TRADER_003', 'NASDAQ'),
('TSLA', 'PUT', 180.00, '2024-04-19', 8.75, 0.4800, -0.4800, 0.0105, -0.0580, 0.2050, 1650, 6780, 8.70, 8.80, 8.75, 195.30, 'TRADER_001', 'NASDAQ'),
('SPY', 'CALL', 420.00, '2024-02-16', 3.15, 0.1850, 0.5500, 0.0180, -0.0320, 0.0950, 5600, 15680, 3.10, 3.20, 3.15, 418.50, 'TRADER_004', 'CBOE'),
('SPY', 'PUT', 410.00, '2024-02-16', 1.95, 0.2050, -0.2800, 0.0165, -0.0280, 0.0880, 3200, 9870, 1.90, 2.00, 1.95, 418.50, 'TRADER_002', 'CBOE'),
('NVDA', 'CALL', 500.00, '2024-05-17', 25.80, 0.5200, 0.8100, 0.0075, -0.0850, 0.3200, 3500, 12450, 25.60, 26.00, 25.80, 485.20, 'TRADER_005', 'NASDAQ'),
('NVDA', 'PUT', 450.00, '2024-05-17', 18.90, 0.5500, -0.6200, 0.0085, -0.0780, 0.3100, 2800, 8960, 18.80, 19.00, 18.90, 485.20, 'TRADER_003', 'NASDAQ');

-- Insert Sample Market Data
INSERT INTO market_data (symbol, price, volume, market_cap, pe_ratio, volatility_30d, beta, sector) VALUES
('AAPL', 148.75, 45680000, 2400000000000, 28.5, 0.2850, 1.15, 'Technology'),
('TSLA', 195.30, 32150000, 620000000000, 45.2, 0.4500, 2.05, 'Automotive'),
('SPY', 418.50, 78950000, NULL, NULL, 0.1850, 1.00, 'ETF'),
('NVDA', 485.20, 28750000, 1200000000000, 65.8, 0.5200, 1.85, 'Technology'),
('MSFT', 385.90, 22340000, 2850000000000, 32.1, 0.2650, 0.95, 'Technology'),
('AMZN', 142.80, 35680000, 1480000000000, 42.3, 0.3200, 1.25, 'Consumer Discretionary');

-- Insert Sample Risk Metrics
INSERT INTO risk_metrics (portfolio_id, var_1d, var_5d, max_drawdown, sharpe_ratio, total_exposure, delta_exposure, gamma_exposure, vega_exposure, theta_exposure, concentration_risk, liquidity_risk) VALUES
('PORTFOLIO_001', -125000.50, -285000.75, -0.0850, 1.85, 2500000.00, 850000.00, 12500.00, 45000.00, -8500.00, 0.2500, 'LOW'),
('PORTFOLIO_002', -89000.25, -205000.00, -0.0650, 2.15, 1800000.00, 620000.00, 8900.00, 32000.00, -6200.00, 0.1800, 'MEDIUM'),
('PORTFOLIO_003', -245000.80, -560000.50, -0.1250, 1.45, 4200000.00, 1450000.00, 22000.00, 78000.00, -15000.00, 0.3500, 'HIGH');

-- Insert Sample Trading Alerts
INSERT INTO trading_alerts (alert_type, severity, symbol, message, details, triggered_by, status) VALUES
('VOLATILITY_SPIKE', 'HIGH', 'TSLA', 'Implied volatility increased by 25% in the last hour', '{"iv_change": 0.25, "current_iv": 0.45, "previous_iv": 0.36}', 'IV_MONITOR_RULE', 'ACTIVE'),
('UNUSUAL_VOLUME', 'MEDIUM', 'AAPL', 'Options volume 300% above average for this strike/expiration', '{"volume_ratio": 3.2, "avg_volume": 890, "current_volume": 2850}', 'VOLUME_ANOMALY_RULE', 'ACTIVE'),
('DELTA_EXPOSURE', 'CRITICAL', 'SPY', 'Portfolio delta exposure exceeds risk limits', '{"current_delta": 1250000, "limit": 1000000, "excess": 250000}', 'RISK_LIMIT_RULE', 'ACKNOWLEDGED'),
('PRICE_GAP', 'HIGH', 'NVDA', 'Underlying price gap detected - options may be mispriced', '{"gap_size": 0.15, "expected_price": 485.20, "market_price": 500.50}', 'PRICING_MODEL_RULE', 'ACTIVE');

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_options_trades_symbol ON options_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_options_trades_expiration ON options_trades(expiration_date);
CREATE INDEX IF NOT EXISTS idx_options_trades_timestamp ON options_trades(trade_timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_trading_alerts_severity ON trading_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_trading_alerts_status ON trading_alerts(status);

-- Enable Row Level Security (optional)
ALTER TABLE options_trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading_alerts ENABLE ROW LEVEL SECURITY;

-- Create policies to allow public access (for demo purposes)
CREATE POLICY "Allow public access" ON options_trades FOR ALL USING (true);
CREATE POLICY "Allow public access" ON market_data FOR ALL USING (true);
CREATE POLICY "Allow public access" ON risk_metrics FOR ALL USING (true);
CREATE POLICY "Allow public access" ON trading_alerts FOR ALL USING (true);
