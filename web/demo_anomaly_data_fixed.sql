-- Fixed Anomalous Options Trading Data for Demo
-- This includes all required columns for your comprehensive schema

-- Insert ANOMALOUS data that will trigger alerts
INSERT INTO options_trades (
    symbol, 
    contract_type, 
    strike_price, 
    expiration_date,
    premium, 
    implied_volatility,
    delta,
    gamma,
    theta,
    vega,
    volume,
    open_interest,
    bid_price,
    ask_price,
    last_price,
    underlying_price,
    exchange,
    status
) VALUES
-- HIGH VOLATILITY SPIKE ANOMALIES (IV > 80%)
('MEME', 'CALL', 50.00, '2025-09-15', 15.75, 0.8500, 0.6500, 0.0250, -0.0850, 0.3200, 2500, 1200, 15.50, 16.00, 15.75, 52.30, 'CBOE', 'ACTIVE'),
('CRYPTO', 'PUT', 100.00, '2025-10-20', 22.50, 0.9200, -0.7200, 0.0180, -0.1200, 0.4100, 1800, 950, 22.00, 23.00, 22.50, 98.75, 'NASDAQ', 'ACTIVE'),

-- UNUSUAL VOLUME ANOMALIES (10x+ normal volume)
('AAPL', 'CALL', 155.00, '2025-08-30', 6.25, 0.3500, 0.5800, 0.0320, -0.0450, 0.1850, 25000, 8500, 6.00, 6.50, 6.25, 152.80, 'CBOE', 'ACTIVE'),
('SPY', 'PUT', 415.00, '2025-09-30', 4.80, 0.2200, -0.4200, 0.0280, -0.0320, 0.1200, 45000, 15000, 4.60, 5.00, 4.80, 418.50, 'NYSE', 'ACTIVE'),

-- PREMIUM ANOMALIES (very high premium-to-strike ratios)
('PENNY', 'CALL', 5.00, '2025-12-15', 2.75, 0.6500, 0.7500, 0.0450, -0.0650, 0.2800, 800, 400, 2.60, 2.90, 2.75, 4.85, 'CBOE', 'ACTIVE'),
('BIOTECH', 'PUT', 25.00, '2025-11-15', 12.80, 0.7800, -0.6800, 0.0380, -0.0980, 0.3500, 1200, 600, 12.50, 13.10, 12.80, 26.20, 'NASDAQ', 'ACTIVE'),

-- COMBINED ANOMALIES (multiple red flags)
('SQUEEZE', 'CALL', 75.00, '2025-08-25', 35.00, 0.9500, 0.8200, 0.0520, -0.1250, 0.4800, 18000, 12000, 34.50, 35.50, 35.00, 78.90, 'CBOE', 'ACTIVE'),
('VOLATILE', 'PUT', 200.00, '2025-09-20', 85.00, 0.8800, -0.7800, 0.0420, -0.1150, 0.4200, 12000, 8000, 84.00, 86.00, 85.00, 205.50, 'NYSE', 'ACTIVE');

-- Verify the anomalous data was inserted
SELECT 
    symbol, 
    contract_type, 
    strike_price,
    expiration_date,
    premium, 
    volume, 
    implied_volatility,
    ROUND((premium / strike_price * 100), 2) as premium_ratio_pct,
    CASE 
        WHEN implied_volatility > 0.8 THEN 'HIGH_IV_SPIKE'
        WHEN volume > 15000 THEN 'UNUSUAL_VOLUME'
        WHEN (premium / strike_price) > 0.4 THEN 'PREMIUM_ANOMALY'
        ELSE 'NORMAL'
    END as anomaly_type
FROM options_trades 
WHERE symbol IN ('MEME', 'CRYPTO', 'PENNY', 'BIOTECH', 'SQUEEZE', 'VOLATILE') 
   OR volume > 15000
ORDER BY implied_volatility DESC, volume DESC;
