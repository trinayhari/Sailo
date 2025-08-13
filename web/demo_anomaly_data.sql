-- Enhanced Options Trading Data with Anomalies for Demo
-- Copy and paste this into your Supabase SQL Editor to add anomalous data

-- First, add the missing implied_volatility column if it doesn't exist
ALTER TABLE options_trades ADD COLUMN IF NOT EXISTS implied_volatility DECIMAL(6,4);

-- Update existing records with normal implied volatility values
UPDATE options_trades SET implied_volatility = 0.2850 WHERE symbol = 'AAPL' AND contract_type = 'CALL';
UPDATE options_trades SET implied_volatility = 0.3100 WHERE symbol = 'AAPL' AND contract_type = 'PUT';
UPDATE options_trades SET implied_volatility = 0.4500 WHERE symbol = 'TSLA' AND contract_type = 'CALL';
UPDATE options_trades SET implied_volatility = 0.4800 WHERE symbol = 'TSLA' AND contract_type = 'PUT';
UPDATE options_trades SET implied_volatility = 0.1850 WHERE symbol = 'SPY' AND contract_type = 'CALL';
UPDATE options_trades SET implied_volatility = 0.3200 WHERE symbol = 'NVDA' AND contract_type = 'CALL';

-- Insert ANOMALOUS data that will trigger alerts
INSERT INTO options_trades (symbol, contract_type, strike_price, premium, volume, implied_volatility) VALUES
-- HIGH VOLATILITY SPIKE ANOMALIES (IV > 80%)
('MEME', 'CALL', 50.00, 15.75, 2500, 0.8500),  -- 85% IV - CRITICAL
('CRYPTO', 'PUT', 100.00, 22.50, 1800, 0.9200), -- 92% IV - CRITICAL

-- UNUSUAL VOLUME ANOMALIES (10x+ normal volume)
('AAPL', 'CALL', 155.00, 6.25, 25000, 0.3500),  -- 25k volume vs ~1-5k normal
('SPY', 'PUT', 415.00, 4.80, 45000, 0.2200),    -- 45k volume vs ~5k normal

-- PREMIUM ANOMALIES (very high premium-to-strike ratios)
('PENNY', 'CALL', 5.00, 2.75, 800, 0.6500),     -- 55% premium/strike ratio
('BIOTECH', 'PUT', 25.00, 12.80, 1200, 0.7800), -- 51% premium/strike ratio

-- COMBINED ANOMALIES (multiple red flags)
('SQUEEZE', 'CALL', 75.00, 35.00, 18000, 0.9500), -- High IV + High Volume + High Premium
('VOLATILE', 'PUT', 200.00, 85.00, 12000, 0.8800); -- Multiple anomalies

-- Verify the data
SELECT 
    symbol, 
    contract_type, 
    strike_price, 
    premium, 
    volume, 
    implied_volatility,
    ROUND((premium / strike_price * 100), 2) as premium_ratio_pct
FROM options_trades 
ORDER BY implied_volatility DESC, volume DESC;
