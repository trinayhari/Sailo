-- Quick Options Trading Table Setup
-- Copy and paste this into your Supabase SQL Editor

-- Create a simple options_trades table
CREATE TABLE options_trades (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    contract_type VARCHAR(4) CHECK (contract_type IN ('CALL', 'PUT')),
    strike_price DECIMAL(10,2) NOT NULL,
    premium DECIMAL(8,4) NOT NULL,
    volume INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert sample data
INSERT INTO options_trades (symbol, contract_type, strike_price, premium, volume) VALUES
('AAPL', 'CALL', 150.00, 5.25, 1250),
('AAPL', 'PUT', 145.00, 2.80, 890),
('TSLA', 'CALL', 200.00, 12.50, 2100),
('TSLA', 'PUT', 180.00, 8.75, 1650),
('SPY', 'CALL', 420.00, 3.15, 5600),
('NVDA', 'CALL', 500.00, 25.80, 3500);

-- Enable public access
ALTER TABLE options_trades ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow public access" ON options_trades FOR ALL USING (true);
