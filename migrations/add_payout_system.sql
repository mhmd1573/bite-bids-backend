-- ============================================
-- DEVELOPER PAYOUT SYSTEM MIGRATION
-- ============================================
-- Run this script to add payout functionality to BiteBids
--
-- This migration:
-- 1. Adds payout preference fields to the users table
-- 2. Creates the developer_payouts tracking table
-- ============================================

BEGIN;

-- ============================================
-- 1. ADD PAYOUT FIELDS TO USERS TABLE
-- ============================================
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS payout_method VARCHAR(50),
    ADD COLUMN IF NOT EXISTS payout_email VARCHAR(255),
    ADD COLUMN IF NOT EXISTS payout_details JSONB,
    ADD COLUMN IF NOT EXISTS payout_currency VARCHAR(10) DEFAULT 'USD',
    ADD COLUMN IF NOT EXISTS payout_verified BOOLEAN DEFAULT FALSE;

COMMENT ON COLUMN users.payout_method IS 'Developer preferred payout method: paypal, wise, bank_transfer, crypto, other';
COMMENT ON COLUMN users.payout_email IS 'Email for PayPal/Wise payments';
COMMENT ON COLUMN users.payout_details IS 'JSON with bank details, crypto wallet, etc.';
COMMENT ON COLUMN users.payout_currency IS 'Preferred currency for payouts';
COMMENT ON COLUMN users.payout_verified IS 'Whether admin has verified payout details';

-- ============================================
-- 2. CREATE DEVELOPER_PAYOUTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS developer_payouts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- References
    developer_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    checkout_session_id UUID REFERENCES checkout_sessions(id) ON DELETE SET NULL,
    investor_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Amount details
    gross_amount DECIMAL(12, 2) NOT NULL,
    platform_fee DECIMAL(12, 2) NOT NULL,
    net_amount DECIMAL(12, 2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',

    -- Payout method (snapshot at time of payout request)
    payout_method VARCHAR(50),
    payout_email VARCHAR(255),
    payout_details JSONB,

    -- Status tracking
    status VARCHAR(50) DEFAULT 'pending' NOT NULL,

    -- Processing details
    processed_by UUID REFERENCES users(id) ON DELETE SET NULL,
    processed_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- Transaction reference
    transaction_id VARCHAR(255),
    transaction_notes TEXT,

    -- Failure tracking
    failure_reason TEXT,
    retry_count INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Description
    description TEXT,

    -- Constraints
    CONSTRAINT developer_payouts_status_check
        CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled'))
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_payout_developer_id ON developer_payouts(developer_id);
CREATE INDEX IF NOT EXISTS idx_payout_status ON developer_payouts(status);
CREATE INDEX IF NOT EXISTS idx_payout_developer_status ON developer_payouts(developer_id, status);
CREATE INDEX IF NOT EXISTS idx_payout_created_at ON developer_payouts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_payout_project_id ON developer_payouts(project_id);

-- Add comments
COMMENT ON TABLE developer_payouts IS 'Tracks all developer payouts for completed projects';
COMMENT ON COLUMN developer_payouts.gross_amount IS 'Original project amount before fees';
COMMENT ON COLUMN developer_payouts.platform_fee IS 'Platform commission (6%)';
COMMENT ON COLUMN developer_payouts.net_amount IS 'Amount to pay developer after fees';
COMMENT ON COLUMN developer_payouts.transaction_id IS 'External payment reference (PayPal ID, bank ref, etc.)';

COMMIT;

-- ============================================
-- VERIFICATION QUERIES
-- ============================================

-- Verify users table has new columns
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'users'
    AND column_name IN ('payout_method', 'payout_email', 'payout_details', 'payout_currency', 'payout_verified')
ORDER BY column_name;

-- Verify developer_payouts table exists
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'developer_payouts'
ORDER BY ordinal_position;

-- Check indexes
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'developer_payouts';

-- ============================================
-- SAMPLE QUERIES FOR ADMIN DASHBOARD
-- ============================================

-- Get pending payouts with developer info
-- SELECT
--     dp.*,
--     u.name as developer_name,
--     u.email as developer_email,
--     p.title as project_title
-- FROM developer_payouts dp
-- JOIN users u ON dp.developer_id = u.id
-- LEFT JOIN projects p ON dp.project_id = p.id
-- WHERE dp.status = 'pending'
-- ORDER BY dp.created_at ASC;

-- Get payout stats by status
-- SELECT
--     status,
--     COUNT(*) as count,
--     SUM(net_amount) as total_amount
-- FROM developer_payouts
-- GROUP BY status;

-- Get developer's payout history
-- SELECT * FROM developer_payouts
-- WHERE developer_id = 'your-developer-uuid'
-- ORDER BY created_at DESC;
