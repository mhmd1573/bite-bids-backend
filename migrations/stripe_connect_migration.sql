-- ================================================
-- Stripe Connect Migration Script
-- Run this script to add Stripe Connect support
-- ================================================

-- Add Stripe Connect fields to users table
ALTER TABLE users
ADD COLUMN IF NOT EXISTS stripe_account_id VARCHAR(255) UNIQUE,
ADD COLUMN IF NOT EXISTS stripe_account_status VARCHAR(50),
ADD COLUMN IF NOT EXISTS stripe_payouts_enabled BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS stripe_onboarding_completed BOOLEAN DEFAULT FALSE;

-- Add Stripe transfer fields to developer_payouts table
ALTER TABLE developer_payouts
ADD COLUMN IF NOT EXISTS stripe_transfer_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS stripe_transfer_status VARCHAR(50);

-- Create index for Stripe account lookups
CREATE INDEX IF NOT EXISTS idx_users_stripe_account_id ON users(stripe_account_id);

-- Create index for Stripe transfer lookups
CREATE INDEX IF NOT EXISTS idx_developer_payouts_stripe_transfer_id ON developer_payouts(stripe_transfer_id);

-- ================================================
-- IMPORTANT: Stripe Environment Variables Required
-- ================================================
-- Make sure to add these to your .env file:
--
-- STRIPE_SECRET_KEY=sk_live_... (or sk_test_... for testing)
-- STRIPE_CONNECT_WEBHOOK_SECRET=whsec_...
--
-- For Stripe Connect, you also need to:
-- 1. Enable Connect in your Stripe Dashboard
-- 2. Set up the redirect URL: https://yourdomain.com/dashboard?tab=payout-settings
-- 3. Configure the webhook endpoint: https://yourdomain.com/api/stripe-connect/webhook
--    - Subscribe to: account.updated, transfer.paid, transfer.failed
-- ================================================

-- Verification query (run after migration)
-- SELECT
--     column_name,
--     data_type,
--     is_nullable
-- FROM information_schema.columns
-- WHERE table_name = 'users'
--   AND column_name LIKE 'stripe%';
