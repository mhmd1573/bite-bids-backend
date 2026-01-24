-- Fix checkout_sessions status constraint to include 'refunded'
-- Run this script to update existing database constraint

BEGIN;

-- Drop the old constraint
ALTER TABLE checkout_sessions
    DROP CONSTRAINT IF EXISTS checkout_sessions_status_check;

-- Add the new constraint with 'refunded' status
ALTER TABLE checkout_sessions
    ADD CONSTRAINT checkout_sessions_status_check
    CHECK (status IN ('pending', 'completed', 'cancelled', 'expired', 'refunded'));

COMMIT;

-- Verify the change
SELECT
    conname AS constraint_name,
    pg_get_constraintdef(oid) AS constraint_definition
FROM pg_constraint
WHERE conrelid = 'checkout_sessions'::regclass
    AND contype = 'c';
