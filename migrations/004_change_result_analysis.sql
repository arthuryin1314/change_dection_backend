BEGIN;

ALTER TABLE change_results
    ADD COLUMN IF NOT EXISTS calculation_version VARCHAR(64),
    ADD COLUMN IF NOT EXISTS grid_policy_version VARCHAR(64),
    ADD COLUMN IF NOT EXISTS analysis_identity_sha256 VARCHAR(64),
    ADD COLUMN IF NOT EXISTS analysis_metadata JSON;

UPDATE change_results
SET calculation_version = COALESCE(calculation_version, 'transition-matrix-v1'),
    grid_policy_version = COALESCE(grid_policy_version, 'same-grid-v0');

ALTER TABLE change_results
    ALTER COLUMN calculation_version SET NOT NULL,
    ALTER COLUMN grid_policy_version SET NOT NULL;

COMMIT;
