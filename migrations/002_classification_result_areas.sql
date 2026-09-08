BEGIN;

ALTER TABLE classification_results
    ADD COLUMN IF NOT EXISTS class_area_m2 JSON,
    ADD COLUMN IF NOT EXISTS area_status VARCHAR(16) NOT NULL DEFAULT 'NOT_COMPUTED',
    ADD COLUMN IF NOT EXISTS area_completed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS area_failure_detail TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_classification_results_area_status'
          AND conrelid = 'classification_results'::regclass
    ) THEN
        ALTER TABLE classification_results
            ADD CONSTRAINT ck_classification_results_area_status
            CHECK (area_status IN ('NOT_COMPUTED', 'SUCCEEDED', 'FAILED'));
    END IF;
END
$$;

COMMIT;
