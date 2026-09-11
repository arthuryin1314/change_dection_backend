BEGIN;

ALTER TABLE classification_results
    ADD COLUMN IF NOT EXISTS source_snapshot JSON;

CREATE INDEX IF NOT EXISTS ix_classification_results_user_history
    ON classification_results (user_id, completed_at DESC, id DESC)
    WHERE status = 'SUCCEEDED';

CREATE INDEX IF NOT EXISTS ix_change_results_user_history
    ON change_results (user_id, completed_at DESC, id DESC)
    WHERE status = 'SUCCEEDED';

COMMIT;
