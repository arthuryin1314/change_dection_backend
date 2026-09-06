BEGIN;

ALTER TABLE images
    ADD COLUMN IF NOT EXISTS content_sha256 VARCHAR(64),
    ADD COLUMN IF NOT EXISTS content_sha256_size BIGINT,
    ADD COLUMN IF NOT EXISTS content_sha256_mtime_ns BIGINT;

ALTER TABLE model_library
    ADD COLUMN IF NOT EXISTS weight_content_sha256 VARCHAR(64),
    ADD COLUMN IF NOT EXISTS weight_content_sha256_size BIGINT,
    ADD COLUMN IF NOT EXISTS weight_content_sha256_mtime_ns BIGINT;

CREATE TABLE IF NOT EXISTS classification_results (
    id VARCHAR(32) PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES user_info(id) ON DELETE CASCADE,
    source_image_id INTEGER REFERENCES images(id) ON DELETE SET NULL,
    source_model_id INTEGER REFERENCES model_library(id) ON DELETE SET NULL,
    identity_sha256 VARCHAR(64) NOT NULL,
    image_content_sha256 VARCHAR(64) NOT NULL,
    weight_content_sha256 VARCHAR(64) NOT NULL,
    inference_parameters JSON NOT NULL,
    classification_scheme_version VARCHAR(64) NOT NULL,
    pipeline_version VARCHAR(64) NOT NULL,
    grid_policy_version VARCHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    heartbeat_at TIMESTAMPTZ NOT NULL,
    lease_expires_at TIMESTAMPTZ,
    lease_owner VARCHAR(128) NOT NULL,
    completed_at TIMESTAMPTZ,
    failure_detail TEXT,
    classes_path TEXT,
    valid_mask_path TEXT,
    crs TEXT,
    transform JSON,
    raster_width INTEGER,
    raster_height INTEGER,
    resolution JSON,
    bounds JSON,
    generation_metrics JSON,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_classification_results_status
        CHECK (status IN ('PROCESSING', 'SUCCEEDED', 'FAILED'))
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_classification_results_user_identity
    ON classification_results (user_id, identity_sha256);

CREATE INDEX IF NOT EXISTS ix_classification_results_user_status
    ON classification_results (user_id, status);

COMMIT;
