BEGIN;

CREATE TABLE IF NOT EXISTS change_results (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(36) NOT NULL,
    user_id BIGINT NOT NULL REFERENCES user_info(id) ON DELETE CASCADE,
    before_image_id INTEGER REFERENCES images(id) ON DELETE SET NULL,
    after_image_id INTEGER REFERENCES images(id) ON DELETE SET NULL,
    source_model_id INTEGER REFERENCES model_library(id) ON DELETE SET NULL,
    before_result_id VARCHAR(32) REFERENCES classification_results(id) ON DELETE SET NULL,
    after_result_id VARCHAR(32) REFERENCES classification_results(id) ON DELETE SET NULL,
    before_identity_sha256 VARCHAR(64) NOT NULL,
    after_identity_sha256 VARCHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL,
    lease_owner VARCHAR(128) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    heartbeat_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    error_http_status INTEGER,
    error_code VARCHAR(64),
    error_message TEXT,
    error_data JSON,
    before_snapshot JSON,
    after_snapshot JSON,
    matrix_m2 JSON,
    common_valid_area_m2 DOUBLE PRECISION,
    crs TEXT,
    transform JSON,
    raster_width INTEGER,
    raster_height INTEGER,
    bounds JSON,
    before_window JSON,
    after_window JSON,
    calculated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_change_results_user_request UNIQUE (user_id, request_id),
    CONSTRAINT ck_change_results_status CHECK (status IN ('PROCESSING', 'SUCCEEDED', 'FAILED'))
);

CREATE INDEX IF NOT EXISTS ix_change_results_user_status
    ON change_results (user_id, status);

COMMIT;
