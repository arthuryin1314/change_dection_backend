BEGIN;

ALTER TABLE change_results
    ALTER COLUMN before_identity_sha256 DROP NOT NULL,
    ALTER COLUMN after_identity_sha256 DROP NOT NULL,
    ADD COLUMN IF NOT EXISTS preparation_key VARCHAR(64),
    ADD COLUMN IF NOT EXISTS orchestration_identity_sha256 VARCHAR(64),
    ADD COLUMN IF NOT EXISTS phase VARCHAR(32),
    ADD COLUMN IF NOT EXISTS submitted_inputs JSON,
    ADD COLUMN IF NOT EXISTS frozen_inputs JSON;

UPDATE change_results
SET phase = COALESCE(
        phase,
        CASE WHEN status = 'SUCCEEDED' THEN 'SUCCEEDED'
             WHEN status = 'FAILED' THEN 'FAILED'
             ELSE 'COMPUTING_MATRIX'
        END
    ),
    submitted_inputs = COALESCE(
        submitted_inputs,
        json_build_object(
            'before_image_id', before_image_id,
            'after_image_id', after_image_id,
            'model_id', source_model_id
        )
    );

ALTER TABLE change_results
    ALTER COLUMN phase SET NOT NULL,
    ALTER COLUMN submitted_inputs SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_change_results_active_preparation
    ON change_results (user_id, preparation_key)
    WHERE status = 'PROCESSING';

CREATE UNIQUE INDEX IF NOT EXISTS uq_change_results_active_identity
    ON change_results (user_id, orchestration_identity_sha256)
    WHERE status = 'PROCESSING' AND orchestration_identity_sha256 IS NOT NULL;

CREATE TABLE IF NOT EXISTS change_requests (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES user_info(id) ON DELETE CASCADE,
    request_id VARCHAR(36) NOT NULL,
    request_fingerprint VARCHAR(64) NOT NULL,
    change_result_id BIGINT NOT NULL REFERENCES change_results(id) ON DELETE CASCADE,
    retry_of_request_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_change_requests_user_request UNIQUE (user_id, request_id)
);

UPDATE change_requests
SET request_fingerprint = 'legacy:' || LEFT(request_fingerprint, 32)
WHERE LENGTH(request_fingerprint) = 64
  AND LEFT(request_fingerprint, 32) = RIGHT(request_fingerprint, 32);

INSERT INTO change_requests (
    user_id,
    request_id,
    request_fingerprint,
    change_result_id
)
SELECT
    user_id,
    request_id,
    'legacy:' || md5(
        concat_ws(
            ':',
            COALESCE(before_image_id::text, ''),
            COALESCE(after_image_id::text, ''),
            COALESCE(source_model_id::text, '')
        )
    ),
    id
FROM change_results
ON CONFLICT (user_id, request_id) DO NOTHING;

CREATE INDEX IF NOT EXISTS ix_change_requests_result
    ON change_requests (change_result_id);

COMMIT;
