BEGIN;

ALTER TABLE user_info
    ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT false;

-- System catalog has a separate identity space from the retired, user-owned
-- model_library. No legacy model or asset is imported into the new catalog.
CREATE TABLE IF NOT EXISTS system_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(32) NOT NULL,
    description TEXT,
    tags JSON NOT NULL DEFAULT '[]',
    production_version_id INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_system_models_type CHECK (model_type IN ('land-cover', 'change-detection'))
);

CREATE TABLE IF NOT EXISTS system_model_versions (
    id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES system_models(id) ON DELETE RESTRICT,
    version VARCHAR(100) NOT NULL,
    status VARCHAR(32) NOT NULL,
    framework VARCHAR(100) NOT NULL,
    input_description TEXT NOT NULL,
    input_bands JSON NOT NULL DEFAULT '[]',
    resource_requirements JSON NOT NULL DEFAULT '{}',
    code_config_sha256 VARCHAR(64) NOT NULL,
    weight_sha256 VARCHAR(64) NOT NULL,
    image_digest TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_system_model_version_name UNIQUE (model_id, version),
    CONSTRAINT uq_system_model_version_owner UNIQUE (model_id, id),
    CONSTRAINT ck_system_model_version_status CHECK (
        status IN ('UPLOADING', 'BUILDING', 'VALIDATING', 'AVAILABLE',
                   'BUILD_FAILED', 'VALIDATION_FAILED', 'OFFLINE')
    ),
    CONSTRAINT ck_system_model_version_available_identity CHECK (
        status <> 'AVAILABLE' OR image_digest IS NOT NULL
    )
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'system_models'::regclass
          AND conname = 'fk_system_models_production_version'
    ) THEN
        ALTER TABLE system_models ADD CONSTRAINT fk_system_models_production_version
            FOREIGN KEY (id, production_version_id)
            REFERENCES system_model_versions(model_id, id);
    END IF;
END $$;

-- Detach references explicitly before clearing legacy models. Keep result
-- rows, frozen names/hashes, masks, areas and change snapshots intact.
-- Do not use TRUNCATE CASCADE: it would delete the result tables as well.
UPDATE classification_results SET source_model_id = NULL WHERE source_model_id IS NOT NULL;
UPDATE change_results SET source_model_id = NULL WHERE source_model_id IS NOT NULL;
DELETE FROM model_library;

COMMIT;
