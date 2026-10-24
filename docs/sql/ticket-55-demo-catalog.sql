-- Optional fixture for a TEST database, after migrations/010_system_model_catalog.sql.
-- No model assets are created. Run once; these records cannot execute inference.
BEGIN;

DO $$
DECLARE
    land_id INTEGER;
    change_id INTEGER;
    land_version_id INTEGER;
BEGIN
    INSERT INTO system_models (model_name, model_type, description, tags)
    VALUES ('#55 验收地物模型', 'land-cover', '合成目录条目，没有实际模型资产', '["#55验收"]')
    RETURNING id INTO land_id;

    INSERT INTO system_models (model_name, model_type, description, tags)
    VALUES ('#55 验收变化模型', 'change-detection', '合成双期模型目录条目', '["#55验收"]')
    RETURNING id INTO change_id;

    INSERT INTO system_model_versions (
        model_id, version, status, framework, input_description, input_bands,
        resource_requirements, code_config_sha256, weight_sha256, image_digest
    ) VALUES (
        land_id, 'demo-v1', 'AVAILABLE', 'PyTorch', '单幅 RGB GeoTIFF', '[1,2,3]',
        '{"cpu_cores":2,"memory_mb":4096,"gpu_count":1}',
        repeat('a', 64), repeat('b', 64), 'sha256:' || repeat('c', 64)
    ) RETURNING id INTO land_version_id;

    UPDATE system_models SET production_version_id = land_version_id WHERE id = land_id;

    INSERT INTO system_model_versions (
        model_id, version, status, framework, input_description, input_bands,
        resource_requirements, code_config_sha256, weight_sha256
    ) VALUES (
        change_id, 'demo-v1', 'VALIDATION_FAILED', 'PyTorch', '两幅空间对齐的 RGB GeoTIFF', '[1,2,3]',
        '{"cpu_cores":2,"memory_mb":4096,"gpu_count":1}', repeat('d', 64), repeat('e', 64)
    );
END $$;

SELECT id, model_name, model_type FROM system_models
WHERE tags::jsonb @> '["#55验收"]'::jsonb ORDER BY id;

COMMIT;
