BEGIN;

-- 旧分类结果没有计算时名称快照。仍存在来源记录时，以迁移时名称补齐并冻结；
-- 已有快照必须保留，避免覆盖真实的计算时名称。
UPDATE classification_results AS result
SET source_snapshot = json_build_object(
        'image', json_build_object(
            'id', result.source_image_id,
            'name', image.image_name
        ),
        'model', json_build_object(
            'id', result.source_model_id,
            'name', model.model_name
        )
    )
FROM images AS image, model_library AS model
WHERE result.status = 'SUCCEEDED'
  AND (
      result.source_snapshot IS NULL
      OR result.source_snapshot::jsonb = 'null'::jsonb
  )
  AND result.source_image_id = image.id
  AND result.source_model_id = model.id;

-- 旧变化结果已经冻结了两期分类快照，因此还需把来源名称写入这两份 JSON。
UPDATE change_results AS result
SET before_snapshot = jsonb_set(
        result.before_snapshot::jsonb,
        '{source}',
        jsonb_build_object(
            'image', jsonb_build_object(
                'id', result.before_image_id,
                'name', image.image_name
            ),
            'model', jsonb_build_object(
                'id', result.source_model_id,
                'name', model.model_name
            )
        ),
        true
    )::json
FROM images AS image, model_library AS model
WHERE result.status = 'SUCCEEDED'
  AND result.before_snapshot IS NOT NULL
  AND result.before_snapshot->>'source' IS NULL
  AND result.before_image_id = image.id
  AND result.source_model_id = model.id;

UPDATE change_results AS result
SET after_snapshot = jsonb_set(
        result.after_snapshot::jsonb,
        '{source}',
        jsonb_build_object(
            'image', jsonb_build_object(
                'id', result.after_image_id,
                'name', image.image_name
            ),
            'model', jsonb_build_object(
                'id', result.source_model_id,
                'name', model.model_name
            )
        ),
        true
    )::json
FROM images AS image, model_library AS model
WHERE result.status = 'SUCCEEDED'
  AND result.after_snapshot IS NOT NULL
  AND result.after_snapshot->>'source' IS NULL
  AND result.after_image_id = image.id
  AND result.source_model_id = model.id;

COMMIT;
