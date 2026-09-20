BEGIN;

-- 分类结果已经保存面积时，补齐变化结果的冻结快照；已有快照值保持不变。
UPDATE change_results AS result
SET before_snapshot = jsonb_set(
        jsonb_set(
            result.before_snapshot::jsonb,
            '{class_area_m2}',
            to_jsonb(classification.class_area_m2),
            true
        ),
        '{area_status}',
        to_jsonb(classification.area_status),
        true
    )::json
FROM classification_results AS classification
WHERE result.status = 'SUCCEEDED'
  AND result.before_snapshot IS NOT NULL
  AND result.before_snapshot->>'class_area_m2' IS NULL
  AND classification.id = result.before_snapshot->>'result_id'
  AND classification.status = 'SUCCEEDED'
  AND classification.area_status = 'SUCCEEDED'
  AND jsonb_array_length(
        CASE
            WHEN jsonb_typeof(to_jsonb(classification.class_area_m2)) = 'array'
                THEN to_jsonb(classification.class_area_m2)
            ELSE '[]'::jsonb
        END
    ) = 6;

UPDATE change_results AS result
SET after_snapshot = jsonb_set(
        jsonb_set(
            result.after_snapshot::jsonb,
            '{class_area_m2}',
            to_jsonb(classification.class_area_m2),
            true
        ),
        '{area_status}',
        to_jsonb(classification.area_status),
        true
    )::json
FROM classification_results AS classification
WHERE result.status = 'SUCCEEDED'
  AND result.after_snapshot IS NOT NULL
  AND result.after_snapshot->>'class_area_m2' IS NULL
  AND classification.id = result.after_snapshot->>'result_id'
  AND classification.status = 'SUCCEEDED'
  AND classification.area_status = 'SUCCEEDED'
  AND jsonb_array_length(
        CASE
            WHEN jsonb_typeof(to_jsonb(classification.class_area_m2)) = 'array'
                THEN to_jsonb(classification.class_area_m2)
            ELSE '[]'::jsonb
        END
    ) = 6;

COMMIT;
