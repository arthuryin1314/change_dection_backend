BEGIN;

ALTER TABLE change_results
    ADD COLUMN IF NOT EXISTS transition_pixel_counts JSON,
    ADD COLUMN IF NOT EXISTS transition_area_m2 JSON,
    ADD COLUMN IF NOT EXISTS unchanged_pixel_count INTEGER,
    ADD COLUMN IF NOT EXISTS unchanged_area_m2 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS nodata_pixel_count INTEGER,
    ADD COLUMN IF NOT EXISTS nodata_area_m2 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS nodata_value INTEGER,
    ADD COLUMN IF NOT EXISTS classification_scheme_version VARCHAR(64),
    ADD COLUMN IF NOT EXISTS transition_snapshot JSON;

COMMIT;
