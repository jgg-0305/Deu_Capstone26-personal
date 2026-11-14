ALTER TABLE media_assets DROP CONSTRAINT media_assets_kind_check;
ALTER TABLE media_assets ADD CHECK (kind IN ('clip', 'snapshot', 'thumbnail', 'source'));
CREATE UNIQUE INDEX media_assets_one_source_per_run ON media_assets (run_id)
    WHERE kind = 'source';
