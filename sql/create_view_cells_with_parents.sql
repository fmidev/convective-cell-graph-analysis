CREATE MATERIALIZED VIEW IF NOT EXISTS raincells.cells_with_parents_children_with_areas AS

    SELECT
        cur_cell.*,
        ST_Area(cur_cell.geometry) / 1e6 as cur_area,
        array_agg(DISTINCT prev_cell.identifier) as prev_identifiers,
        array_agg(DISTINCT ST_Area(prev_cell_sc.geometry) / 1e6) as prev_areas,
        array_agg(DISTINCT next_cell.next_identifier) as next_identifiers,
        array_agg(DISTINCT ST_Area(next_cell_sc.geometry) / 1e6) as next_areas
    FROM raincells.stormcells as cur_cell

    -- Get previous cell information (next_cell from previous timestamp, current id is next_cell.next_identifier)
    LEFT JOIN raincells.next_cells as prev_cell
        ON cur_cell.method = prev_cell.method AND cur_cell.timestamp = prev_cell.timestamp + interval '5 minutes' and cur_cell.identifier = prev_cell.next_identifier

    -- Get next cell information (next_cell from current timestamp, next cell id stored in next_cell.next_identifier)
    LEFT OUTER JOIN raincells.next_cells as next_cell
        ON cur_cell.method = next_cell.method AND cur_cell.timestamp = next_cell.timestamp and cur_cell.identifier = next_cell.identifier

    LEFT JOIN raincells.stormcells as prev_cell_sc
        ON prev_cell."timestamp" = prev_cell_sc."timestamp" AND prev_cell."method" = prev_cell_sc."method" AND prev_cell."identifier" = prev_cell_sc."identifier"

    LEFT JOIN raincells.stormcells as next_cell_sc
        ON next_cell."timestamp" = next_cell_sc."timestamp" AND next_cell."method" = next_cell_sc."method" AND next_cell."identifier" = next_cell_sc."identifier"

    GROUP BY cur_cell.timestamp, cur_cell.identifier, cur_cell.method

;
