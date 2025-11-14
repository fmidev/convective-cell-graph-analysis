# Data for manuscript "Analysis of convective cell development with split and merge events using a graph-based methodology" by Ritvanen et al.

This repository contains data for the manuscript "Analysis of convective cell development with split and merge events using a graph-based methodology" by Ritvanen et al. submitted to Atmospheric Measurement Techniques.

The following data is provided:

- `cell_graph_database_v20251105.tar`: a database dump file containing the database with cell track data
- `cell_subgraph_data_v20251105.tar`: a tar file containing subgraph data for all split and merge events. The file contains daily parquet files for each day in the study period.
- `numerical_figure_data_v20251105.tar`: a tar file containing data files with numerical data used to create the figures in the manuscript.
- `README_database.md`: file describing the database contents.
- `README_data.md`: this file describing the data files.

## Description of database file

The database dump was created with the command

```bash
pg_dump --no-owner --no-privileges -C --format=t --blobs --verbose --user <username> --file "<filename.tar>" <database-name>
```

Restoring the database can be done with the commands

```bash
createdb -O <username> <database-name>
pg_restore -vxOW --role=<username> -U <username> -d <database-name> <filename.tar>
```

Note that this assumes you have PostgreSQL with PostGIS installed and a PostgreSQL user with the given username exists and has permission to create databases.

For more information about the database structure and contents, see the file `README_database.md`.

## Description of subgraph data files

The subgraph data tar file contains daily parquet files for each day in the study period. Each parquet file contains subgraph data for all split and merge events that occurred on that day. The columns in the parquet files are:

- `method`: cell tracking method used (only 'opencv_vil_1.0:minArea_10:clusters_0' in this data)
- `type`: type of event ('split', 'merged', 'split-merge')
- `identifier`: identifier of the cell, unique in combination with `timestamp` and `method`
- `t0_node`: identifier of the t0 / event node of the subgraph
- `timestamp`: timestamp of the cell
- `level`: level of the cell from the event node (0 = event node, -1 = one timestep before event, 1 = one timestep after event, etc.)
- `area`: area of the cell in km^2
- `event`: sting descriptor of cell tracking status
- `num_cells_at_level`: number of cells at the given level in the subgraph
- `t0_time`: timestamp of the event node

For examples of how to read and use the subgraph data, see the Jupyter notebook `notebooks/plot_article_figures.ipynb` in the code provided with manuscript

## Description of files containing numerical version of figures

- Figure 7:
  - `fig7a_split_merge_num_cells_histogram.csv`: fraction of events as a function of number of cells participating in splits and merges. Columns: 'num_cells' (number of cells), ´type´ (split or merged), ´fraction_of_events´ (fraction of events with given number of cells).
  - `fig7b_split_merge_num_cells_histogram.csv`: fraction of events as a function of number of cells participating in splits and merges in merge-split events. First row contains number of merging cells and first column number of splitting cells. Values are fraction of events with given number of merging and splitting cells.
- Figure 8:
  - `fig8_split_merge_cell_area_histograms.json`: JSON file containing x and y values for cell area histograms for split, merge, and merge-split event and cells involved in split and merge events. The structure is:
    ```json
    {
      "split": {
        "area": {
            "<group label>": {
              "x": [<list of x values>],
              "y": [<list of y values>]
            },
            ...
        },
        ...
      }
    ```
    The first level keys are 'split', 'merged', 'split-merge', 'splitted', 'merging' corresponding to different event and cell types. The second level keys are 'area' (cell area). The third level keys are group labels used in the histograms. Each group label contains x and y values for the histogram. Note that the x, y values correspond to the linepoints used to draw the histogram (not the bar heights).
- Figure 9:
  - `fig8_split_merge_area_ratio_histograms.json`: JSON file containing x and y values for area ratio histograms for split and merged cells. The structure is:
    ```json
    {
      "split": {
        "<area_interval>": {
          "x": [<list of x values>],
          "y": [<list of y values>]
        },
        ...
      },
      "merged": {
        "<area_interval>": {
          "x": [<list of x values>],
          "y": [<list of y values>]
        },
        ...
      }
    }
    ```
    where `<area_interval>` is a string representation of the area interval used in the histogram (e.g., "(0, 1000]").
- Figure 10:

  - `fig10_trajectory_development_split_merge_vil_thr_20.json`: JSON file containing x and y values for trajectory development plots for split and merged cells with maximum VIL threshold of 20 dBZ. The structure is:

    ```json
    {
      "<subplot title": {
        "Subgraph count [1000]": {
          "x": [<list of x values],
          "y": [<list of y values in 1000s]
        },
        "<variable_title>": {
          "x": [<list of x values for mean line],
          "y": [<list of y values for mean line]
        },
        "<variable_title>_ci": {
          "bottom_x": [<list of x values for lower bound],
          "bottom_y": [<list of y values for lower bound],
          "top_x": [<list of x values for upper bound],
          "top_y": [<list of y values for upper bound]
        },
        ...
      },
      ...
    }
    ```

    where `<variable_title>` is the title of the variable plotted (e.g., "Total cell Area") and `<subplot title>` is the title of the subplot.

  - `fig10_trajectory_development_split_merge_all_available_between_min_-3_max_0_max_vil_thr_20.json`: same as above but for all available cells between timesteps -3 and 0 with maximum VIL threshold of 20 dBZ.
  - `fig10_trajectory_development_split_merge_all_available_between_min_-3_max_6_max_vil_thr_20.json`: same as above but for all available cells between timesteps -3 and 6 with maximum VIL threshold of 20 dBZ.

- Figure C1:
  - `figC1_trajectory_development_split_merge.json`: same structure as Figure 10 but without any VIL threshold applied.
  - `figC1_trajectory_development_split_merge_all_available_between_min_-3_max_0.json`: same as above but for all available cells between timesteps -3 and 0.
  - `figC1_trajectory_development_split_merge_all_available_between_min_-3_max_6.json`: same as above but for all available cells between timesteps -3 and 6.
