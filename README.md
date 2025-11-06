# Graph-based analysis of convective cell development

This repository contains code for the manuscript

> Analysis of convective cell development with split and merge events using a graph-based methodology

by Ritvanen et al., submitted to Atmospheric Measurement Techniques.

To investigate the analysis results provided in the manuscript, you can restore the data provided with the manuscript on [FMI METIS repository](https://doi.org/10.57707/fmi-b2share.ac2197da4a034d21bee1fd9cb75ecfaf). Instruction for this are provided in the section "Restoring data provided with manuscript" below.

Workflow to re-create the analysis in the manuscript is the following. Each step is described in more detail in this README.

1. Create a conda environment with the required packages.
2. Create a database to contain the cell tracking results
3. Identify cells in the radar data
4. Track the identified cells
5. Create and materialize views of cell information in the database.
6. For large datasets, determine continuous stretches of cell tracks to parallelize the graph building.
7. Build cell track graphs.
8. Extract cell features from data and store them in the database.
9. Determine distances of cells from radars.
10. Select $t_0$ event nodes and build subgraphs for them.
11. Plot result figures.

All instructions below assume that you have a working installation of PostgreSQL with PostGIS extensions and conda.

- [Graph-based analysis of convective cell development](#graph-based-analysis-of-convective-cell-development)
  - [Restoring data provided with manuscript](#restoring-data-provided-with-manuscript)
    - [Restore the database](#restore-the-database)
    - [Restore cell graph data](#restore-cell-graph-data)
  - [Workflow to re-create the analysis for your own data](#workflow-to-re-create-the-analysis-for-your-own-data)
    - [1. Create a conda environment with the required packages](#1-create-a-conda-environment-with-the-required-packages)
    - [2. Create a database to contain the cell tracking results](#2-create-a-database-to-contain-the-cell-tracking-results)
    - [3. Identify cells in the radar data](#3-identify-cells-in-the-radar-data)
    - [4. Track the identified cells](#4-track-the-identified-cells)
    - [5. Create and materialize view of cell parent-child information in the database](#5-create-and-materialize-view-of-cell-parent-child-information-in-the-database)
    - [6. For large datasets, determine continuous stretches of cell tracks to parallelize the cell track graph building](#6-for-large-datasets-determine-continuous-stretches-of-cell-tracks-to-parallelize-the-cell-track-graph-building)
    - [7. Extract cell features from data and store them in the database](#7-extract-cell-features-from-data-and-store-them-in-the-database)
    - [8. Determine distances of cells from radars](#8-determine-distances-of-cells-from-radars)
    - [7. Build cell track graphs](#7-build-cell-track-graphs)
    - [10. Select $t\_0$ events and build subgraph graphs for them](#10-select-t_0-events-and-build-subgraph-graphs-for-them)
    - [11. Plot result figures.](#11-plot-result-figures)

## Restoring data provided with manuscript

First, follow step 1 to create conda environments with the required packages.

Then, download the provided data from [FMI METIS repository](https://doi.org/10.57707/fmi-b2share.ac2197da4a034d21bee1fd9cb75ecfaf) and unzip it to a folder of your choice. The data contains:

- A PostgreSQL database dump file `raincells_db_dump.tar` that contains the cell tracking results and extracted cell features for the Swiss radar data used in the manuscript. The database was dumped with the command

```bash
pg_dump --no-owner --no-privileges -C --format=t --blobs --verbose --user <username> --file "<filename.tar>" <database-name>
```

- A folder `cell_graph_data_v20250827` that contains the cell track graphs and subgraph graphs used in the manuscript.

Use the instructions below to restore the database. The article figures can then be plotted by using the notebook [plot_article_figures.ipynb](notebooks/plot_article_figures.ipynb) and the case study figures in the notebook [plot_case_study_figures.ipynb](notebooks/plot_case_study_figures.ipynb).

### Restore the database

1. Create a PostgreSQL database with PostGIS extensions. If you are using PostgreSQL in a Linux environment, this can be done by using the `createdb` command:

```bash
createdb -O <username> <database-name>
```

2. Restore the database dump file into the created database:

```bash
pg_restore -vxOW --role=<username> -U <username> -d <database-name> <filename.tar>
```

If prompted, provide the password for the database user.
After this, the database is ready to be used.

Set the database connection parameters in the [`config/database/database.yaml`](config/database/database.yaml) file to point to your database.

3. Set database user and password as environment variables:

```bash
export DB_USER=<your-database-username>
export DB_PASSWD=<your-database-password>
```

### Restore cell graph data

1. Unpack the `cell_graph_data_v20250827.tar` file to a folder of your choice.

2. To re-plot the figures in the manuscript, adjust the paths in [`notebooks/plot_article_figures.ipynb`](notebooks/plot_article_figures.ipynb)

- in 3rd cell, set the `OUTPATH` variable to the path where you want to store the output figures.
- in 6th cell, set the `storagepath` variable to the path where you unpacked the `cell_graph_data_v20250827` folder.

## Workflow to re-create the analysis for your own data

### 1. Create a conda environment with the required packages

Create a conda environment with the required packages. This can be done by using the provided [env_cell_stats.yaml](env_cell_stats.yaml) file:

```bash
conda env create -f env_cell_stats.yaml
conda activate raincells
```

An additional conda environment for running Jupyter notebooks is provided in the [env_jupyter.yaml](env_jupyter.yaml) file.

Note that the use of Swiss radar data in the original data format (not HDF5) requires the installation of the `radlib` Python package. This is not included in the `env_cell_stats.yaml` file, as it is not available through conda or pip publicly, so the package should be installed separately.

### 2. Create a database to contain the cell tracking results

1. Create a PostgreSQL database with PostGIS extensions. If you are using PostgreSQL in a Linux environment, this can be done by using the `createdb` command:

```bash
createdb <db-name>
```

Then log in to the database and create the 'raincells' schema:

```bash
psql <db-name>
CREATE SCHEMA raincells;
```

Note that the database name can be freely chosen, but the schema name "raincells" is hardcoded in the scripts.

2. Create configuration folder under [`stormcell_nowcasting/stormcell_nowcasting/config`](stormcell_nowcasting/stormcell_nowcasting/config) by copying the `template` folder.
3. Edit the `database.yaml` file in the created configuration folder to include your database connection details and the projection string for your radar data.
4. Edit the database connection parameters in the [`config/database/database.yaml`](config/database/database.yaml) file.

5. Create the database tables

```bash
cd stormcell_nowcasting/stormcell_nowcasting/
# Add parent path to PYTHONPATH
export PYTHONPATH=<path-to-parent-folder-of-stormcell_nowcasting>:$PYTHONPATH
python create_database_tables.py <start-year> <end-year> --config <config-folder-name> --schema raincells
```

where

- `<start-year>` and `<end-year>` define the range of years for which you want to create table partitiona. For example, if you want to create tables for the years 2021 to 2023, you would use `2021 2023`.
- `<config-folder-name>` is the name of the configuration folder you created in step 2 (only folder name, no prepending path).

This will create the necessary tables in the `raincells` schema of your database. For more details on the tables, see [database readme](stormcell_nowcasting/README_database.md).

### 3. Identify cells in the radar data

1. Adjust radar data settings in the `raster_datasources.yaml` file. Note that you might need to implement a importer for the format of your radar data. See [stormcell_nowcasting/stormcell_nowcasting/datasources/importers.py](stormcell_nowcasting/stormcell_nowcasting/datasources/importers.py) for examples of existing importers.
2. Adjust cell identification settings in the `identify_storm_cells.yaml` file.
3. Run cell identification (see the docstring of the script for more details):

```bash
cd stormcell_nowcasting/stormcell_nowcasting/
# Add parent path to PYTHONPATH
export PYTHONPATH=<path-to-parent-folder-of-stormcell_nowcasting>:$PYTHONPATH
python identify_storm_cells.py <start-date> <end-date> --config <config-folder-name>
```

where

- `<start-date>` and `<end-date>` define the time range for which you want to identify cells. The format is `YYYYMMDDHHMM`.
- `<config-folder-name>` is the name of the configuration folder you created in step 2 (only folder name, no prepending path).

### 4. Track the identified cells

1. Adjust cell tracking settings in the `track_storm_cells.yaml` file.
2. Run cell tracking (see the docstring of the script for more details):

```bash
cd stormcell_nowcasting/stormcell_nowcasting/
# Add parent path to PYTHONPATH
export PYTHONPATH=<path-to-parent-folder-of-stormcell_nowcasting>:$PYTHONPATH
python track_storm_cells.py <start-date> <end-date> --config <config-folder-name>
```

where

- `<start-date>` and `<end-date>` define the time range for which you want to identify cells. The format is `YYYYMMDDHHMM`.
- `<config-folder-name>` is the name of the configuration folder you created in step 2 (only folder name, no prepending path).

### 5. Create and materialize view of cell parent-child information in the database

The SQL script [`sql/create_view_cells_with_parents.sql`](sql/create_view_cells_with_parents.sql) creates a view that contains each cell's parents and children along with some additional information such as cell areas. You can run the script by using the `psql` command line tool:

```bash
psql -d <db-name> -f sql/create_view_cells_with_parents.sql
```

where `<db-name>` is the name of your database. The command might also require user and password parameters depending on your database setup.

### 6. For large datasets, determine continuous stretches of cell tracks to parallelize the cell track graph building

This can be done with the notebook [`get_time_intervals.ipynb`](notebooks/get_time_intervals.ipynb). The aim is to find continuous time intervals of cell tracks that can be processed independently, i.e. there is a gap of no cells existing between the time intervals. The length of the gap should be determined so to create a suitable amount of intervals.

### 7. Extract cell features from data and store them in the database

1. Adjust settings in the [`stormcell_nowcasting/stormcell_nowcasting/config/<config-folder>/identify_storm_cells.yaml`](stormcell_nowcasting/stormcell_nowcasting/config/template/extract_cell_raster_stats.yaml) file and the `raster_datasources.yaml` file. Note that again, you might need to implement a importer for the format of your radar data. See [stormcell_nowcasting/stormcell_nowcasting/datasources/importers.py](stormcell_nowcasting/stormcell_nowcasting/datasources/importers.py) for examples of existing importers.
2. Run the cell feature extraction script:

```bash
cd stormcell_nowcasting/stormcell_nowcasting/
# Add parent path to PYTHONPATH
export PYTHONPATH=<path-to-parent-folder-of-stormcell_nowcasting>:$PYTHONPATH
python extract_cell_raster_stats.py <start-date> <end-date> --config <config-folder-name>
```

### 8. Determine distances of cells from radars

The notebook [`calculate_cell_distances_from_radars.ipynb`](notebooks/calculate_cell_distances_from_radars.ipynb) can be used to calculate distances of cells from radars and store the distances in the database. You need to provide the radar locations in a shapefile.

### 7. Build cell track graphs

1. Adjust settings in the [`track_queries.yml`](config/plots/swiss-data/track_queries.yml) file. In particular, set the `cell_database.table` parameter to the name of the view created in step 5 and other database connection parameters to point to your database. Other settings:

- `settings.cell_identification_methods`: List of cell identification methods to include in the analysis. The methods should match the methods used in step 3.
- `output.storage_path`: Path to the folder where the output files will be stored.

2. Set database user and password as environment variables:

```bash
export DB_USER=<your-database-username>
export DB_PASSWD=<your-database-password>
```

1. Run the cell track graph building script:

```bash
python scripts/dataprocessing/query_storm_track_graphs.py --config <path-to-track_queries.yml-file> --datelist <path-to-datelist> -n <num-parallel-processes>
```

where

- `<path-to-track_queries.yml-file>` is the path to the `track_queries.yml` file you edited in step 1.
- `<path-to-datelist>` is the path to a text file that contains the start and end times of the continuous time intervals you determined in step 6. Each line in the file should contain a start and end time in the format `YYYYMMDDHHMM,YYYYMMDDHHMM`.
- `<num-parallel-processes>` is the number of parallel processes to use for building the graphs

### 10. Select $t_0$ events and build subgraph graphs for them

The subgraphs are built with the [`select_trajectories.py](scripts/dataprocessing/select_trajectories.py) and the settings are given in the [`select_trajectories.yml`](config/plots/swiss-data/select_trajectories.yml) file. The settings are described in more detail in the comments in the YAML file. Important settings include:

- `track_graphs.storagepath`: Path where the cell track graph files created in step 7 are stored.
- `track_graphs.min_duration`: Minimum duration of cell tracks to be processed.
- `trajectory_selection.select_split_merges`: Whether to store subgraphs for splits and merges.
- `trajectory_selection.select_cell_conditions`: Whether to filter cells based on conditions defined in a YAML file.
- `trajectory_selection.cell_conditions_file`: Path to the YAML file that defines the cell conditions if `select_cell_conditions` is `true`.
- `subgraphs.subgraph_storagepath`: Path where subgraph files (graph storage format) will be stored.
- `subgraphs.store_subgraphs`: Whether to store graph format files for each subgraph, e.g. for visualization.
- `output.save_interval`: Interval for saving the output files. The output files will be saved in intervals of this length.

The cell condition yaml file is expected to have the following format:

```yaml
<condition-name>:
  <feature-1>.<statistic-1>:
    min: <min-value> # Optional
    max: <max-value> # Optional
    isnull: <true-or-false> # Optional
  <feature-2>.<statistic-2>:
    min: <min-value> # Optional
    max: <max-value> # Optional
    isnull: <true-or-false> # Optional
  ...
```

where `<condition-name>` is a name for the condition, e.g. "strong-cells", `<feature-1>.<statistic-1>` is the name of a feature and statistic to filter on, e.g. "max_radar_reflectivity.max", and `min`, `max`, and `isnull` define the filtering criteria. All criteria defined under a condition must be met for a cell to be selected. The feature and statistic names must match the names used in the database.

The conditions can also be nested to allow selection of multiple groups of conditions with the same name. For example:

```yaml
<condition-name>:
  <sub-condition-name-1>:
    <feature-1>.<statistic-1>:
      min: <min-value> # Optional
      max: <max-value> # Optional
      isnull: <true-or-false> # Optional
    ...
  <sub-condition-name-2>:
    <feature-1>.<statistic-1>:
      min: <min-value> # Optional
      max: <max-value> # Optional
      isnull: <true-or-false> # Optional
    ...
```

The notebook [`calculate_t0_variable_limits.ipynb`](notebooks/calculate_t0_variable_limits.ipynb) can be used as a starting point to explore the data and determine suitable limits for the cell conditions and write the limits in the expected output format.

Run the subgraph selection script:

```bash
python scripts/dataprocessing/select_trajectories.py <path-to-select_trajectories.yml-file> <start-date> <end-date> --dbconf <path-to-database-yaml-file> -n <num-parallel-processes>
```

where

- `<path-to-select_trajectories.yml-file>` is the path to the `select_trajectories.yml` file.
- `<start-date>` and `<end-date>` define the time range for which you want to select subgraphs. The format is `YYYYMMDDHHMM`.
- `<path-to-database-yaml-file>` is the path to the database configuration YAML file (i.e. [this](config/database/database.yaml)).
- `<num-parallel-processes>` is the number of parallel processes to use for processing the data.

### 11. Plot result figures.

The result figures are plotted in the notebook [plot_article_figures.ipynb](notebooks/plot_article_figures.ipynb) and the case study figures in the notebook [plot_case_study_figures.ipynb](notebooks/plot_case_study_figures.ipynb).
