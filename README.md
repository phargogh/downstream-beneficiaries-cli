# downstream-beneficiaries-cli

Count the number of people downstream of pixels of interest.

## Installation

```
conda create -p ./env python=3.9 -c conda-forge pygeoprocessing python=3.9 taskgraph
conda activate ./env
```

## Running the Program

```
python downstream-beneficiaries.py \
    --parallelize=1 \
    --dem=DEM_Colombia300m.tif \
    --population=LandscanPopulation2017_Colombia.tif \
    --areas-of-interest=MaskServiceProvHotspots.tif \
    ./downstream-beneficiaries-workspace
```

### Inputs:

1. DEM: A raster indicating elevation of each pixel. This raster must be
   linearly projected (e.g. into meters) and have square pixels.
2. Population: A raster of population counts (people per pixel), also in a
   linear projection.
   * NOTE: This tool only counts what is along a flow path. Be sure that the
   population raster has population appropriately attributed to pixels that are
   along a relevant flow path. This is particularly relevant for D8 routing
   because of its 1-pixel stream widths.
3. Areas of Interest: A raster indicating pixels of interest. Pixel values of
   1 are considered pixels of interest. Any other pixel values (including
   nodata) are pixels of non-interest.
4. `--parallelize`: if `1`, run on 2 CPU cores. If `0`, run on 1 core.
5. Routing algorithm: whether to run this with D8 or MFD routing. Default: MFD.
6. Workspace: A directory where output rasters should be stored.


### Outputs:

**The total population count is printed to standard output.**

The tool also produces several files in the workspace:

* The input rasters are warped to match the DEM's projection and pixel size and
  the intersection of the datasets' bounding boxes:
  * `aligned_dem.tif` is the aligned DEM.
  * `aligned_areas_of_interest.tif` is the aligned areas of interest raster.
  * In order to maintain consistent population counts during warping, the
    population raster is first converted to a population density before it is
    warped.
    * `population_density.tif` is the population raster, converted to people
      per unit area.
    * `aligned_population_density.tif` is the aligned version of
      `population_density.tif`, which has also been bilinearly interpolated.
    * `aligned_population_count.tif` is the result of taking
      `aligned_population_density.tif` and converting its units back to
      population counts per pixel.
* `filled_dem.tif` is a pit-filled version of `aligned_dem.tif`.
* `flow_dir_{MFD/D8}.tif` is the result of calculating flow direction on
  `filled_dem.tif` using pygeoprocessing's MFD or D8 routing.
* `flow_accumulation.tif` is the flow accumulation using the provided areas of
  interest as a weight. Thus, pixels with a value > 0 are downstream of pixels
  of interest.
* `pop_downstream_of_areas_of_interest.tif` identifies which aligned population
  pixels are downstream of the areas of interest.


## About this script

This came about in late 2021 as a request to the software team from @lmandle
and @jagoldstein as a utility that would have been useful in a project for IDB
and would be useful for an upcoming project in Sri Lanka.

The original software team issue is documented here: https://github.com/natcap/softwareteam/issues/125

The approach taken is based on the weighted flow accumulation pygeoprocessing pipeline described here:
https://github.com/therealspring/downstream-beneficiaries/blob/main/downstream_beneficiaries.py,
but the script in this repo is adapted to run as a CLI application instead of
as a global pipeline.
