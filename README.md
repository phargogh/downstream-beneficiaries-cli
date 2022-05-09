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

## About this script

This came about in late 2021 as a request to the software team from @lmandle
and @jagoldstein as a utility that would have been useful in a project for IDB
and would be useful for an upcoming project in Sri Lanka.

The original software team issue is documented here: https://github.com/natcap/softwareteam/issues/125

The approach taken is based on an approach described here:
https://github.com/therealspring/downstream-beneficiaries/blob/main/downstream_beneficiaries.py,
but the script in this repo is adapted to run as a CLI application.
