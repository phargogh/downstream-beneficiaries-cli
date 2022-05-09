# downstream-beneficiaries-cli

Count the number of people downstream of pixels of interest.

## Installation

```
conda create -p ./env python=3.9 -c conda-forge pygeoprocessing python=3.9 taskgraph
conda activate ./env
```

## Running the Program

```
python downstream-beneficiaries.py --parallelize=1 --dem=DEM_Colombia300m.tif --population=LandscanPopulation2017_Colombia.tif --areas-of-interest=MaskServiceProvHotspots.tif ./downstream-beneficiaries-workspace
```
