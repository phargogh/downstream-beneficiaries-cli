# coding=UTF-8
import argparse
import logging
import os

import numpy
import pygeoprocessing
import pygeoprocessing.routing
import taskgraph
from osgeo import gdal

logging.basicConfig(level=logging.INFO)
FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
BYTE_NODATA = 255
LOGGER = logging.getLogger(__name__)


def mask_areas_of_interest(aoi_path, target_path):
    aoi_nodata = pygeoprocessing.get_raster_info(aoi_path)['nodata'][0]

    def _convert(array):
        result = numpy.full(array.shape, 0, dtype=numpy.uint8)
        result[array == 1] = 1
        result[numpy.isclose(array, aoi_nodata)] = BYTE_NODATA
        return result

    pygeoprocessing.raster_calculator(
        [(aoi_path, 1)], _convert, target_path, gdal.GDT_Byte, BYTE_NODATA)


def convert_population_units(
        population_raster, target_population_density, to_density=True):

    # assume linearly projected
    # TODO: how would we handle some other projection, where pixels are not
    # equal in size?
    raster_info = pygeoprocessing.get_raster_info(population_raster)
    pixel_area = abs(raster_info['pixel_size'][0] *
                     raster_info['pixel_size'][1])
    pop_nodata = raster_info['nodata'][0]

    def _convert(pop_array):
        result = numpy.full(pop_array.shape, FLOAT32_NODATA,
                            dtype=numpy.float32)
        valid_mask = ~numpy.isclose(pop_array, pop_nodata)
        if to_density:
            result[valid_mask] = pop_array[valid_mask] / pixel_area
        else:
            result[valid_mask] = pop_array[valid_mask] * pixel_area
        return result

    pygeoprocessing.raster_calculator(
        [(population_raster, 1)], _convert, target_population_density,
        gdal.GDT_Float32, FLOAT32_NODATA)


def calculate_downstream_beneficiaries(
        dem_path, population_path, areas_of_interest_path, workspace_dir,
        n_workers=-1):

    if not os.path.join(workspace_dir):
        os.makedirs(workspace_dir)

    graph = taskgraph.TaskGraph(
        os.path.join(workspace_dir, '.taskgraph'),
        n_workers=n_workers)

    # raster_calculator: convert the population to population per unit area
    population_density_path = os.path.join(
        workspace_dir, f'population_density.tif')
    pop_density_task = graph.add_task(
        convert_population_units,
        args=(population_path, population_density_path, True),
        target_path_list=[population_density_path],
        task_name='convert population count to density',
        dependent_task_list=[]
    )

    # align the input raster stack
    aligned_dem_path = os.path.join(
        workspace_dir, 'aligned_dem.tif')
    aligned_population_density_path = os.path.join(
        workspace_dir, 'aligned_population_density.tif')
    aligned_areas_of_interest_path = os.path.join(
        workspace_dir, 'aligned_areas_of_interest.tif')
    dem_info = pygeoprocessing.get_raster_info(dem_path)
    align_task = graph.add_task(
        pygeoprocessing.align_and_resize_raster_stack,
        kwargs={
            'base_raster_path_list': [
                dem_path, population_path, areas_of_interest_path],
            'target_raster_path_list': [
                aligned_dem_path, aligned_population_density_path,
                aligned_areas_of_interest_path],
            'resample_method_list': ['bilinear', 'bilinear', 'near'],
            'target_pixel_size': dem_info['pixel_size'],
            'bounding_box_mode': 'intersection',
            'raster_align_index': 0,
            'target_projection_wkt': dem_info['projection_wkt']
        },
        target_path_list=[
            aligned_dem_path, aligned_population_density_path,
            aligned_areas_of_interest_path,
        ],
        task_name='align rasters',
        dependent_task_list=[pop_density_task]
    )

    # raster_calculator: mask to create raster of 1-value and 0-values.
    masked_areas_of_interest_path = os.path.join(
        workspace_dir, 'masked_areas_of_interest.tif')
    masked_aoi_task = graph.add_task(
        mask_areas_of_interest,
        args=(aligned_areas_of_interest_path, masked_areas_of_interest_path),
        target_path_list=[masked_areas_of_interest_path],
        task_name='mask areas of interest',
        dependent_task_list=[align_task]
    )

    # raster_calculator: convert the population back to population count
    population_count_path = os.path.join(
        workspace_dir, f'aligned_population_count.tif')
    pop_density_task = graph.add_task(
        convert_population_units,
        args=(population_path, population_density_path, False),
        target_path_list=[population_density_path],
        task_name='convert population density to count',
        dependent_task_list=[align_task]
    )

    # fill the dem
    filled_dem_path = os.path.join(
        workspace_dir, 'filled_dem.tif')
    filled_dem_task = graph.add_task(
        pygeoprocessing.routing.fill_pits,
        args=((aligned_dem_path, 1), filled_dem_path, workspace_dir),
        target_path_list=[filled_dem_path],
        task_name='fill pits',
        dependent_task_list=[align_task]
    )

    # flow_direction
    flow_dir_path = os.path.join(
        workspace_dir, 'flow_dir_mfd.tif')
    flow_dir_task = graph.add_task(
        pygeoprocessing.routing.flow_dir_mfd,
        args=((filled_dem_path, 1), flow_dir_path, workspace_dir),
        target_path_list=[flow_dir_path],
        task_name='flow direction mfd',
        dependent_task_list=[filled_dem_task]
    )

    # weighted flow accumulation.
    flow_accum_path = os.path.join(
        workspace_dir, 'flow_accumulation.tif')
    _ = graph.add_task(
        pygeoprocessing.routing.flow_accumulation_mfd,
        args=((flow_dir_path, 1), flow_accum_path,
              (masked_areas_of_interest_path, 1)),
        target_path_list=[flow_accum_path],
        task_name='weighted flow accumulation',
        dependent_task_list=[flow_dir_task, masked_aoi_task]
    )


def main():
    pass


if __name__ == '__main__':
    calculate_downstream_beneficiaries(
        'DEM_Colombia300m.tif',
        'LandscanPopulation2017_Colombia.tif',
        'MaskServiceProvHotspots.tif',
        'downstream-beneficiaries-workspace')
