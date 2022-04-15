# coding=UTF-8
import argparse
import logging
import os

import numpy
import pygeoprocessing
import taskgraph
from osgeo import gdal

logging.basicConfig(level=logging.INFO)
FLOAT32_NODATA = numpy.finfo(numpy.float32).min
LOGGER = logging.getLogger(__name__)


def convert_to_population_per_unit_area(
        population_raster, target_population_density):

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
        result[valid_mask] = pop_array[valid_mask] / pixel_area
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
        convert_to_population_per_unit_area,
        args=(population_path, population_density_path),
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
    # raster_calculator: convert the population back to population count
    # fill the dem
    # flow_direction
    # weighted flow accumulation.
    # done.




def main():
    pass


if __name__ == '__main__':
    main()
