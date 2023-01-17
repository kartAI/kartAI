from typing import List
import cv2
import numpy as np
from PIL import Image, ImageOps
from osgeo import gdal
import os
from kartai.dataset.format_conversion import gdal_to_np, image_to_np, np_to_image, np_to_gdal_mem


def merge_raster_paths(paths, normalize_data=False, save_to_disk=False, output_dir=None, output_name=None):
    data_to_merge = []
    for path in paths:
        ds = gdal.Open(path)
        data = gdal_to_np(ds)
        if normalize_data:
            data = cv2.normalize(data, None, alpha=0, beta=1,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        target = np_to_gdal_mem(data, ds)
        data_to_merge.append(target)

    mosaic_res = merge_data(data_to_merge, save_to_disk,
                            output_dir, output_name)
    return mosaic_res


def merge_raster_np_images(np_images, predictions_path, save_to_disk=False, output_dir=None, output_name=None):
    data_to_merge = []
    for i in range(len(predictions_path)):
        np_image = np_images[i]
        # Get corresponding prediction to set the correct transformation and projection
        input_ds = gdal.Open(predictions_path[i])
        target = np_to_gdal_mem(np_image, input_ds)
        data_to_merge.append(target)

    mosaic_res = merge_data(data_to_merge, save_to_disk,
                            output_dir, output_name)
    return mosaic_res


def merge_data(data_to_merge, save_to_disk, output_dir, output_name):

    if(save_to_disk == False):
        mosaic_res = gdal.Warp("MEM", data_to_merge, format="MEM",
                               options=["COMPRESS=LZW", "TILED=YES"], outputType=gdal.GDT_Float32)
        return mosaic_res
    else:
        mosaic_res = gdal.Warp(os.path.join(output_dir, output_name+'.tif'), data_to_merge, format='GTiff', options=[
            "COMPRESS=LZW", "TILED=YES"], outputType=gdal.GDT_Float32)
        return mosaic_res


def get_adjacent_images(path, paths: List[str]):
    center_x, center_y = get_X_Y_from_path(path)
    images = [[None for i in range(3)] for j in range(3)]
    for x in range(center_x-1, center_x+1+1):
        for y in range(center_y-1, center_y+1+1):
            if x == center_x and y == center_y:
                image_path = path
            else:
                image_path: str = get_path(paths, x, y)

            x_index = x-center_x+1
            y_index = 2-(y-center_y+1)
            if image_path == None:
                images[y_index][x_index] = None
            else:
                image = Image.open(image_path)
                images[y_index][x_index] = image
    
    return images


def get_path(paths: List[str], x: int, y: int) -> str:
    # Consider making a matrix of paths for more efficient search
    for path in paths:
        path_x, path_y = get_X_Y_from_path(path)
        if path_x == x and path_y == y:
            return path
    return None


def get_X_Y_from_path(path: str):
    filename: str = os.path.split(path)[-1]
    x, y = filename.split("_")[0], filename.split("_")[1]
    return int(x), int(y)


def remove_noise_batch(predictions_path_batch: List[str], all_prediction_paths: List[str], minimize_radius: int) -> List[np.ndarray]:
    np_denoised_predictions = []

    for path in predictions_path_batch:
        images: List = get_adjacent_images(path, all_prediction_paths)
        noise_removed_prediction_image = remove_noise(images, minimize_radius)
        np_denoised_predictions.append(
            image_to_np(noise_removed_prediction_image))
    
    return np_denoised_predictions


def remove_noise(images: list, minimize_radius: int) -> Image:
    # Creating a two pass minimize

    # Create RGB to test if expand function works
    
    adjecent_border: int = 50

    img: Image = images[1][1] # Get the center image
    size: int = img.width
    np_image_with_adjacent = np.zeros((size*3, size*3))
    for x in range(len(images)):
        for y in range(len(images)):
            if images[y][x] == None:
                np_image = np.zeros((size, size))
            else:
                np_image = image_to_np(images[y][x])
            np_image_with_adjacent[y*size:y*size+size, x*size:x*size+size] = np_image

    # Crop image to improve minimize performance
    np_image_with_adjacent_croped = image_to_np(center_crop(np_to_image(np_image_with_adjacent), size - adjecent_border))
    
    # Normalize input image:
    normalized_img_arr = cv2.normalize(np_image_with_adjacent_croped, None, alpha=0, beta=1,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    normalized_image = np_to_image(normalized_img_arr)

    # img to read values from - never changed
    padded_image = ImageOps.expand(        
        normalized_image, border=minimize_radius, fill=0)

    vert_minimized_img = vertical_minimize(padded_image, minimize_radius)
    fully_minimized_img = horizontal_minimize(
        vert_minimized_img, minimize_radius)
    buffer_img = box_buffer(fully_minimized_img, minimize_radius-1)
    return_img = center_crop(buffer_img, adjecent_border+minimize_radius)
    return return_img


def box_buffer(img, buffer_radius):
    vert_buffered_img = linear_buffer(img, buffer_radius, True)
    fully_buffered_img = linear_buffer(vert_buffered_img, buffer_radius, False)
    return fully_buffered_img


def box_blur(img, blur_radius, pad_border):
    # Creating a two pass box blur
    # pad image to perform correct calc on edges - removed again after blurring
    vert_blurred_img = linear_blur(img, blur_radius, pad_border, True)
    fully_blurred_img = linear_blur(
        vert_blurred_img, blur_radius, pad_border, False)
    return fully_blurred_img


def linear_blur(input_img_padded, blur_radius, pad_border, run_horizontal):
    # input_img_padded = ImageOps.expand(img, border=blur_radius, fill = "rgb(0,0,0)") #img to read values from - never changed
    input_pixels = input_img_padded.load()
    return_img = Image.new("L", input_img_padded.size, "rgb(0,0,0)")
    return_pixels = return_img.load()

    pix_val = 0
    rows = input_img_padded.size[0]-pad_border
    cols = input_img_padded.size[1]-pad_border
    for x in range(pad_border, rows):
        for y in range(pad_border, cols):
            pix_val = 0
            for i in range(-blur_radius, blur_radius+1):
                if(run_horizontal):
                    pix_val += input_pixels[x+i, y]
                else:  # run vertically
                    pix_val += input_pixels[x, y+i]
            return_pixels[x, y] = pix_val // ((blur_radius*2)+1)

    return return_img


def linear_buffer(img_padded, buffer_radius, run_horizontal):

    # input_img_padded = ImageOps.expand(img, border=buffer_radius, fill = "rgb(0,0,0)") #img to read values from - never changed
    input_pixels = img_padded.load()
    return_img = Image.new("L", img_padded.size, "rgb(0,0,0)")
    return_pixels = return_img.load()
    current_pixel = input_pixels[0, 0]  # init value

    rows = img_padded.size[0]-buffer_radius
    cols = img_padded.size[1]-buffer_radius

    for x in range(buffer_radius, rows):  # img.size = (512,512)
        for y in range(buffer_radius, cols):
            for i in range(-buffer_radius, buffer_radius):
                if(run_horizontal):  # first two timws blur in x direction
                    current_pixel = input_pixels[x+i, y]
                else:  # run vertically
                    current_pixel = input_pixels[x, y+i]
                if(current_pixel > 0):  # if white pixel
                    return_pixels[x, y] = 1  # white
                    break

    return return_img


def vertical_minimize(img, radius):
    pad_border = radius
    input_pixels = img.load()
    return_img = Image.new("L", img.size)
    return_pixels = return_img.load()

    rows = img.size[0]-pad_border
    cols = img.size[1]-pad_border
    start_index = [0, 0]
    end_index = [0, 0]
    for y in range(pad_border, cols):
        for x in range(pad_border, rows):
            if(input_pixels[x, y] == 1):
                start_index = (x, y)
                end_index = (x, y+1)
                while(input_pixels[end_index] == 1):
                    # move vertical
                    end_index = (end_index[0], end_index[1]+1)
                    test = input_pixels[end_index]
                if(end_index[1] - start_index[1] > ((2*radius) + 1)):
                    for y_keep in range(start_index[1]+radius, end_index[1]-radius):
                        return_pixels[x, y_keep] = 1
                x = end_index[0]+1  # skip to next unhandled pixel
    return return_img


def horizontal_minimize(img, radius):
    pad_border = radius
    input_pixels = img.load()
    return_img = Image.new("L", img.size, "rgb(0,0,0)")
    return_pixels = return_img.load()

    pix_val = 0
    rows = img.size[0]-pad_border
    cols = img.size[1]-pad_border
    start_index = [0, 0]
    end_index = [0, 0]
    for x in range(pad_border, rows):
        for y in range(pad_border, cols):
            if(input_pixels[x, y] == 1):
                start_index = (x, y)
                end_index = (x+1, y)
                while(input_pixels[end_index] == 1):
                    # move horizontal
                    end_index = (end_index[0]+1, end_index[1])
                if(end_index[0] - start_index[0] > ((2*radius) + 1)):
                    for x_keep in range(start_index[0]+radius, end_index[0]-radius):
                        return_pixels[x_keep, y] = 1
                y = end_index[1]+1  # skip to next unhandled pixel
    return return_img


def center_crop(img, border_size):
    bbox = (border_size, border_size,
            img.size[0] - border_size, img.size[1] - border_size)
    cropped_img = img.crop(bbox)
    return cropped_img
