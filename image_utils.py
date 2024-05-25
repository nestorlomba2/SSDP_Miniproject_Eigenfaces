from PIL import Image
import numpy as np

def img_normalize(image_matrix):

    mean_face = np.mean(image_matrix, axis=0)
    std = np.std(image_matrix, axis=0)
    image_matrix_0 = (image_matrix - mean_face) / std

    return image_matrix_0, mean_face, std