from PIL import Image, ImageOps
import numpy as np
import os

def load_images(dataset):

    flattened_images = []
    labels = []

    if dataset == "YaleB":
        for filename in os.listdir("data/YaleB"):
            file_path = os.path.join("data/YaleB", filename)
            with Image.open(file_path) as img:
                flattened_img = np.array(img).flatten()
                flattened_images.append(flattened_img)
                # Extract the class label from the filename
                # Example filename: yaleB01_P00A+000E+00.png
                # Extract '01' from 'yaleB01', which is the subject ID
                label = filename.split('_')[0]
                label = label[5:7]
                labels.append(label)

        # Convert lists to numpy arrays
        image_matrix = np.array(flattened_images)
        label_array = np.array(labels)
        img_size = img.size

    elif dataset == "lfwcrop_grey":
        for filename in os.listdir("data/lfwcrop_grey"):
            file_path = os.path.join("data/lfwcrop_grey", filename)
            with Image.open(file_path) as img:
                flattened_img = np.array(img).flatten()
                flattened_images.append(flattened_img)
                # Extract the class label from the filename
                # Example filename: yaleB01_P00A+000E+00.png
                # Extract '01' from 'yaleB01', which is the subject ID
                label = "_".join(filename.split('.')[0].split('_')[:-1])
                labels.append(label)

        # Convert lists to numpy arrays
        image_matrix = np.array(flattened_images)
        label_array = np.array(labels)
        img_size = img.size
    elif dataset == "vggface2_224":
        NUMBER_OF_PEOPLE = 20
        for person_count, person_folder in enumerate(os.listdir("data/vggface2_224")):
            if person_count == NUMBER_OF_PEOPLE:
                break
            for filename in os.listdir(f"data/vggface2_224/{person_folder}"):
                file_path = os.path.join(f"data/vggface2_224/{person_folder}", filename)
                with Image.open(file_path) as img:
                    flattened_img = np.array(ImageOps.grayscale(img)).flatten()
                    flattened_images.append(flattened_img)
                    label = person_folder
                    labels.append(label)
            print(f"Loaded: {person_folder}")
            

        # Convert lists to numpy arrays
        image_matrix = np.array(flattened_images)
        label_array = np.array(labels)
        img_size = img.size
    
    return image_matrix, label_array, (img_size[1],img_size[0]) # (H,W) instead of (W,H)
