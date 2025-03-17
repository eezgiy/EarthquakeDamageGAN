from PIL import Image
import os

def resize_images(input_folder, output_folder, size=(150, 150)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')): 
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img = img.resize(size) 
            img.save(os.path.join(output_folder, filename))  

resize_images('input_folder', 'output_folder')


