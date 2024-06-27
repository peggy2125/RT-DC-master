import cv2
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor

def trace_contours(edge_image):
    contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return [contour.reshape(-1, 2) for contour in contours]

def load_and_blur_image(image_path, kernel_size=(5, 5)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return image, blurred_image

def show_image(name, img, images_dict):
    images_dict[name] = img

def process_image(image_path, background_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    images_dict = {}
    show_image('raw', image, images_dict)
    
    start_time = time.time()
    
    with ThreadPoolExecutor() as executor:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        executor.submit(show_image, 'blurred', blurred, images_dict)

        bg_sub = cv2.subtract(blurred_bg, blurred)
        executor.submit(show_image, 'bg_sub', bg_sub, images_dict)

        _, binary = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)
        executor.submit(show_image, 'binary', binary, images_dict)

        dilate1 = cv2.dilate(binary, kernel, iterations=2)
        executor.submit(show_image, 'dilate1', dilate1, images_dict)

        erode1 = cv2.erode(dilate1, kernel, iterations=2)
        executor.submit(show_image, 'erode1', erode1, images_dict)

        erode2 = cv2.erode(erode1, kernel, iterations=1)
        executor.submit(show_image, 'erode2', erode2, images_dict)

        dilate2 = cv2.dilate(erode2, kernel, iterations=1)
        executor.submit(show_image, 'dilate2', dilate2, images_dict)

        edges = cv2.Canny(dilate2, 50, 150)
        executor.submit(show_image, 'canny edges', edges, images_dict)

        contours = trace_contours(edges)

    end_time = time.time()
    dif_time = end_time - start_time
    print(dif_time)

    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (255), 1)
    show_image('Processed Image', contour_image, images_dict)

    while True:
        for name, img in images_dict.items():      
            cv2.imshow(name, img)
        
        key = cv2.waitKey(1)
        if key!=-1:          #按下任意鍵
            break

    cv2.destroyAllWindows()

# 使用方法
process_image('Test_images/Slight under focus/0066.tiff', 'Test_images/Slight under focus/background.tiff')



# Set the directory containing your files
directory = 'Test_images/Slight under focus'
 # Get a list of all tiff files
files = [f for f in os.listdir(directory) if f.endswith('.tiff')]
for image in files:
    image_path = os.path.join(directory, image)
    print(f"Processing: {image_path}")
    process_image(image_path, 'Test_images/Slight under focus/background.tiff')
    print("Press any key to continue to the next image...")
    