import cv2
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor

def trace_contours(edge_image):
    # 使用 OpenCV 的 findContours 函數
    contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # 將輪廓轉換為所需的格式
    return [contour.reshape(-1, 2) for contour in contours]


def load_and_blur_image(image_path, kernel_size=(5, 5)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return image, blurred_image
def show_image(name, img):
    cv2.imshow(name, img) 
    #cv2.waitKey(0)   
def process_image(image_path, background_path):
    # 以灰度模式加载图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cv2.imshow('raw', image)
    start_time = time.time()
    
    with ThreadPoolExecutor() as executor:
        executor.submit(show_image, 'raw', image)
        
        # Apply Gaussian blur to smooth the image
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        executor.submit(show_image, 'blurred', blurred)

        # Background subtraction
        bg_sub = cv2.subtract(blurred_bg, blurred)
        executor.submit(show_image, 'bg_sub', bg_sub)

        # Apply threshold
        _, binary = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)
        executor.submit(show_image, 'binary', binary)

        # Erode and dilate to remove noise
        dilate1 = cv2.dilate(binary, kernel, iterations=2)
        executor.submit(show_image, 'dilate1', dilate1)

        erode1 = cv2.erode(dilate1, kernel, iterations=2)
        executor.submit(show_image, 'erode1', erode1)

        erode2 = cv2.erode(erode1, kernel, iterations=1)
        executor.submit(show_image, 'erode2', erode2)

        dilate2 = cv2.dilate(erode2, kernel, iterations=1)
        executor.submit(show_image, 'dilate2', dilate2)

        # Apply Canny edge detector to find edges
        edges = cv2.Canny(dilate2, 50, 150)
        executor.submit(show_image, 'canny edges', edges)
        # Trace contours from the edge image
        contours = trace_contours(edges)

        end_time = time.time()
        dif_time = end_time - start_time
        print(dif_time)
    # Prepare an image to draw the contours
    contour_image = np.zeros_like(image)
    
    # Draw each contour
    cv2.drawContours(contour_image, contours, -1, (255), 1)

    # Show the resulting image
    cv2.imshow('Processed Image', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Replace 'path_to_image.tif' with your image file path
process_image('Test_images/Slight under focus/0066.tiff', 'Test_images/Slight under focus/background.tiff')

# Set the directory containing your files
# directory = 'Test_images/Slight under focus'
# # Get a list of all tiff files
# files = [f for f in os.listdir(directory) if f.endswith('.tiff')]
# for image in files:
#     image_path = os.path.join(directory, image)
#     print(image_path)
#     process_image(image_path, 'Test_images/Slight under focus/background.tiff')