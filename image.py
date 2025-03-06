import cv2
import numpy as np

def main():
    image = cv2.imread("dog_image.jpeg")
    if image is None:
        print("Error: Image not found!")
        return

    # 1. Image Rotation
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    # 2. Image Scaling
    scaled = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    
    # 3. Adding Border
    bordered = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 0, 0])
    
    # 4. Image Flipping
    flipped = cv2.flip(image, 1)
    
    # 5. Grayscale Conversion
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 6. Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5,5), 0)
    
    # 7. Canny Edge Detection
    edges = cv2.Canny(image, 100, 200)
    
    # 8. Erosion
    eroded = cv2.erode(image, np.ones((5,5), np.uint8), iterations=1)
    
    # 9. Dilation
    dilated = cv2.dilate(image, np.ones((5,5), np.uint8), iterations=1)
    
    # 10. Histogram Equalization
    equalized = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    # 11. Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    

    # 12. Translation (Shifting)
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, 50], [0, 1, 50]])
    translated = cv2.warpAffine(image, M, (cols, rows))
    
    # 13. Perspective Transform
    pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250], [250, 200]])
    M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
    perspective = cv2.warpPerspective(image, M_perspective, (cols, rows))
    
    # 14. Bitwise AND
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cols//2, rows//2), 100, (255), -1)
    bitwise_and = cv2.bitwise_and(image, image, mask=mask)
    
    # 15. Bitwise NOT
    bitwise_not = cv2.bitwise_not(image)
    
    # Display images one by one
    transformations = [
        ("Rotated", rotated),
        ("Scaled", scaled),
        ("Bordered", bordered),
        ("Flipped", flipped),
        ("Grayscale", grayscale),
        ("Blurred", blurred),
        ("Edges", edges),
        ("Eroded", eroded),
        ("Dilated", dilated),
        ("Equalized", equalized),
        ("Adaptive Threshold", adaptive_thresh),
        ("Translated", translated),
        ("Perspective", perspective),
        ("Bitwise AND", bitwise_and),
        ("Bitwise NOT", bitwise_not)
    ]
    
    for name, img in transformations:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()