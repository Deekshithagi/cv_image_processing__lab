import cv2
import numpy as np

def main():
    image = cv2.imread("dog.jpeg")
    if image is None:
        print("Error: Image not found!")
        return
    
    # 1. Image Rotation
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    # 2. Image Scaling
    scaled = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    
    # 3. Translation (Shifting)
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, 50], [0, 1, 50]])
    translated = cv2.warpAffine(image, M, (cols, rows))
    
    transformations = [
        ("Rotated", rotated),
        ("Scaled", scaled),
        ("Translated", translated)
      
    ]
    
    for name, img in transformations:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()