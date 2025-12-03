# WORKSHOP - 5: License Plate Detection using OpenCV and Haar Cascade Classifier
<H3>Name:Sai Ram E</H3>
<H3>Register no: 212224240141</H3>

# AIM:
To perform Gray scale Morphology Real Time Bone Fracture Detection.

# PROGRAM:
```
import cv2
import matplotlib.pyplot as plt
import os
import urllib.request

# -------------------------------------------------------------
# Step 1: Read and display the input image
# -------------------------------------------------------------
image_path = 'group.png'  # <-- Change this to your image filename
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Please check the 'image_path' variable.")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 2: Convert to grayscale
# -------------------------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 3: Preprocessing (optional)
# -------------------------------------------------------------
# Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Histogram Equalization for better contrast
equalized = cv2.equalizeHist(blurred)

plt.imshow(equalized, cmap='gray')
plt.title("Preprocessed Image (Blur + Equalized)")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 4: Load or Download Haar Cascade for Face Detection
# -------------------------------------------------------------
cascade_path = 'haarcascade_frontalface_default.xml'

# Auto-download if not present
if not os.path.exists(cascade_path):
    print("Cascade file not found. Downloading...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_path)
    print("Cascade file downloaded successfully!")

# Load classifier
face_cascade = cv2.CascadeClassifier(cascade_path)

# -------------------------------------------------------------
# Step 5: Detect faces using Haar Cascade
# -------------------------------------------------------------
faces = face_cascade.detectMultiScale(
    equalized,          # Preprocessed grayscale image
    scaleFactor=1.1,    # Scaling factor between image pyramid layers
    minNeighbors=5,     # Higher value -> fewer false detections
    minSize=(30, 30)    # Minimum object size
)

print(f"Total Faces Detected: {len(faces)}")

# -------------------------------------------------------------
# Step 6: Draw bounding boxes and save cropped faces
# -------------------------------------------------------------
output = image.copy()
save_dir = "Detected_Faces"
os.makedirs(save_dir, exist_ok=True)

for i, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    face_crop = image[y:y+h, x:x+w]
    save_path = f"{save_dir}/face_{i+1}.jpg"
    cv2.imwrite(save_path, face_crop)

if len(faces) > 0:
    print(f"{len(faces)} face(s) saved in '{save_dir}' folder.")
else:
    print("⚠️ No faces detected. Try adjusting parameters or using a clearer image.")

# -------------------------------------------------------------
# Step 7: Display the final output
# -------------------------------------------------------------
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Faces")
plt.axis('off')
plt.show()

```
# OUTPUT:
<img width="515" height="370" alt="image" src="https://github.com/user-attachments/assets/3161b8f3-80a7-4c1a-bf4f-06aaeaec68ff" />
<br>
<img width="515" height="370" alt="image" src="https://github.com/user-attachments/assets/d42b1931-3281-4871-ad94-f5d587ebdd42" />
<br>
<img width="515" height="370" alt="image" src="https://github.com/user-attachments/assets/7a08ba0a-de83-42e4-8726-02697640cfd5" />
<br>
<img width="515" height="370" alt="image" src="https://github.com/user-attachments/assets/1cc2555e-8eb6-45a5-8564-68bab53178b1" />







 
