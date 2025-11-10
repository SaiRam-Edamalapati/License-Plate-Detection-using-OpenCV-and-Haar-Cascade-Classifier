# WORKSHOP - 5: License Plate Detection using OpenCV and Haar Cascade Classifier
<H3>Name: DHANUSH C</H3>
<H3>Register no: 212224040066</H3>
<H3>Date: 10.11.2025 </H3>

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

<img width="425" height="512" alt="Screenshot 2025-11-10 111526" src="https://github.com/user-attachments/assets/85c40ee8-5196-4fa0-be81-687f633f49b7" />
<br>
<img width="391" height="507" alt="Screenshot 2025-11-10 111551" src="https://github.com/user-attachments/assets/88f94960-83e3-4145-adae-97fcaca87a79" />
<br>
<img width="420" height="512" alt="Screenshot 2025-11-10 111607" src="https://github.com/user-attachments/assets/e5417685-22d4-47b3-af66-769be8b82aa9" />
<br>
<img width="418" height="558" alt="Screenshot 2025-11-10 111626" src="https://github.com/user-attachments/assets/1f5f2500-3479-47c8-a246-74e84bd2f051" />







 
