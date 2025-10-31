# Sakshi Bhatt(EXP 7)
from deepface import DeepFace
import cv2

# Paths to my two images
img1_path = "person1.jpeg"
img2_path = "person3.jpeg"

# Verifying if both images belong to the same person
result = DeepFace.verify(img1_path, img2_path, detector_backend='opencv')  # using OpenCV detector for fewer errors

print(result)

if result["verified"]:
    print("✅ Same person")
else:
    print(" Different persons")


img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)


if img1 is None or img2 is None:
    raise ValueError(" Could not load one of the images. Check the file paths.")

height = 400
img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))

if len(img1.shape) == 2:
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
if len(img2.shape) == 2:
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

# Concatenating 
combined = cv2.hconcat([img1, img2])

# Putting result text on the image
text = "Match" if result["verified"] else "Not Match"
color = (0, 255, 0) if result["verified"] else (0, 0, 255)
cv2.putText(combined, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Show window
cv2.imshow("Face Comparison", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
