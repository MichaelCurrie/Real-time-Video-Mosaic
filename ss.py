import cv2
import cv2.aruco as aruco
import numpy as np
import time

# ---------------------------
# Create a synthetic test image with an ArUco marker
# ---------------------------
# Choose an ArUco dictionary and marker ID.
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
marker_id = 23
marker_size = 200  # Marker image size in pixels.
# Create an image for the marker.
marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
marker_img = aruco.drawMarker(aruco_dict, marker_id, marker_size, marker_img, 1)

# Create a blank image (grayscale) and embed the marker at a fixed location.
canvas = np.zeros((500, 500), dtype=np.uint8)
x, y = 150, 150
canvas[y : y + marker_size, x : x + marker_size] = marker_img
# Convert to BGR (color) because the detector expects a color image.
test_image = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

# ---------------------------
# Benchmark detector creation
# ---------------------------
start_time = time.time()
# Create dictionary and detector parameters.
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
parameters = aruco.DetectorParameters_create()
# Create the detector instance. (Note: In later OpenCV versions you might use ArucoDetector.)
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
detector_creation_time = time.time() - start_time

# ---------------------------
# Benchmark marker detection ("prediction")
# ---------------------------
start_time = time.time()
# You can either use the detector instance or the convenience function.
# Here we use the function for detection using the same dictionary and parameters.
corners, ids, rejected = cv2.aruco.detectMarkers(
    test_image, aruco_dict, parameters=parameters
)
detection_time = time.time() - start_time

# ---------------------------
# Display benchmark results
# ---------------------------
print("Time to create the detector: {:.6f} seconds".format(detector_creation_time))
print("Time to detect markers: {:.6f} seconds".format(detection_time))
print("Detected marker IDs:", ids)

# Optionally, display the test image with detected markers drawn
if ids is not None:
    output_image = cv2.aruco.drawDetectedMarkers(test_image.copy(), corners, ids)
    cv2.imshow("Detected Markers", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
