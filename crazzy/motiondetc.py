import cv2

# Create VideoCapture object to read from the webcam
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame1 = cap.read()

# Convert the frame to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Infinite loop for continuous motion detection
while True:
    # Read the current frame
    ret, frame2 = cap.read()

    # Convert the frame to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(gray1, gray2)

    # Apply a threshold to the frame difference
    _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill in the holes
    dilated = cv2.dilate(threshold, None, iterations=3)

    # Find contours of the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around the contours representing moving objects
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 500:
            continue
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame with motion detection
    cv2.imshow('Motion Detection', frame2)

    # Update the previous frame
    gray1 = gray2

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
