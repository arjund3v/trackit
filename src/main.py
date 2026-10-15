# Using OpenCV to capture video from the webcam and draw circles
import cv2 

# Using mediapipe to detect face landmarks, using the new mediapipe tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Using numpy to perform matrix operations, which will be used to calculate the head pose
import numpy as np

# This is a path to the model directly from google, this model tells mediapipe how to detect face landmarks
# without it the program will not work or know how to detect face landmarks
MODEL_PATH = "models/face_landmarker.task"


def main():

    # Basic setup/config for media pipe
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    # Create specified options for the landmarker, which will be used when intializing the landmarker object
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    # Create the actual object
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # Open the webcam using OpenCV, the 0 just means the default webcam.
    camera = cv2.VideoCapture(0)

    # If the camera wasn't opened successfully, raise an error and exit the program
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")

    # Initialize a timestamp variable to keep track of the time in milliseconds for each frame
    timestamp_ms = 0

    # Primary loop to read frames from the webcam, process them, and display the results
    # Continues to run until the user presses the 'q' key or the webcam feed is interrupted
    while True:

        # Allows us to read a frame from the webcam, and the success flag
        success, frame = camera.read()

        # Here we invert the frame horizontally, so it looks like a mirror.
        frame = cv2.flip(frame, 1)

        if not success:
            break

        # OpenCV uses BGR format by default
        # Mediapipe expects RGB format
        # We convert the frame from BGR to RGB before passing it to the landmarker
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a mediapipe Image object from the RGB frame, which is required for the landmarker to process the image
        # The image object is used by the landmarker to detect face landmarks in the current frame. Landmarker
        # doesn't use opencv frames, it wants its own image object, hence the converesion.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Use the landmarker to detect face landmarks in the current frame
        # Send the current frame, into the AI model, media pipe looks at the face in that exact frame and 
        # returns the result. (Find face, return landmarks to the result variable)
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # We are estimating a 60fps stream, so we increment the timestamp by 16 milliseconds for each frame. 
        # This is a rough estimate and may not be accurate for all webcams or lighting conditions.
        timestamp_ms += 16


        if result.face_landmarks:
            # If face landmarks are detected, we extract the face itself first
            # Landmarks becomes a list of facial points 
            landmarks = result.face_landmarks[0]

            # Get the height and width of the frame, which will be used to convert the normalized 
            # landmark coordinates to pixel coordinates
            height, width, _ = frame.shape
            
            chin = landmarks[152]
            nose_tip = landmarks[4]
            left_eye_inner_corner = landmarks[154]
            right_eye_inner_corner = landmarks[463]
            left_mouth_corner = landmarks[146]
            right_mouth_corner = landmarks[375]
            left_cheekbone = landmarks[116]
            right_cheekbone = landmarks[345]

            # Convert normalized coordinates to pixel coordinates
            chin_x, chin_y = int(chin.x * width), int(chin.y * height)
            nose_tip_x, nose_tip_y = int(nose_tip.x * width), int(nose_tip.y * height)
            left_eye_inner_corner_x, left_eye_inner_corner_y = int(left_eye_inner_corner.x * width), int(left_eye_inner_corner.y * height)
            right_eye_inner_corner_x, right_eye_inner_corner_y = int(right_eye_inner_corner.x * width), int(right_eye_inner_corner.y * height)
            left_mouth_corner_x, left_mouth_corner_y = int(left_mouth_corner.x * width), int(left_mouth_corner.y * height)
            right_mouth_corner_x, right_mouth_corner_y = int(right_mouth_corner.x * width), int(right_mouth_corner.y * height)
            left_cheekbone_x, left_cheekbone_y = int(left_cheekbone.x * width), int(left_cheekbone.y * height)
            right_cheekbone_x, right_cheekbone_y = int(right_cheekbone.x * width), int(right_cheekbone.y * height)

            # Draw lines between the landmarks 
            LINE_COLOUR = (0, 255, 0)  # Green color in BGR

            # line from chin to nose tip
            cv2.line(frame, (chin_x, chin_y), (nose_tip_x, nose_tip_y), LINE_COLOUR, 2) 

            # line from nose tip to left eye inner corner
            cv2.line(frame, (nose_tip_x, nose_tip_y), (left_eye_inner_corner_x, left_eye_inner_corner_y), LINE_COLOUR, 2)

            # line from nose tip to right eye inner corner
            cv2.line(frame, (nose_tip_x, nose_tip_y), (right_eye_inner_corner_x, right_eye_inner_corner_y), LINE_COLOUR, 2)

            # line from nose tip to left cheekbone
            cv2.line(frame, (nose_tip_x, nose_tip_y), (left_cheekbone_x, left_cheekbone_y), LINE_COLOUR, 2)

            # line from nose tip to right cheekbone
            cv2.line(frame, (nose_tip_x, nose_tip_y), (right_cheekbone_x, right_cheekbone_y), LINE_COLOUR, 2)

            # line from left mouth corner to right mouth corner
            cv2.line(frame, (left_mouth_corner_x, left_mouth_corner_y), (right_mouth_corner_x, right_mouth_corner_y), LINE_COLOUR, 2)  

            image_points = [
                (nose_tip_x, nose_tip_y),  # Nose tip
                (chin_x, chin_y),  # Chin
                (left_eye_inner_corner_x, left_eye_inner_corner_y),  # Left eye inner corner
                (right_eye_inner_corner_x, right_eye_inner_corner_y),  # Right eye inner corner
                (left_mouth_corner_x, left_mouth_corner_y),  # Left mouth corner
                (right_mouth_corner_x, right_mouth_corner_y)  # Right mouth corner
            ]

            model_points = [
                (0.0, 0.0, 0.0),          # Nose tip
                (0.0, -63.0, -12.0),      # Chin
                (-30.0, 28.0, -25.0),     # Left eye inner corner
                (30.0, 28.0, -25.0),      # Right eye inner corner
                (-28.0, -28.0, -24.0),    # Left mouth corner
                (28.0, -28.0, -24.0)      # Right mouth corner
            ]

            model_points = np.array(model_points, dtype=np.float64)
            image_points = np.array(image_points, dtype=np.float64)

            distortion_coefficients = np.zeros((4, 1))

            focal_length = width

            center = (width / 2, height / 2)

            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                distortion_coefficients
            )

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            angles, *_ = cv2.RQDecomp3x3(rotation_matrix)

            pitch = angles[0]
            roll = angles[1]
            yaw = angles[2]

            print(f"Pitch: {pitch:.2f} | Roll: {roll:.2f} | Yaw: {yaw:.2f}")




        # Display the frame with the drawn landmarks in a window named "TrackIt"
        cv2.imshow("TrackIt", frame)

        # Check if the user pressed the 'q' key, if so, break the loop and exit the program
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources and close windows when the loop is exited
    landmarker.close()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()