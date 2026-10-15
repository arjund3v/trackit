# Using OpenCV to capture video from the webcam and draw circles
import cv2 

# Using mediapipe to detect face landmarks, using the new mediapipe tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

            # For each landmark, we convert the normalized coordinates (which are between 0 and 1) to pixel coordinates
            for landmark in landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                # Draw a small green circle at each landmark position on the frame
                # Frame, Position, Radius, Color (BGR), Thickness (-1 means filled circle)
                cv2.circle(frame, (x, y), 1, (230, 216, 173), -1)


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