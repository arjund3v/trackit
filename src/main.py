import time

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.utils.helper import apply_deadzone, normalize_angle

from output.udp_output import OpenTrackUDP

MODEL_PATH = "models/face_landmarker.task"

# landmark visualization on the webcam feed.
DRAW_DEBUG_LANDMARKS = True

# Allows us to send OpenTrack our telemetry data over UDP protocol
opentrack = OpenTrackUDP()

def main():

    # --------------------------------------------------
    # MediaPipe setup
    # --------------------------------------------------

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,

        # This tells MediaPipe to also return a 4x4 facial
        # transformation matrix for the detected face.
        #
        # We use this matrix for head pose estimation.
        output_facial_transformation_matrixes=True
    )

    landmarker = vision.FaceLandmarker.create_from_options(options)


    # --------------------------------------------------
    # Camera setup
    # --------------------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")


    # --------------------------------------------------
    # Center position
    # --------------------------------------------------
    #
    # These values represent the pose that we consider
    # "looking straight ahead".
    #
    # Pressing C updates these values.

    center_pitch = 0.0
    center_yaw = 0.0
    center_roll = 0.0


    # --------------------------------------------------
    # Smoothed pose values
    # --------------------------------------------------

    smoothed_pitch = 0.0
    smoothed_yaw = 0.0
    smoothed_roll = 0.0

    # EMA smoothing factor.
    #
    # Higher alpha:
    #   faster response
    #   more jitter
    #
    # Lower alpha:
    #   smoother movement
    #   more delay
    alpha = 0.5

    # Deadzone for small movements.
    #
    # This prevents small jittery movements from being sent to the output.
    #
    # Higher deadzone: 
    #   less jitter
    #   more delay
    #
    # Lower deadzone:
    #   more jitter
    #   less delay
    deadzone = 1.0


    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:
            break


        # Keeps track of whether we successfully calculated
        # a valid pose on this frame.
        #
        # We only allow recentering when a pose exists.
        has_pose = False


        # --------------------------------------------------
        # Convert OpenCV image into MediaPipe image
        # --------------------------------------------------
        #
        # OpenCV uses BGR by default.
        # MediaPipe expects RGB.

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # --------------------------------------------------
        # MediaPipe timestamp
        # --------------------------------------------------
        #
        # VIDEO mode requires timestamps that always increase.
        #
        # Instead of pretending every frame takes exactly
        # 16 ms, use the real monotonic clock.

        timestamp_ms = int(
            time.monotonic() * 1000
        )


        # --------------------------------------------------
        # Detect face
        # --------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # --------------------------------------------------
        # Face detected
        # --------------------------------------------------

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            height, width, _ = frame.shape


            # ==================================================
            # DEBUG LANDMARK VISUALIZATION
            # ==================================================
            #
            # These landmarks are NOT used to calculate head
            # rotation anymore.
            #
            # They are only drawn so we can visually confirm
            # that MediaPipe is tracking the face correctly.
            #
            # Earlier in the prototype, these points were used
            # as the 2D image points for solvePnP().
            #
            # That approach has now been replaced by MediaPipe's
            # facial transformation matrix.

            if DRAW_DEBUG_LANDMARKS:

                # ----------------------------------------------
                # Selected facial landmarks
                # ----------------------------------------------

                chin = landmarks[152]
                nose_tip = landmarks[4]

                left_eye_inner_corner = landmarks[154]
                right_eye_inner_corner = landmarks[463]

                left_mouth_corner = landmarks[146]
                right_mouth_corner = landmarks[375]

                left_cheekbone = landmarks[116]
                right_cheekbone = landmarks[345]


                # ----------------------------------------------
                # Convert normalized coordinates to pixels
                # ----------------------------------------------
                #
                # MediaPipe coordinates are normalized:
                #
                # x = 0.0 -> left side
                # x = 1.0 -> right side
                #
                # y = 0.0 -> top
                # y = 1.0 -> bottom

                chin_x = int(chin.x * width)
                chin_y = int(chin.y * height)

                nose_tip_x = int(nose_tip.x * width)
                nose_tip_y = int(nose_tip.y * height)

                left_eye_inner_corner_x = int(
                    left_eye_inner_corner.x * width
                )

                left_eye_inner_corner_y = int(
                    left_eye_inner_corner.y * height
                )

                right_eye_inner_corner_x = int(
                    right_eye_inner_corner.x * width
                )

                right_eye_inner_corner_y = int(
                    right_eye_inner_corner.y * height
                )

                left_mouth_corner_x = int(
                    left_mouth_corner.x * width
                )

                left_mouth_corner_y = int(
                    left_mouth_corner.y * height
                )

                right_mouth_corner_x = int(
                    right_mouth_corner.x * width
                )

                right_mouth_corner_y = int(
                    right_mouth_corner.y * height
                )

                left_cheekbone_x = int(
                    left_cheekbone.x * width
                )

                left_cheekbone_y = int(
                    left_cheekbone.y * height
                )

                right_cheekbone_x = int(
                    right_cheekbone.x * width
                )

                right_cheekbone_y = int(
                    right_cheekbone.y * height
                )


                # ----------------------------------------------
                # Draw debug head lines
                # ----------------------------------------------

                LINE_COLOUR = (0, 255, 0)


                # Chin -> nose
                cv2.line(
                    frame,
                    (chin_x, chin_y),
                    (nose_tip_x, nose_tip_y),
                    LINE_COLOUR,
                    2
                )


                # Nose -> left eye
                cv2.line(
                    frame,
                    (nose_tip_x, nose_tip_y),
                    (
                        left_eye_inner_corner_x,
                        left_eye_inner_corner_y
                    ),
                    LINE_COLOUR,
                    2
                )


                # Nose -> right eye
                cv2.line(
                    frame,
                    (nose_tip_x, nose_tip_y),
                    (
                        right_eye_inner_corner_x,
                        right_eye_inner_corner_y
                    ),
                    LINE_COLOUR,
                    2
                )


                # Nose -> left cheek
                cv2.line(
                    frame,
                    (nose_tip_x, nose_tip_y),
                    (
                        left_cheekbone_x,
                        left_cheekbone_y
                    ),
                    LINE_COLOUR,
                    2
                )


                # Nose -> right cheek
                cv2.line(
                    frame,
                    (nose_tip_x, nose_tip_y),
                    (
                        right_cheekbone_x,
                        right_cheekbone_y
                    ),
                    LINE_COLOUR,
                    2
                )


                # Mouth corner -> mouth corner
                cv2.line(
                    frame,
                    (
                        left_mouth_corner_x,
                        left_mouth_corner_y
                    ),
                    (
                        right_mouth_corner_x,
                        right_mouth_corner_y
                    ),
                    LINE_COLOUR,
                    2
                )


            # ==================================================
            # HEAD POSE ESTIMATION
            # ==================================================
            #
            # Originally, this prototype calculated head pose
            # using OpenCV's solvePnP().
            #
            # The solvePnP approach required us to:
            #
            #   1. Select a small group of 2D facial landmarks
            #      from the webcam image.
            #
            #   2. Create matching points on an approximate
            #      generic 3D face model.
            #
            #   3. Create an estimated camera matrix.
            #
            #   4. Pass everything into solvePnP() to estimate
            #      rotation and translation.
            #
            # That method worked for small movements, but larger
            # head rotations caused unstable pose solutions.
            #
            # In particular, yaw could suddenly jump to another
            # valid pose solution and would sometimes remain
            # incorrect even after returning to center.
            #
            # MediaPipe's Face Landmarker already uses a full
            # canonical 3D face model and can return its own
            # facial transformation matrix.
            #
            # This gives us a much more stable estimate of the
            # face pose, so we use MediaPipe's transform directly
            # instead of rebuilding the pose with solvePnP().
            #
            # MediaPipe returns a 4x4 transformation matrix:
            #
            #     [ R R R Tx ]
            #     [ R R R Ty ]
            #     [ R R R Tz ]
            #     [ 0 0 0  1 ]
            #
            # The upper-left 3x3 part contains the rotation.

            if result.facial_transformation_matrixes:

                transform = np.array(
                    result.facial_transformation_matrixes[0],
                    dtype=np.float64
                )


                # --------------------------------------------------
                # Extract rotation matrix
                # --------------------------------------------------

                rotation_matrix = transform[:3, :3]


                # --------------------------------------------------
                # Rotation matrix -> Euler angles
                # --------------------------------------------------
                #
                # Euler angles give us three intuitive rotations:
                #
                # pitch = look up / down
                # yaw   = turn left / right
                # roll  = tilt head sideways

                angles, *_ = cv2.RQDecomp3x3(
                    rotation_matrix
                )


                # --------------------------------------------------
                # Normalize raw angles
                # --------------------------------------------------

                # Flip pitch so:
                #
                # looking up   = positive
                # looking down = negative

                true_pitch = -normalize_angle(
                    angles[0]
                )

                true_yaw = normalize_angle(
                    angles[1]
                )

                true_roll = normalize_angle(
                    angles[2]
                )


                # --------------------------------------------------
                # Calculate pose relative to center
                # --------------------------------------------------
                #
                # Example:
                #
                # current yaw = 15
                # center yaw  = 10
                #
                # relative yaw = 5

                relative_pitch = normalize_angle(
                    true_pitch - center_pitch
                )

                relative_yaw = normalize_angle(
                    true_yaw - center_yaw
                )

                relative_roll = normalize_angle(
                    true_roll - center_roll
                )


                # --------------------------------------------------
                # EMA smoothing
                # --------------------------------------------------
                #
                # Webcam landmark tracking naturally contains
                # small amounts of noise.
                #
                # Exponential moving average smoothing reduces
                # this jitter while still allowing the tracker
                # to respond quickly.

                smoothed_pitch = (
                    alpha * relative_pitch
                    + (1 - alpha) * smoothed_pitch
                )

                smoothed_yaw = (
                    alpha * relative_yaw
                    + (1 - alpha) * smoothed_yaw
                )

                smoothed_roll = (
                    alpha * relative_roll
                    + (1 - alpha) * smoothed_roll
                )

                output_pitch = apply_deadzone(smoothed_pitch, deadzone)
                output_yaw = apply_deadzone(smoothed_yaw, deadzone)
                output_roll = apply_deadzone(smoothed_roll, deadzone)


                # --------------------------------------------------
                # Current rotation pose
                # --------------------------------------------------
                #
                # Later this will likely expand into 6DOF:
                #
                # x
                # y
                # z
                # pitch
                # yaw
                # roll

                pose = (
                    output_pitch,
                    output_yaw,
                    output_roll
                )


                has_pose = True

                opentrack.send_pose(
                    0.0,
                    0.0,
                    0.0,
                    output_yaw,
                    output_pitch,
                    output_roll
                )


                # --------------------------------------------------
                # Debug output
                # --------------------------------------------------

                print(
                    f"Pitch: {output_pitch:7.2f} | "
                    f"Yaw: {output_yaw:7.2f} | "
                    f"Roll: {output_roll:7.2f}",
                    end="\r"
                )


        # --------------------------------------------------
        # Display webcam
        # --------------------------------------------------

        cv2.imshow(
            "TrackIt",
            frame
        )


        # --------------------------------------------------
        # Keyboard input
        # --------------------------------------------------
        #
        # Only call waitKey once per frame.

        key = cv2.waitKey(1) & 0xFF


        # --------------------------------------------------
        # Center tracking
        # --------------------------------------------------

        if key == ord("c") and has_pose:

            center_pitch = true_pitch
            center_yaw = true_yaw
            center_roll = true_roll


            # Reset EMA history.
            #
            # Without this, the smoothed values would still contain
            # information from before the recenter and would slowly
            # drift toward zero instead of immediately becoming zero.

            smoothed_pitch = 0.0
            smoothed_yaw = 0.0
            smoothed_roll = 0.0


        # --------------------------------------------------
        # Quit
        # --------------------------------------------------

        elif key == ord("q"):
            break


    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    landmarker.close()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()