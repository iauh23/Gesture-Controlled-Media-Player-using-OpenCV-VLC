import vlc
import time
import cv2 as cv
import numpy as np

# Fix VLC DirectX Issues
vlc_instance = vlc.Instance([
    '--no-video-title-show',
    '--quiet',
    '--no-osd',
    '--disable-screensaver',
    '--vout=directdraw',  # Use DirectDraw instead of D3D11
    '--aout=waveout'
])

media_player = vlc_instance.media_player_new()
media_player.set_media(vlc_instance.media_new("video.mp4"))

def play_video():
    if not media_player.is_playing():
        media_player.play()

def pause_video():
    if media_player.is_playing():
        media_player.pause()


def volume_up():
    volume = media_player.audio_get_volume()
    media_player.audio_set_volume(min(volume+10, 100))

def volume_down():
    volume = media_player.audio_get_volume()
    media_player.audio_set_volume(max(volume-10, 0))

def forward():
    length = media_player.get_length()
    current = media_player.get_time()
    media_player.set_time(min(current + 10000, length))

def backward():
    length = media_player.get_length()
    current = media_player.get_time()
    media_player.set_time(max(current - 10000, 0))

def detect_hand_gesture(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 155, 255], dtype=np.uint8)
    clr_threshold = cv.inRange(hsv, lower_skin, upper_skin)

    # Reduce noise with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    clr_threshold = cv.morphologyEx(clr_threshold, cv.MORPH_OPEN, kernel)
    clr_threshold = cv.morphologyEx(clr_threshold, cv.MORPH_CLOSE, kernel)

    # Apply Gaussian blur to smooth
    clr_threshold = cv.GaussianBlur(clr_threshold, (5, 5), 0)

    contours, _ = cv.findContours(clr_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "none"

    largest_contour = max(contours, key=cv.contourArea)
    area = cv.contourArea(largest_contour)

    # Much higher threshold to prevent false positives
    if area < 10000:
        return "none"

    # Check if contour looks hand-like (aspect ratio test)
    x, y, w, h = cv.boundingRect(largest_contour)
    aspect_ratio = float(w) / h
    if aspect_ratio < 0.4 or aspect_ratio > 2.5:
        return "none"

    try:
        # Calculate solidity (area of contour / area of convex hull) for closed fist detection
        hull = cv.convexHull(largest_contour)
        hull_area = cv.contourArea(hull)

        # If hull_area is zero, it causes a division by zero error, so check
        if hull_area == 0:
            return "none"

        solidity = float(area) / hull_area

        # If solidity is high (> 0.85), it's likely a closed fist
        if solidity > 0.85:
            return "close"

        # For finger counting, use convexity defects
        epsilon = 0.02 * cv.arcLength(largest_contour, True)
        simplified_contour = cv.approxPolyDP(largest_contour, epsilon, True)

        if len(simplified_contour) < 3:
            return "none"

        convex_hull = cv.convexHull(simplified_contour, returnPoints=False)

        if convex_hull is None or len(convex_hull) <= 3:
            return "none"

        defects = cv.convexityDefects(simplified_contour, convex_hull)

        if defects is None:
            # No defects likely means closed fist
            return "close"

        # Count significant defects (fingers)
        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(simplified_contour[s][0])
            end = tuple(simplified_contour[e][0])
            far = tuple(simplified_contour[f][0])

            # Adjust depth threshold for significant defects
            if d > 20000:  # Increase this value for stricter detection
                # Calculate angles to filter out non-finger defects
                a = np.linalg.norm(np.array(far) - np.array(start))
                b = np.linalg.norm(np.array(far) - np.array(end))
                c = np.linalg.norm(np.array(start) - np.array(end))

                # Use cosine rule to find angle
                angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b)) if (2 * a * b) != 0 else 0

                # Filter defects that represent fingers (angle < 90 degrees)
                if angle <= np.pi / 2:
                    finger_count += 1

        # Add 1 to finger count (thumb is not counted in defects)
        finger_count += 1

        # Adjust the logic for counting fingers
        if finger_count < 1:
            return "close"
        elif finger_count == 1:
            return "one"
        elif finger_count == 2:
            return "two"
        elif finger_count == 3:
            return "three"
        elif finger_count >= 4:
            return "open"

    except cv.error:
        return "none"

    return "none"


def gesture_to_action(gesture):
    mapping = {
    "open": "play",
    "close": "pause", 
    "one": "volume_up",
    "two": "volume_down", 
    "three": "forward"
 }

    return mapping.get(gesture)

vid = cv.VideoCapture(0)
last_gesture = "none"
last_action_time = 0
gesture_buffer = []  # Add gesture buffer for stability

while True:
    ret, frame = vid.read()
    if not ret:
        break
    
    gesture = detect_hand_gesture(frame)

    # Add to buffer and keep only last 5 detections
    gesture_buffer.append(gesture)
    if len(gesture_buffer) > 5:
        gesture_buffer.pop(0)

    # Only act if gesture is stable (appears 3+ times in buffer)
    stable_gesture = gesture
    if gesture_buffer.count(gesture) >= 3:
        action = gesture_to_action(stable_gesture)
        current_time = time.time()
        
        print(f"Detected Gesture: {stable_gesture}")  # Debug statement

        if action and stable_gesture != "none" and (action != last_gesture or current_time - last_action_time > 3.0):
            
            if action == "play":
                play_video()  # Call play_video() when the action is "play"
            elif action == "pause":

                pause_video()
            elif action == "volume_up":
                volume_up()
            elif action == "volume_down":
                volume_down()
            elif action == "forward":
                forward()
            elif action == "backward":
                backward()

            last_gesture = action
            last_action_time = current_time

    cv.putText(frame, f"Gesture: {stable_gesture}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("Gesture Media Player", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
media_player.stop()
