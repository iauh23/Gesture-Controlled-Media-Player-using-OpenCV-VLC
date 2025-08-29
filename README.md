A real-time **hand gesture recognition-based media player** built with **OpenCV, NumPy, and VLC Python bindings**.  
This project allows users to control video playback through **simple hand gestures**, creating a touchless and intuitive user experience.  

## Features  
-  **Hand Gesture Detection**  
  - Open Palm → Play Video  
  - Closed Fist → Pause Video  
  - One Finger → Volume Up  
  - Two Fingers → Volume Down  
  - Three Fingers → Forward (10s)  

-  **VLC Integration**  
  - Direct media playback with VLC’s Python API.  
  - Supports play, pause, volume control, forward, and backward navigation.  

-  **Gesture Stability**  
  - Uses gesture buffering (last 5 frames) to reduce false positives.  

-  **Real-Time Performance**  
  - Optimized with morphological operations, Gaussian blurring, contour analysis, and convex hulls.  

---

##  Tech Stack  
- **Python **  
- **OpenCV** (Computer Vision & Gesture Recognition)  
- **NumPy** (Array operations)  
- **python-vlc** (Media playback)  

---

## Limitations

- Works best in good lighting conditions (bright, uniform background).

- **May sometimes misinterpret gestures due to shadows, skin tone variations, or complex backgrounds.

- Gesture recognition is sensitive to camera angle and distance.

- --

## Contributing

- Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
