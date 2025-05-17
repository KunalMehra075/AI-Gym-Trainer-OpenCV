# AI GYM TRAINER OPENCV PROJECT 💪

An AI-powered Virtual Gym Trainer 🏋️ using OpenCV and MediaPipe that counts your exercise repetitions (e.g., bicep curls) by analyzing your form. It provides feedback by counting full reps when the full angle is reached and partial reps (e.g., 0.5) for halfway movements.

## 🖼 Demo

![Demo](assets/Demo.gif)

## How to run:

1.  **Install the required dependencies, globlally or by creating a virtual environment:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the application:**

    ```bash
    python virtualsGymTrainer.py
    ```

## Project Description

This project utilizes computer vision techniques 👀 to track body movements during exercises. It uses OpenCV for video processing 📹 and MediaPipe for pose estimation, allowing it to identify key body points and calculate angles. This enables the application to:

- Count exercise repetitions in real-time ⏱️.
- Differentiate between full and partial repetitions.
- Provide a foundation for more advanced form analysis in the future 🚀.

## Features

- Real-time repetition counting for exercises.
- OpenCV and MediaPipe integration.
- Calculates repetitions based on exercise angle.
- Supports partial repetition counting.

## Dependencies

- OpenCV
- MediaPipe
- Numpy

#### Buy me a coffee 🥹:

<a href="https://www.buymeacoffee.com/kunalmehra" target="_blank">
<img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" >
</a>

Contributions are welcome! 🙏 If you have any ideas for improvements, feel free to submit a pull request.\
Follow me for more exciting projects like this! 🤩
