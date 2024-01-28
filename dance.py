import cv2
import sys
import numpy as np
from mediapipe import solutions
from PyQt5.QtCore import QTimer, pyqtSignal, QSize, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton
import matplotlib.pyplot as plt



# Initialize MediaPipe Pose model
pose = solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

class ImageButton(QLabel):
    clicked = pyqtSignal()

    def __init__(self, image_path, size=QSize(100, 100), parent=None):
        super().__init__(parent)
        pixmap = QPixmap(image_path).scaled(size)  # Scale the pixmap to the desired size
        self.setPixmap(pixmap)
        self.setFixedSize(size)  # Set the size of the label

    def mousePressEvent(self, event):
        self.clicked.emit()


class MainMenuWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Menu")
        self.resize(300, 600)  # Set the initial size of the main menu window

        # Create a layout for the main menu window
        self.layout = QVBoxLayout()  # Use QVBoxLayout for vertical arrangement
        self.central_widget = QLabel()  # Create a central widget
        self.central_widget.setLayout(self.layout)  # Set the layout for the central widget
        self.setCentralWidget(self.central_widget)  # Set the central widget for the main menu window


        # Set background image
        background_image_path = "C:\\Users\\hp\\Downloads\\School\\SheInnovates\\pastel.jpg"
        background_pixmap = QPixmap(background_image_path)
        background_label = QLabel(self)
        background_label.setPixmap(background_pixmap)
        background_label.setGeometry(0, 0, background_pixmap.width(), background_pixmap.height())


        # Create a QIcon with an image file path
        self.banner = ImageButton("C:\\Users\\hp\\Downloads\\School\\SheInnovates\\banner.png", QSize(150, 75))
        self.renegade_button = ImageButton("C:\\Users\\hp\\Downloads\\School\\SheInnovates\\renegade.png", QSize(300, 150))
        self.scenario_button = ImageButton("C:\\Users\\hp\\Downloads\\School\\SheInnovates\\scenario.png", QSize(300, 150))
        self.gangnam_button = ImageButton("C:\\Users\\hp\\Downloads\\School\\SheInnovates\\gangnamstyle.png", QSize(300, 150))

        # Connect button clicks to functions
        self.renegade_button.clicked.connect(self.play_video1)
        self.scenario_button.clicked.connect(self.play_video2)
        self.gangnam_button.clicked.connect(self.play_video3)

        # Layout for the main menu window
        layout = QVBoxLayout()
        layout.addWidget(self.banner, alignment=Qt.AlignCenter)
        layout.addWidget(self.renegade_button, alignment=Qt.AlignCenter)
        layout.addWidget(self.scenario_button, alignment=Qt.AlignCenter)
        layout.addWidget(self.gangnam_button, alignment=Qt.AlignCenter)

        # Set the central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    #set paths
    def play_video1(self):
        video_path = "C:\\Users\\hp\\Downloads\\School\\SheInnovates\\renegade.mp4" 
        self.start_video(video_path)
    def play_video2(self):
        video_path = "C:\\Users\\hp\\Downloads\\School\\SheInnovates\\scenario.mp4" 
        self.start_video(video_path)
    def play_video3(self):
        video_path = "C:\\Users\\hp\\Downloads\\School\\SheInnovates\\gangnamstyle.mp4"  
        self.start_video(video_path)

    def start_video(self, video_path):
        # Initialize VideoWindow with the selected video path
        self.video_window = VideoWindow(video_path)
        self.video_window.show()


class VideoWindow(QMainWindow):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.accuracy_history = []

        # Set background image
        background_image_path = "C:\\Users\\hp\\Downloads\\School\\SheInnovates\\landscape.png"
        background_pixmap = QPixmap(background_image_path)
        background_label = QLabel(self)
        background_label.setPixmap(background_pixmap)
        background_label.setGeometry(0, 0, background_pixmap.width(), background_pixmap.height())


        # Set window title and dimensions
        self.setWindowTitle("Live and Saved Video with Landmarks")
        self.setGeometry(100, 100, 1600, 600)  # Increased width to accommodate the graph

        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()  # Use QHBoxLayout for horizontal arrangement
        self.central_widget.setLayout(self.layout)

        # Create live video label
        self.live_label = QLabel()
        self.live_label.setFixedSize(600, 600)  # Set fixed size for the live video label
        self.layout.addWidget(self.live_label)

        # Create saved video label
        self.saved_label = QLabel()
        self.saved_label.setFixedSize(480, 600)  # Set fixed size for the saved video label
        self.layout.addWidget(self.saved_label)

        # Create a widget for the graph
        self.graph_widget = QWidget()
        self.graph_layout = QVBoxLayout()  # Use QVBoxLayout for vertical arrangement
        self.graph_widget.setLayout(self.graph_layout)
        self.layout.addWidget(self.graph_widget)

        # Create accuracy label
        self.accuracy_label = QLabel("Accuracy: ")  # Initialize accuracy label with default text
        
        # Create buttons
        self.start_button = ImageButton("C:\\Users\\hp\\Downloads\\School\\SheInnovates\\start.png", QSize(150, 75))
        self.restart_button = ImageButton("C:\\Users\\hp\\Downloads\\School\\SheInnovates\\restart.png", QSize(150, 75))


        self.button_layout = QHBoxLayout()

        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.restart_button)
        self.button_layout.addWidget(self.accuracy_label)

        self.graph_layout.addLayout(self.button_layout)

        # Create Matplotlib figure and axes for the graph
        self.figure, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], color='pink', linewidth=2, label='Accuracy Over Time')
        self.ax.set_xlim(0, 60)
        self.ax.set_ylim(0, 100)
        self.ax.set_title('Accuracy Over Time')
        self.ax.get_xaxis().set_visible(False)

        self.ax.set_xlabel('Progress!')
        self.ax.set_ylabel('Accuracy')
        self.graph_layout.addWidget(self.figure.canvas)


        # Connect button clicks to functions
        self.start_button.clicked.connect(self.start_timer)
        self.restart_button.clicked.connect(self.restart_video)

        # Initialize timer for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Initialize video capture objects
        self.live_cap = None
        self.saved_cap = None

        # Flag to track if video playback has ended
        self.video_ended = False

        # Apply stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: transparent;
            }
            QLabel {
                font-family: Vivaldi;
                font-size: 40px; 
            }
        """)



    def start_timer(self):

        self.start_button.setEnabled(False)  # Disable the start button
        self.start_button.hide()  # Hide the start button
        self.timer.singleShot(1000, self.start_video)

    def start_video(self):

        # Show all widgets except the start button
        self.show_widgets()
        self.start_button.setEnabled(False)  # Disable the start button
        self.start_button.hide()  # Hide the start button


        # Release existing video capture objects
        if self.live_cap:
            self.live_cap.release()
        if self.saved_cap:
            self.saved_cap.release()

        # Open the live video capture
        self.live_cap = cv2.VideoCapture(0)

        # Open the saved video file
        self.saved_cap = cv2.VideoCapture(self.video_path)

        # Start updating the frames
        self.timer.start(20)  # fps

    def restart_video(self):

        # Release existing video capture objects
        if self.live_cap:
            self.live_cap.release()
        if self.saved_cap:
            self.saved_cap.release()

        # Clear accuracy history
        self.accuracy_history.clear()

        # Clear previous graph data
        self.line.set_data([], [])  # Clear line data
        self.ax.set_xlim(0, 60)  # Reset x-axis limits
        self.ax.set_ylim(0, 100)  # Reset y-axis limits

        # Restart camera feed
        self.live_cap = cv2.VideoCapture(0)

        # Restart video playback
        self.saved_cap = cv2.VideoCapture(self.video_path)

        # Start updating the frames
        self.timer.start(20)  # fps

    def update_frame(self):

        _, live_frame = self.live_cap.read()
        _, saved_frame = self.saved_cap.read()

        if live_frame is None or saved_frame is None:
            # Skip processing if frames are None
            return

        try:
            # Mirror live frame horizontally
            live_frame = cv2.flip(live_frame, 1)

            # Mirror saved frame horizontally
            saved_frame = cv2.flip(saved_frame, 1)

            # Convert BGR frames to RGB
            live_rgb_frame = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
            saved_rgb_frame = cv2.cvtColor(saved_frame, cv2.COLOR_BGR2RGB)

            # Detect pose landmarks in real-time
            live_results = pose.process(live_rgb_frame)
            saved_results = pose.process(saved_rgb_frame)

            if live_results.pose_landmarks is not None:
                # Draw landmarks on live frame
                solutions.drawing_utils.draw_landmarks(
                    live_rgb_frame,
                    live_results.pose_landmarks,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style()
                )

            if saved_results.pose_landmarks is not None:
                # Draw landmarks on saved frame
                solutions.drawing_utils.draw_landmarks(
                    saved_rgb_frame,
                    saved_results.pose_landmarks,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style()
                )

            # Convert RGB frames to QImage
            live_qimage = QImage(live_rgb_frame.data, live_rgb_frame.shape[1], live_rgb_frame.shape[0],
                                QImage.Format_RGB888)
            saved_qimage = QImage(saved_rgb_frame.data, saved_rgb_frame.shape[1], saved_rgb_frame.shape[0],
                                QImage.Format_RGB888)

            # Convert QImage to QPixmap
            live_qpixmap = QPixmap.fromImage(live_qimage)
            saved_qpixmap = QPixmap.fromImage(saved_qimage)

            # Update labels with the latest images
            self.live_label.setPixmap(live_qpixmap)
            self.saved_label.setPixmap(saved_qpixmap)

            if live_results.pose_landmarks is not None and saved_results.pose_landmarks is not None:
                # Calculate accuracy with a scaling factor of 0.5 (adjust as needed)
                accuracy = calculate_accuracy(live_results.pose_landmarks, saved_results.pose_landmarks, scaling_factor=0.5)
                self.accuracy_history.append(accuracy)

                # Update accuracy label
                self.accuracy_label.setText(f"Accuracy: {accuracy:.2f}%")

                # Update line graph
                self.line.set_data(np.arange(len(self.accuracy_history)), self.accuracy_history)
                self.ax.relim()
                self.ax.autoscale_view()
                self.figure.canvas.draw()

        except cv2.error as e:
            # Handle OpenCV color conversion error
            print(f"OpenCV error: {e}")
            return

    def show_widgets(self):
        for widget in self.central_widget.findChildren(QWidget):
            widget.show()

# Function to calculate accuracy between landmarks
def calculate_accuracy(landmarks1, landmarks2, scaling_factor=1.0):
    similarity_scores = []
    num_landmarks = min(len(landmarks1.landmark), len(landmarks2.landmark))

    for i in range(num_landmarks):
        landmark1 = landmarks1.landmark[i]
        landmark2 = landmarks2.landmark[i]

        # Calculate Euclidean distance between landmarks
        distance = ((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) ** 0.5

        # Scale the distance by the specified factor
        scaled_distance = distance * scaling_factor

        # Add the scaled distance to the list
        similarity_scores.append(scaled_distance)

    # Calculate accuracy as the mean of scaled distances
    accuracy = (100- (100 * (np.mean(similarity_scores))))/1.5
    return accuracy



# Run the PyQt5 application
if __name__ == '__main__':
    app_qt = QApplication(sys.argv)
    main_menu_window = MainMenuWindow()
    main_menu_window.show()
    sys.exit(app_qt.exec_())