import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image

class CameraApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        self.image = Image()
        self.layout.add_widget(self.image)
        
        btn_capture = Button(text="Capture", size_hint=(1, 0.1))
        btn_capture.bind(on_press=self.capture)
        self.layout.add_widget(btn_capture)
        
        self.capture_camera()
        
        return self.layout

    def capture_camera(self):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            height, width = frame.shape[:2]
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_frame, 120, 255, cv2.THRESH_BINARY)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # A4 size in pixels (adjust these values based on camera resolution)
            a4_width = int(210 * (width / 1000))  # Convert mm to pixel (approximate)
            a4_height = int(297 * (height / 1000))  # Convert mm to pixel (approximate)

            buffer_space = 20  # Buffer space in pixels
            corners_touched = [False, False, False, False]  # top-left, top-right, bottom-left, bottom-right
            
            for contour in contours:
                # Approximate the contour to a polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Check if the contour is approximately rectangular
                if len(approx) == 4:
                    # Check bounding box dimensions
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    if (w >= a4_width * 0.9 and w <= a4_width * 1.1) and (h >= a4_height * 0.9 and h <= a4_height * 1.1):
                        # Define corner points for the A4 paper with buffer
                        corner_coords = [
                            (x - buffer_space, y - buffer_space),                # Top-left
                            (x + w + buffer_space, y - buffer_space),            # Top-right
                            (x - buffer_space, y + h + buffer_space),            # Bottom-left
                            (x + w + buffer_space, y + h + buffer_space)         # Bottom-right
                        ]

                        for i, (cx, cy) in enumerate(corner_coords):
                            if (0 <= cx < width) and (0 <= cy < height):
                                corners_touched[i] = True

            # Check if all corners are touched
            object_touches_all_corners = all(corners_touched)

            # Set color based on detection
            color = (255, 255, 255) if not object_touches_all_corners else (0, 255, 0)  # White or Green

            # Define corner points for "L" shapes
            line_length = 20  # Length of the lines
            top_left = (50, 50)
            top_right = (width - 50, 50)
            bottom_left = (50, height - 50)
            bottom_right = (width - 50, height - 50)

            # Draw "L" shapes at the corners
            cv2.line(frame, top_left, (top_left[0] + line_length, top_left[1]), color, 2)
            cv2.line(frame, top_left, (top_left[0], top_left[1] + line_length), color, 2)

            cv2.line(frame, top_right, (top_right[0] - line_length, top_right[1]), color, 2)
            cv2.line(frame, top_right, (top_right[0], top_right[1] + line_length), color, 2)

            cv2.line(frame, bottom_left, (bottom_left[0] + line_length, bottom_left[1]), color, 2)
            cv2.line(frame, bottom_left, (bottom_left[0], bottom_left[1] - line_length), color, 2)

            cv2.line(frame, bottom_right, (bottom_right[0] - line_length, bottom_right[1]), color, 2)
            cv2.line(frame, bottom_right, (bottom_right[0], bottom_right[1] - line_length), color, 2)

            # Convert the image to Texture
            buf = cv2.flip(frame, 0)
            buf = buf.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def capture(self, instance):
        ret, frame = self.capture.read()
        if ret:
            cv2.imwrite("captured_image.png", frame)
            print("Image captured and saved as 'captured_image.png'")

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    CameraApp().run()
