# Import kivy dependencies first
from kivy.app import App

# Import kivy UX components
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2, os
import tensorflow as tf; import numpy as np
from layers import L1_Dist_Layer

###############################################################################
CWD = os.getcwd()

PATHS = {
    'POS': os.path.join(CWD, 'data', 'positive'),
    'NEG': os.path.join(CWD, 'data', 'negative'),
    'ANC': os.path.join(CWD, 'data', 'anchor'),
    'CHECKPOINT': os.path.join(CWD, 'training_checkpoints', 'ckpt'),
    'MODEL': os.path.join(CWD, 'model'),
    'INFERENCE_DATA': os.path.join(CWD, 'inference_data'),
    'VERIFICATION_IMAGES': os.path.join(CWD, 'inference_data', 'verification_images'),
    'INPUT_IMAGE': os.path.join(CWD, 'inference_data', 'input_image'),
    'APP': os.path.join(CWD, 'app')
}

FILES = {
    'MODEL': os.path.join(PATHS['MODEL'], 'model.h5'),
    'INPUT_IMAGE': os.path.join(PATHS['INPUT_IMAGE'], 'input_image.jpg'),
}

FRAME_WIDTH = 480
FRAME_HEIGHT = 640
FRAME_WIDTH_HALF = FRAME_WIDTH // 2
FRAME_HEIGHT_HALF = FRAME_HEIGHT // 2
BOX_SIZE = 250
BOX_SIZE_HALF = BOX_SIZE // 2

IMG_SIZE_TO_MODEL = (100, 100)
###############################################################################

# Build app and layout 
class CamApp(App):

    def build(self):
        # Main layout components 
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model(FILES['MODEL'], 
                                                custom_objects={'L1_Dist_Layer':L1_Dist_Layer})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    
    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and conver to 100x100px
    def preprocess(file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, IMG_SIZE_TO_MODEL)
        # Scale image to be between 0 and 1 
        img = img / 255.0
        return img

    def verify(self, *args):
        # Specify thresholds
        threshold_detection = 0.5
        threshold_verification = 0.5

        # Capture input image from our webcam
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(FILES['INPUT_IMAGE'], frame)

        # Build results array
        results = []
        for image in os.listdir(PATHS['VERIFICATION_IMAGES']):
            img_input = self.preprocess(FILES['INPUT_IMAGE'])
            img_validation = self.preprocess(os.path.join(PATHS['VERIFICATION_IMAGES'], image))
            # Wrap 02 input img in a list, add 01 more dimension
            result = self.model.predict(list(np.expand_dims([img_input, img_validation], axis=1)))
            results.append(result)
    
        # Threshold Detection: Metric above which a prediction is considered postitive
        num_detections = np.sum(np.array(results) > threshold_detection)
        # Verification Threshold: Proportion of positive predictions/total positive samples
        num_verifications = num_detections/len(os.listdir(PATHS['VERIFICATION_IMAGES']))
        is_verified = num_verifications > threshold_verification

        # Set verification text 
        self.verification_label.text = 'Verified' if is_verified == True else 'Unverified'

        # Log out details
        Logger.info(results)
        Logger.info(num_detections)
        Logger.info(num_verifications)
        Logger.info(is_verified)

        return results, is_verified

if __name__ == '__main__':
    CamApp().run()