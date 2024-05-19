import torch
from PIL import Image
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
from kivy.clock import Clock
import cv2
from matplotlib import pyplot as plt

from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from transformers import ViTForImageClassification, ViTImageProcessor

import phq9_xai
from oculsion_sensitivity_vit import compute_occlusion_sensitivity

model_path = "new_model/"
emotion_model = ViTForImageClassification.from_pretrained(model_path)
# Load the image processor
processor = ViTImageProcessor.from_pretrained(model_path)
# Dictionary mapping emotion labels to human-readable emotions
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
global input_tensor
global IMAGE
user_options = []


# Load the trained model with custom metric
class MainActivity(App):

    def build(self):

        self.questions = [
            "Loss of pleasure",
            "Depressed Moods",
            "Sleep Disturbances",
            "Low energy",
            "Appetite Changes",
            "Feeling of failure",
            "Trouble concentrating",
            "Feeling restlessness",
            "Sucidal ideation",

        ]
        self.emotion_label = Label(text="")
        self.options = [
            ["Not at all", "Several Days", "More than half a day", "Nearly every day"],
            ["Not at all", "Several Days", "More than half a day", "Nearly every day"],
            ["Not at all", "Several Days", "More than half a day", "Nearly every day"],
            ["Not at all", "Several Days", "More than half a day", "Nearly every day"],
            ["Not at all", "Several Days", "More than half a day", "Nearly every day"],
            ["Not at all", "Several Days", "More than half a day", "Nearly every day"],
            ["Not at all", "Several Days", "More than half a day", "Nearly every day"],
            ["Not at all", "Several Days", "More than half a day", "Nearly every day"],
            ["Not at all", "Several Days", "More than half a day", "Nearly every day"],
        ]
        self.ans = []
        self.question_locked = [False, False, False, False, False, False, False, False, False]
        self.current_question_index = 0
        self.total = 0
        self.timer_active = False
        self.capture = cv2.VideoCapture(0)
        self.capture.set(3, 640)
        self.capture.set(4, 480)
        self.layout = BoxLayout(orientation='vertical', spacing=10)
        self.question_label = Label(text=self.questions[self.current_question_index], font_size=20, size_hint_y=None,
                                    height=50)
        self.layout.add_widget(self.question_label)

        self.option_buttons = [ToggleButton(text=option, group='options', on_press=self.on_option_button_click) for
                               option in self.options[self.current_question_index]]
        for button in self.option_buttons:
            self.layout.add_widget(button)

        self.progress_bar = ProgressBar(max=20, value=20, size_hint_y=None, height=20)
        self.layout.add_widget(self.progress_bar)
        self.layout.add_widget(self.emotion_label)
        Clock.schedule_interval(self.update_timer, 1)
        return self.layout

    def pause_timer(self):
        self.timer_active = False

    def resume_timer(self):
        self.timer_active = True
    def capture_frame(self, instance):
        ret, frame = self.capture.read()
        global IMAGE
        IMAGE = frame
        input_frame = cv2.resize(frame, (64, 64))

        # Convert the NumPy array to a PIL image
        input_frame_pil = Image.fromarray(input_frame)

        # Define transformations for preprocessing the image
        image_size = processor.size["height"]
        normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
        transform = Compose([
            Resize((image_size, image_size)),
            ToTensor(),
            normalize
        ])

        # Load and preprocess the image
        image_tensor = transform(input_frame_pil).unsqueeze(0)  # Add batch dimension
        global input_tensor
        input_tensor = image_tensor
        # Perform prediction
        with torch.no_grad():
            outputs = emotion_model(image_tensor)

        # Get predicted probabilities and predicted label
        predicted_probabilities = torch.softmax(outputs.logits, dim=-1)[0].numpy()
        predicted_label_id = predicted_probabilities.argmax()
        predicted_label = emotion_model.config.id2label[predicted_label_id]

        self.emotion_label.text = "Emotion: " + predicted_label

    def on_option_button_click(self, instance):
        if not self.question_locked[self.current_question_index]:
            self.capture_frame(instance)
            option_index = self.options[self.current_question_index].index(instance.text)
            if self.check_contradiction(option_index):
                return  # Don't proceed if contradiction detected
            self.question_locked[self.current_question_index] = True
            user_options.append(option_index)
            self.total += option_index

    def check_contradiction(self, option_index):
        emotion_label = self.emotion_label.text.split(": ")[1]
        if (emotion_label == 'sad' or emotion_label == 'disgust' or emotion_label == 'neutral' or emotion_label == 'angry') and option_index == 0:  # Sad emotion with "Not at all" option
            self.pause_timer()
            self.xai()
            self.show_popup("Contradiction Detected" + emotion_label, "Please attend the question properly.")
            self.resume_timer()
            return True
        elif (emotion_label == 'happy' or emotion_label == "surprise") and (
                option_index == 1 or option_index == 2 or option_index == 3):  # Happy emotion with "Nearly every day" option
            self.pause_timer()
            self.xai()
            self.show_popup("Contradiction Detected" + emotion_label, "Please attend the question properly.")
            self.resume_timer()
            return True
        return False

    def show_popup(self, title, content):
        popup_layout = BoxLayout(orientation='vertical', spacing=10)
        popup_label = Label(text=content, font_size=16)
        popup_layout.add_widget(popup_label)

        ok_button = Button(text='OK', size_hint_y=None, height=40)
        popup_layout.add_widget(ok_button)

        popup = Popup(title=title, content=popup_layout, size_hint=(None, None), size=(300, 200))
        ok_button.bind(on_release=popup.dismiss)

        popup.open()

    def move_to_next_question(self, dt):
        self.current_question_index += 1
        if self.current_question_index < len(self.questions):
            self.update_question()
            self.reset_timer()
            self.enable_option_buttons()
        else:
            # Display the final result
            self.show_result()

    def update_question(self):
        self.question_label.text = self.questions[self.current_question_index]
        self.emotion_label.text = ""
        # Clear previous buttons
        for button in self.option_buttons:
            self.layout.remove_widget(button)

        # Create new option buttons
        self.option_buttons = [ToggleButton(text=option, group='options', on_press=self.on_option_button_click) for
                               option in self.options[self.current_question_index]]
        for button in self.option_buttons:
            self.layout.add_widget(button)

    def update_timer(self, dt):
        if self.progress_bar.value > 0:
            self.progress_bar.value -= 1
        else:
            # Move to the next question after the timer runs out
            self.move_to_next_question(None)

    def reset_timer(self):
        # Set the timer to active only if there are more questions
        self.timer_active = self.current_question_index < len(self.questions) - 1

        # Reset the progress bar for the next question
        self.progress_bar.max = 20
        self.progress_bar.value = 20

        # Schedule the next question only if the timer is active
        if self.timer_active:
            Clock.schedule_once(self.move_to_next_question, 20)


    def reset_timer1(self):
        # Set the timer to active only if there are more questions
        # Reset the progress bar for the next question
        self.progress_bar.max = 20
        self.progress_bar.value = 20
    def enable_option_buttons(self):
        for button in self.option_buttons:
            button.disabled = False

    def xai(self):
        # Perform Occlusion Sensitivity
        occlusion_sensitivity = compute_occlusion_sensitivity(emotion_model, input_tensor)

        # Plot original image and occlusion sensitivity map
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        global IMAGE
        ax[0].imshow(IMAGE)
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        ax[1].imshow(occlusion_sensitivity, cmap='hot', interpolation='nearest')
        ax[1].set_title("Occlusion Sensitivity for emotion " + self.emotion_label.text.split(": ")[1])
        ax[1].axis('off')

        plt.show()

    def show_result(self):
        # Display the final result in a popup
        result_popup = BoxLayout(orientation='vertical', spacing=10)
        msg = ""
        if (self.total <= 4):
            msg = "no depression"
        elif self.total >= 5 and self.total <= 9:
            msg = "mild depression"
        elif self.total >= 10 and self.total <= 14:
            msg = "moderate depression"
        elif self.total >= 15 and self.total <= 19:
            msg = "moderately severe depression"
        else:
            msg = "severe depression"

        result_label = Label(text=f'Total Score: {self.total}' + msg, font_size=20)
        result_popup.add_widget(result_label)

        # close_button = Button(text='Close', size_hint_y=None, height=40, on_press=self.stop)
        # result_popup.add_widget(close_button)

        popup = Popup(title='Result', content=result_popup, size_hint=(None, None), size=(300, 200))
        popup.open()


if __name__ == '__main__':
    MainActivity().run()
    phq9_xai.explainable_ai(user_options)
