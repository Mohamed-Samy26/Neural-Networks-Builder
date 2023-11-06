import customtkinter
import tkinterDnD
from CTkMessagebox import CTkMessagebox

class ClassificationApp:
    def __init__(self):
        self.features = [
            (0, "Area"),
            (1, "Perimeter"),
            (2, "MajorAxisLength"),
            (3, "MinorAxisLength"),
            (4, "roundness"),
        ]
        customtkinter.set_ctk_parent_class(tkinterDnD.Tk)
        customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
        customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
        self.app = customtkinter.CTk()
        self.app.geometry("1920x1080")
        self.app.title("Bean Classification")
        self.create_widgets()

    def create_widgets(self):
        self.first_feature = customtkinter.CTkComboBox(
            self.app, values=self.features, command=self.pick_feature_1, width=170
        )
        self.first_feature.grid(column=0, row=1, sticky="w", pady=(10, 20), padx=(400, 0))
        self.first_feature.set("Select First Feature")

        self.second_feature = customtkinter.CTkComboBox(
            self.app, values=self.features, command=self.pick_feature_2, width=170
        )
        self.second_feature.grid(column=0, row=3, sticky="w", pady=(10, 20), padx=(400, 0))
        self.second_feature.set("Select Second Feature")

        self.classes = customtkinter.CTkComboBox(
            self.app, values=["BOMBAY & CALI", "CALI & SIRA", "SIRA & BOMBAY"]
        )
        self.classes.grid(column=1, row=1, sticky="w", pady=(10, 20), padx=(130, 0))
        self.classes.set("Select Classes")

        self.learning_rate = customtkinter.CTkEntry(
            master=self.app,
            placeholder_text="Please enter a float",
            fg_color="#f9fdff",
            font=("Arial Bold", 16),
            border_width=0,
            corner_radius=5,
            height=50,
            text_color="#000000",
        )
        self.learning_rate.pack(fill="x", anchor="w", padx=(25))

        self.epochs = customtkinter.CTkEntry(
            master=self.app,
            placeholder_text="Please enter an integer",
            fg_color="#f9fdff",
            font=("Arial Bold", 16),
            border_width=0,
            corner_radius=5,
            height=50,
        )
        self.epochs.pack(fill="x", anchor="w", padx=(25))

        self.mse_threshold = customtkinter.CTkEntry(
            master=self.app,
            placeholder_text="Please enter a float",
            fg_color="#f9fdff",
            font=("Arial Bold", 16),
            border_width=0,
            corner_radius=5,
            height=50,
        )
        self.mse_threshold.pack(fill="x", anchor="w", padx=(25), pady=(0, 25))

        self.classify_button = customtkinter.CTkButton(
            self.app,
            command=self.button_callback,
            font=("Arial Bold", 20),
            text="Classify",
            height=100,
        )
        self.classify_button.pack(fill="x", anchor="w", padx=(60))

    def is_number(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def button_callback(self):
        learning_rate_value = self.learning_rate.get()
        epochs_value = self.epochs.get()
        mse_threshold_value = self.mse_threshold.get()
        first_feature_value = self.first_feature.get()
        second_feature_value = self.second_feature.get()
        classes_value = self.classes.get()

        if not first_feature_value or first_feature_value == "Select First Feature":
            CTkMessagebox(title="Error", message="Please select First Feature", icon="cancel")
            return

        if not second_feature_value or second_feature_value == "Select Second Feature":
            CTkMessagebox(title="Error", message="Please select Second Feature", icon="cancel")
            return

        if not classes_value or classes_value == "Select Classes":
            CTkMessagebox(title="Error", message="Please select a class", icon="cancel")
            return

        if not learning_rate_value or not self.is_number(learning_rate_value):
            CTkMessagebox(
                title="Error",
                message="Please enter a valid Learning Rate (numeric value)",
                icon="cancel",
            )
            return

        if not epochs_value or not epochs_value.isnumeric():
            CTkMessagebox(
                title="Error",
                message="Please enter a valid number of epochs (integer value)",
                icon="cancel",
            )
            return

        if not mse_threshold_value or not self.is_number(mse_threshold_value):
            CTkMessagebox(
                title="Error",
                message="Please enter a valid MSE Threshold (numeric value)",
                icon="cancel",
            )
            return

        # Perform classification logic here
        # Use the provided values for learning rate, epochs, MSE threshold, features, and classes
        
        # Example:
        print("Learning Rate:", learning_rate_value)
        print("Epochs:", epochs_value)
        print("MSE Threshold:", mse_threshold_value)
        print("First Feature:", first_feature_value)
        print("Second Feature:", second_feature_value)
        print("Classes:", classes_value)
        
        
    def run(self):
        self.app.mainloop()