import customtkinter
import tkinterDnD
from CTkMessagebox import CTkMessagebox
from controllers import slp_controller, adaline_controller


class ClassificationApp:
    def __init__(self):
        customtkinter.set_ctk_parent_class(tkinterDnD.Tk)

        customtkinter.set_appearance_mode(
            "dark"
        )  # Modes: "System" (standard), "Dark", "Light"
        customtkinter.set_default_color_theme(
            "blue"
        )  # Themes: "blue" (standard), "green", "dark-blue"

        self.app = customtkinter.CTk()

        self.app.after(0, lambda: self.app.state("zoomed"))
        self.app.title("Bean Classification")
        self.features = [
            "Area",
            "Perimeter",
            "MajorAxisLength",
            "MinorAxisLength",
            "roundnes",
        ]

        # Initialize variables to track the selected features in each combobox
        self.selected_feature_1 = None
        self.selected_feature_2 = None
        screen_width = self.app.winfo_screenwidth()
        screen_height = self.app.winfo_screenheight()

        self.frame = customtkinter.CTkScrollableFrame(
            master=self.app, width=screen_width, height=screen_height
        )
        self.frame.grid(row=0, column=0, sticky="nsew")

        # create tabview
        self.tabview = customtkinter.CTkTabview(self.frame, corner_radius=10)


        self.tabview.pack(fill="both")
        self.tabview.add("Task 2")
        self.tabview.add("Task 1")
        self.tabview.tab("Task 2").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Task 1").grid_columnconfigure(0, weight=1)
        self.tabview._segmented_button.configure(font= ("Arial", 20, 'bold'))


        # Make the frame expand to fill the window
        self.app.grid_rowconfigure(0, weight=1)
        self.app.grid_columnconfigure(0, weight=1)

        self.frame_2 = customtkinter.CTkFrame(master=self.tabview.tab("Task 1"))
        self.frame_2.pack(pady=40, padx=60, fill="both")

        self.frame_2.columnconfigure(0, weight=1)
        self.frame_2.columnconfigure(1, weight=1)
        # frame_2.rowconfigure(0 , weight=1)

        self.frame_1 = customtkinter.CTkFrame(master=self.tabview.tab("Task 1"))
        self.frame_1.pack(pady=20, padx=60, fill="both")

        # bad anchor "subset": must be n, ne, e, se, s, sw, w, nw, or center
        customtkinter.CTkLabel(
            master=self.frame_2, text="First Feature", font=("Arial Bold", 16)
        ).grid(column=0, row=0, sticky="w", pady=(5, 5), padx=(20, 0))
        self.first_feature = customtkinter.CTkComboBox(
            self.frame_2, values=self.features, command=self.pickFeature1, width=170
        )
        self.first_feature.grid(
            column=0, row=1, sticky="w", pady=(10, 20), padx=(20, 0)
        )
        self.first_feature.set("Select First Feature")

        customtkinter.CTkLabel(
            master=self.frame_2, text="Second Feature", font=("Arial Bold", 16)
        ).grid(column=0, row=2, sticky="w", pady=(5, 5), padx=(20, 0))

        self.second_feature = customtkinter.CTkComboBox(
            self.frame_2, values=self.features, command=self.pickFeature2, width=170
        )
        self.second_feature.grid(
            column=0, row=3, sticky="w", pady=(10, 20), padx=(20, 0)
        )
        self.second_feature.set("Select Second Feature")

        customtkinter.CTkLabel(
            master=self.frame_2, text="Classes", font=("Arial Bold", 16)
        ).grid(column=1, row=0, sticky="w", pady=(5, 5), padx=(15, 0))

        self.classes = customtkinter.CTkComboBox(
            self.frame_2, values=["BOMBAY & CALI", "CALI & SIRA", "SIRA & BOMBAY"]
        )
        self.classes.grid(column=1, row=1, sticky="w", pady=(10, 20), padx=(15, 0))
        self.classes.set("Select Classes")

        customtkinter.CTkLabel(
            master=self.frame_2, text="Algorithm", font=("Arial Bold", 20)
        ).grid(column=1, row=2, sticky="w", pady=(5, 0), padx=(15, 0))

        self.isPerceptron = customtkinter.IntVar(value=0)

        self.radiobutton_1 = customtkinter.CTkRadioButton(
            self.frame_2, variable=self.isPerceptron, text="Perceptron", value=0
        )
        self.radiobutton_1.grid(column=1, row=3, sticky="w", pady=(0, 0), padx=(15, 0))

        self.radiobutton_2 = customtkinter.CTkRadioButton(
            self.frame_2, variable=self.isPerceptron, text="Adaline", value=1
        )
        self.radiobutton_2.grid(column=1, row=4, sticky="w", pady=(0, 0), padx=(15, 0))

        self.bias = customtkinter.CTkCheckBox(self.frame_2, text="Bias")
        self.bias.grid(column=0, row=5, sticky="e", pady=(0, 0), padx=(0, 15))

        customtkinter.CTkLabel(
            master=self.frame_1,
            text="Learning Rate",
            font=("Arial Bold", 16),
            justify="left",
        ).pack(anchor="w", pady=(5, 7), padx=(25))

        self.learning_rate = customtkinter.CTkEntry(
            master=self.frame_1,
            placeholder_text="Please enter a float",
            fg_color="#f9fdff",
            font=("Arial Bold", 16),
            border_width=0,
            corner_radius=5,
            height=50,
            text_color="#000000",
        )
        self.learning_rate.pack(fill="x", anchor="w", padx=(25))

        customtkinter.CTkLabel(
            master=self.frame_1,
            text="Number of Epochs",
            font=("Arial Bold", 16),
            justify="left",
        ).pack(anchor="w", pady=(30, 7), padx=(25))

        self.epochs = customtkinter.CTkEntry(
            master=self.frame_1,
            placeholder_text="Please enter an integer",
            fg_color="#f9fdff",
            font=("Arial Bold", 16),
            border_width=0,
            corner_radius=5,
            height=50,
            text_color="#000000",
        )
        self.epochs.pack(fill="x", anchor="w", padx=(25))

        customtkinter.CTkLabel(
            master=self.frame_1,
            text="MSE Threshold",
            font=("Arial Bold", 16),
            justify="left",
        ).pack(anchor="w", pady=(30, 7), padx=(25))

        self.mse_threshold = customtkinter.CTkEntry(
            master=self.frame_1,
            placeholder_text="Please enter a float",
            fg_color="#f9fdff",
            font=("Arial Bold", 16),
            border_width=0,
            corner_radius=5,
            height=50,
            text_color="#000000",
        )
        self.mse_threshold.pack(fill="x", anchor="w", padx=(25), pady=(0, 25))

        self.Classify = customtkinter.CTkButton(
            self.tabview.tab("Task 1"),
            command=self.button_callback,
            font=("Arial Bold", 20),
            text="Classify",
            height=100,
        )
        self.Classify.pack(fill="x", anchor="w", padx=(60))

        ################################################################################################################

        self.frame_4 = customtkinter.CTkFrame(master=self.tabview.tab("Task 2"))
        self.frame_4.pack(pady=20, padx=60, fill="both")

        self.frame_3 = customtkinter.CTkFrame(master=self.tabview.tab("Task 2"))
        self.frame_3.pack(pady=40, padx=60, fill="both")

        self.frame_3.columnconfigure(0, weight=1)
        self.frame_3.columnconfigure(1, weight=1)

        # bad anchor "subset": must be n, ne, e, se, s, sw, w, nw, or center

        customtkinter.CTkLabel(
            master=self.frame_4,
            text="Number Of Hidden Layers",
            font=("Arial Bold", 16),
            justify="left",
        ).pack(anchor="w", pady=(5, 7), padx=(25))

        self.task2_number_Of_Hidden_Layers = customtkinter.CTkEntry(
            master=self.frame_4,
            placeholder_text="Please enter a integer",
            fg_color="#f9fdff",
            font=("Arial Bold", 16),
            border_width=0,
            corner_radius=5,
            height=50,
            text_color="#000000",
        )
        self.task2_number_Of_Hidden_Layers.pack(fill="x", anchor="w", padx=(25))

        customtkinter.CTkLabel(
            master=self.frame_4,
            text="Number of Neurons in each hidden layer",
            font=("Arial Bold", 16),
            justify="left",
        ).pack(anchor="w", pady=(30, 7), padx=(25))

        self.task2_numberOfNeurons = customtkinter.CTkEntry(
            master=self.frame_4,
            placeholder_text="Please enter an integer",
            fg_color="#f9fdff",
            font=("Arial Bold", 16),
            border_width=0,
            corner_radius=5,
            height=50,
            text_color="#000000",
        )
        self.task2_numberOfNeurons.pack(fill="x", anchor="w", padx=(25))

        customtkinter.CTkLabel(
            master=self.frame_4,
            text="Learning rate",
            font=("Arial Bold", 16),
            justify="left",
        ).pack(anchor="w", pady=(30, 7), padx=(25))

        self.task2_learningRate = customtkinter.CTkEntry(
            master=self.frame_4,
            placeholder_text="Please enter an number",
            fg_color="#f9fdff",
            font=("Arial Bold", 16),
            border_width=0,
            corner_radius=5,
            height=50,
            text_color="#000000",
        )
        self.task2_learningRate.pack(fill="x", anchor="w", padx=(25))

        customtkinter.CTkLabel(
            master=self.frame_4,
            text="Number of epochs",
            font=("Arial Bold", 16),
            justify="left",
        ).pack(anchor="w", pady=(30, 7), padx=(25))

        self.task2_NumberOfEpochs = customtkinter.CTkEntry(
            master=self.frame_4,
            placeholder_text="Please enter an integer",
            fg_color="#f9fdff",
            font=("Arial Bold", 16),
            border_width=0,
            corner_radius=5,
            height=50,
            text_color="#000000",
        )
        self.task2_NumberOfEpochs.pack(fill="x", anchor="w", padx=(25))

        customtkinter.CTkLabel(
            master=self.frame_3, text="Activation Function", font=("Arial Bold", 20)
        ).grid(column=0, row=1, sticky="w", pady=(5, 0), padx=(15, 0))

        self.isSigmoid = customtkinter.IntVar(value=1)

        self.radiobutton_1 = customtkinter.CTkRadioButton(
            self.frame_3, variable=self.isSigmoid, text="Sigmoid", value=1
        )
        self.radiobutton_1.grid(column=0, row=2, sticky="w", pady=(15, 0), padx=(15, 0))

        self.radiobutton_2 = customtkinter.CTkRadioButton(
            self.frame_3, variable=self.isSigmoid, text="Hyperbolic Tangent", value=0
        )
        self.radiobutton_2.grid(column=0, row=3, sticky="w", pady=(15, 0), padx=(15, 0))

        self.task2_bias = customtkinter.CTkCheckBox(self.frame_3, text="Bias")
        self.task2_bias.grid(column=0, row=6, sticky="w", pady=(50, 0), padx=(15, 0))

        self.Classify = customtkinter.CTkButton(
            self.tabview.tab("Task 2"),
            command=self.task2_button_callback,
            font=("Arial Bold", 20),
            text="Classify",
            height=100,
        )
        self.Classify.pack(fill="x", anchor="w", padx=(60))

    def is_number(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def validate_epochs(self, epochs_value):
        try:
            int_epochs = int(epochs_value)
            if int_epochs < 1:
                raise ValueError("Epochs must be greater than 0")
            return True, int_epochs
        except ValueError:
            CTkMessagebox(
                title="Error",
                message="Please enter a valid number of Epochs (positive integer)",
                icon="cancel",
            )
            return False, None

    def validate_mse_threshold(self):
        mse_threshold_value = self.mse_threshold.get()
        try:
            float_mse_threshold = float(mse_threshold_value)
            if float_mse_threshold < 0:
                raise ValueError("MSE Threshold must be greater than or equal 0")
            return True, float_mse_threshold
        except ValueError:
            CTkMessagebox(
                title="Error",
                message="Please enter a valid MSE Threshold (non negative float)",
                icon="cancel",
            )
            return False, None

    def validate_learning_rate(self, learning_rate_value):
        try:
            float_learning_rate = float(learning_rate_value)
            if float_learning_rate <= 0:
                raise ValueError("Learning Rate must be greater than 0")
            return True, float_learning_rate
        except ValueError:
            CTkMessagebox(
                title="Error",
                message="Please enter a valid Learning Rate (positive float)",
                icon="cancel",
            )
            return False, None

    def validate_features(self):
        first_feature_value = self.first_feature.get()
        second_feature_value = self.second_feature.get()
        if not first_feature_value or first_feature_value == "Select First Feature":
            CTkMessagebox(
                title="Error", message="Please select First Feature", icon="cancel"
            )
            return False, None, None

        if not second_feature_value or second_feature_value == "Select Second Feature":
            CTkMessagebox(
                title="Error", message="Please select Second Feature", icon="cancel"
            )
            return False, None, None

        return True, first_feature_value, second_feature_value

    def validate_classes(self):
        classes_value = self.classes.get()
        if not classes_value or classes_value == "Select Classes":
            CTkMessagebox(title="Error", message="Please select a class", icon="cancel")
            return False, None

        return True, classes_value

    def validate_numberOfHiddenLayers(self, numberOfHiddenLayers):
        try:
            int_numberOfHiddenLayers = int(numberOfHiddenLayers)
            if int_numberOfHiddenLayers < 1:
                raise ValueError("Number Of Hidden Layers must be greater than 0")
            return True, int_numberOfHiddenLayers
        except ValueError:
            CTkMessagebox(
                title="Error",
                message="Please enter a valid number of Hidden Layers (positive integer)",
                icon="cancel",
            )
            return False, None

    def validate_numberOfNeurons(self, numberOfNeurons):
        try:
            int_numberOfNeurons = int(numberOfNeurons)
            if int_numberOfNeurons < 1:
                raise ValueError("Number Of Neurons must be greater than 0")
            return True, int_numberOfNeurons
        except ValueError:
            CTkMessagebox(
                title="Error",
                message="Please enter a valid number of Neurons (positive integer)",
                icon="cancel",
            )
            return False, None

    def validate(self, task):

        if task == 1:
            valid_learning_rate, float_learning_rate = self.validate_learning_rate(self.learning_rate.get())
            if not valid_learning_rate:
                return False

            valid_epochs, int_epochs = self.validate_epochs(self.epochs.get())
            if not valid_epochs:
                return False

            valid_mse_threshold, float_mse_threshold = self.validate_mse_threshold()
            if not valid_mse_threshold:
                return False

            (
                valid_features,
                first_feature_value,
                second_feature_value,
            ) = self.validate_features()
            if not valid_features:
                return False

            valid_classes, classes_value = self.validate_classes()
            if not valid_classes:
                return False

            return (
                True,
                float_learning_rate,
                int_epochs,
                float_mse_threshold,
                first_feature_value,
                second_feature_value,
                classes_value,
            )
        else:

            valid_number_of_hidden_layers, int_number_of_hidden_layers = self.validate_numberOfHiddenLayers(
                self.task2_number_Of_Hidden_Layers.get())
            if not valid_number_of_hidden_layers:
                return False

            valid_number_of_neurons, int_number_of_neurons = self.validate_numberOfNeurons(
                self.task2_numberOfNeurons.get())
            if not valid_number_of_neurons:
                return False

            valid_learning_rate, float_learning_rate = self.validate_learning_rate(self.task2_learningRate.get())
            if not valid_learning_rate:
                return False

            valid_epochs, int_epochs = self.validate_epochs(self.task2_NumberOfEpochs.get())
            if not valid_epochs:
                return False

            return (
                True,
                int_number_of_hidden_layers,
                int_number_of_neurons,
                float_learning_rate,
                int_epochs,
            )

    def use_bias(self, bias_value):
        if bias_value == 0:
            return False
        return True

    def button_callback(self):
        (
            validated,
            learning_rate_value,
            int_epochs,
            mse_threshold_value,
            first_feature_value,
            second_feature_value,
            classes_value,
        ) = self.validate(1)
        if not validated:
            return
        print("Learning Rate:", learning_rate_value)
        print("Epochs:", int_epochs)
        print("MSE Threshold:", mse_threshold_value)
        print("First Feature:", first_feature_value)
        print("Second Feature:", second_feature_value)
        print("Class:", classes_value)
        print("Bias:", self.bias.get())
        print("Algorithm:", self.isPerceptron.get())

        if self.isPerceptron.get() == 0:
            slp_controller.infer_slp(
                feature1=first_feature_value,
                feature2=second_feature_value,
                labels=classes_value.split(" & "),
                epochs=int(int_epochs),
                learning_rate=float(learning_rate_value),
                use_bias=self.use_bias(self.bias.get()),
                mse_threshold=float(mse_threshold_value),
            )
        else:
            adaline_controller.infer_adaline(
                feature1=first_feature_value,
                feature2=second_feature_value,
                labels=classes_value.split(" & "),
                epochs=int_epochs,
                learning_rate=float(learning_rate_value),
                use_bias=self.use_bias(self.bias.get()),
                mse_threshold=float(mse_threshold_value),
            )

    def pickFeature1(self, pickedFeature):
        if pickedFeature == self.selected_feature_1:
            return

        if pickedFeature in self.features:
            if self.selected_feature_1 is not None:
                self.features.append(self.selected_feature_1)

            self.features.remove(pickedFeature)
            self.selected_feature_1 = pickedFeature
            self.first_feature.configure(values=self.features)
            self.second_feature.configure(values=self.features)

        else:
            print(f"{pickedFeature} not found in the list.")

    def pickFeature2(self, pickedFeature):
        if pickedFeature == self.selected_feature_2:
            return

        if pickedFeature in self.features:
            if self.selected_feature_2 is not None:
                self.features.append(self.selected_feature_2)
            self.features.remove(pickedFeature)
            self.selected_feature_2 = pickedFeature

            self.first_feature.configure(values=self.features)
            self.second_feature.configure(values=self.features)

        else:
            print(f"{pickedFeature} not found in the list.")

    ##########################################################################

    def task2_button_callback(self):
        (
            validated,
            int_number_of_hidden_layers,
            int_number_of_neurons,
            learning_rate_value,
            int_epochs,
        ) = self.validate(2)

        if not validated:
            return

        print("Number of Hidden Layers:", int_number_of_hidden_layers)
        print("Number Of Neurons:", int_number_of_neurons)
        print("Learning Rate:", learning_rate_value)
        print("Epochs:", int_epochs)
        print("Bias:", self.task2_bias.get())
        print("Activation:", self.isSigmoid.get())

        if self.isSigmoid.get() == 0:
            print("Hyperbolic Tangent")
        else:
            print("Sigmoid")

    def run(self):
        self.app.mainloop()
