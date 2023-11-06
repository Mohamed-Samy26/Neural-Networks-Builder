import customtkinter
import tkinterDnD
from CTkMessagebox import CTkMessagebox

features = [
    (0, "Area"),
    (1, "Perimeter"),
    (2, "MajorAxisLength"),
    (3, "MinorAxisLength"),
    (4, "roundness"),
]

customtkinter.set_ctk_parent_class(tkinterDnD.Tk)

customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme(
    "blue"
)  # Themes: "blue" (standard), "green", "dark-blue"

app = customtkinter.CTk()

app.geometry("1920x1080")
app.title("Bean Classification")


def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def button_callback():
    learning_rate_value = learning_rate.get()
    epochs_value = epochs.get()
    mse_threshold_value = mse_threshold.get()
    first_feature_value = first_feature.get()
    second_feature_value = second_feature.get()
    classes_value = classes.get()

    if not first_feature_value or first_feature_value == "Select First Feature":
        CTkMessagebox(
            title="Error", message="Please select First Feature", icon="cancel"
        )
        return

    if not second_feature_value or second_feature_value == "Select Second Feature":
        CTkMessagebox(
            title="Error", message="Please select Second Feature", icon="cancel"
        )
        return

        # Check if classes is not selected
    if not classes_value or classes_value == "Select Classes":
        CTkMessagebox(title="Error", message="Please select a class", icon="cancel")
        return

    if not learning_rate_value or not is_number(learning_rate_value):
        CTkMessagebox(
            title="Error",
            message="Please enter a valid Learning Rate (numeric value)",
            icon="cancel",
        )
        return

        # Check if epochs is None or empty
    if not epochs_value or not epochs_value.isdigit():
        CTkMessagebox(
            title="Error",
            message="Please enter a valid number of Epochs (positive integer)",
            icon="cancel",
        )
        return

        # Check if mse_threshold is None or empty
    if not mse_threshold_value or not is_number(mse_threshold_value):
        CTkMessagebox(
            title="Error",
            message="Please enter a valid MSE Threshold (numeric value)",
            icon="cancel",
        )
        return

        # Check if any feature is not selected

    print("Learning Rate:", learning_rate_value)
    print("Epochs:", epochs_value)
    print("MSE Threshold:", mse_threshold_value)
    print("First Feature:", first_feature_value)
    print("Second Feature:", second_feature_value)
    print("Class:", classes_value)


features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundness"]

# Initialize variables to track the selected features in each combobox
selected_feature_1 = None
selected_feature_2 = None


def pickFeature1(pickedFeature):
    global features, selected_feature_1, selected_feature_2

    if pickedFeature == selected_feature_1:
        return

    if pickedFeature in features:
        if selected_feature_1 is not None:
            features.append(selected_feature_1)

        features.remove(pickedFeature)
        selected_feature_1 = pickedFeature
        first_feature.configure(values=features)
        second_feature.configure(values=features)

    else:
        print(f"{pickedFeature} not found in the list.")


def pickFeature2(pickedFeature):
    global features, selected_feature_1, selected_feature_2

    if pickedFeature == selected_feature_2:
        return

    if pickedFeature in features:
        if selected_feature_2 is not None:
            features.append(selected_feature_2)
        features.remove(pickedFeature)
        selected_feature_2 = pickedFeature

        first_feature.configure(values=features)
        second_feature.configure(values=features)

    else:
        print(f"{pickedFeature} not found in the list.")


frame_2 = customtkinter.CTkFrame(master=app)
frame_2.pack(pady=20, padx=60, fill="both")

frame_2.columnconfigure(0, weight=1)
frame_2.columnconfigure(1, weight=1)
# frame_2.rowconfigure(0 , weight=1)

frame_1 = customtkinter.CTkFrame(master=app)
frame_1.pack(pady=20, padx=60, fill="both")

# bad anchor "subset": must be n, ne, e, se, s, sw, w, nw, or center
customtkinter.CTkLabel(
    master=frame_2, text="First Feature", font=("Arial Bold", 16)
).grid(column=0, row=0, sticky="w", pady=(20, 5), padx=(400, 0))
first_feature = customtkinter.CTkComboBox(
    frame_2, values=features, command=pickFeature1, width=170
)
first_feature.grid(column=0, row=1, sticky="w", pady=(10, 20), padx=(400, 0))
first_feature.set("Select First Feature")

customtkinter.CTkLabel(
    master=frame_2, text="Second Feature", font=("Arial Bold", 16)
).grid(column=0, row=2, sticky="w", pady=(20, 5), padx=(400, 0))

second_feature = customtkinter.CTkComboBox(
    frame_2, values=features, command=pickFeature2, width=170
)
second_feature.grid(column=0, row=3, sticky="w", pady=(10, 20), padx=(400, 0))
second_feature.set("Select Second Feature")

customtkinter.CTkLabel(master=frame_2, text="Classes", font=("Arial Bold", 16)).grid(
    column=1, row=0, sticky="w", pady=(20, 5), padx=(130, 0)
)

classes = customtkinter.CTkComboBox(
    frame_2, values=["BOMBAY & CALI", "CALI & SIRA", "SIRA & BOMBAY"]
)
classes.grid(column=1, row=1, sticky="w", pady=(10, 20), padx=(130, 0))
classes.set("Select Classes")


customtkinter.CTkLabel(master=frame_2, text="Algorithm", font=("Arial Bold", 20)).grid(
    column=1, row=2, sticky="w", pady=(20, 0), padx=(130, 0)
)

isPerceptron = customtkinter.IntVar(value=0)

radiobutton_1 = customtkinter.CTkRadioButton(
    frame_2, variable=isPerceptron, text="Perceptron", value=0
)
radiobutton_1.grid(column=1, row=3, sticky="w", pady=(20, 20), padx=(130, 0))

radiobutton_2 = customtkinter.CTkRadioButton(
    frame_2, variable=isPerceptron, text="Adaline", value=1
)
radiobutton_2.grid(column=1, row=4, sticky="w", pady=(20, 20), padx=(130, 0))

bias = customtkinter.CTkCheckBox(frame_2, text="Bias")
bias.grid(column=0, row=5, sticky="e", pady=(20, 20), padx=(0, 130))

customtkinter.CTkLabel(
    master=frame_1, text="Learning Rate", font=("Arial Bold", 16), justify="left"
).pack(anchor="w", pady=(20, 7), padx=(25))

learning_rate = customtkinter.CTkEntry(
    master=frame_1,
    placeholder_text="Please enter a float",
    fg_color="#f9fdff",
    font=("Arial Bold", 16),
    border_width=0,
    corner_radius=5,
    height=50,
    text_color="#000000",
)
learning_rate.pack(fill="x", anchor="w", padx=(25))

customtkinter.CTkLabel(
    master=frame_1, text="Number of Epochs", font=("Arial Bold", 16), justify="left"
).pack(anchor="w", pady=(30, 7), padx=(25))

epochs = customtkinter.CTkEntry(
    master=frame_1,
    placeholder_text="Please enter an integer",
    fg_color="#f9fdff",
    font=("Arial Bold", 16),
    border_width=0,
    corner_radius=5,
    height=50,
)
epochs.pack(fill="x", anchor="w", padx=(25))

customtkinter.CTkLabel(
    master=frame_1, text="MSE Threshold", font=("Arial Bold", 16), justify="left"
).pack(anchor="w", pady=(30, 7), padx=(25))

mse_threshold = customtkinter.CTkEntry(
    master=frame_1,
    placeholder_text="Please enter a float",
    fg_color="#f9fdff",
    font=("Arial Bold", 16),
    border_width=0,
    corner_radius=5,
    height=50,
)
mse_threshold.pack(fill="x", anchor="w", padx=(25), pady=(0, 25))


Classify = customtkinter.CTkButton(
    app, command=button_callback, font=("Arial Bold", 20), text="Classify", height=100
)
Classify.pack(fill="x", anchor="w", padx=(60))

app.mainloop()
