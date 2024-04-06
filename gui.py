import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
from model import predict_generated

# Default threshold value
DEFAULT_THRESHOLD = 0.6

# Global variable to store threshold value
threshold = DEFAULT_THRESHOLD


# Function to classify text
def classify_text():
    # Get the text from the input box
    text = text_input.get("1.0", "end-1c")

    # Use your function to classify the text
    result = predict_generated(text, threshold)

    # Display the classification result
    result_label.config(text=f"Result: {result}")


# Function to handle threshold change
def on_threshold_change(event):
    # Get the new threshold value
    global threshold
    threshold = round(threshold_slider.get(), 2)

    # Update the threshold label
    threshold_display.config(text=f"Threshold: {threshold}")


# Function to reset threshold to default value
def reset_threshold():
    global threshold
    threshold = DEFAULT_THRESHOLD
    threshold_slider.set(threshold)
    threshold_display.config(text=f"Threshold: {threshold}")


# Function to clear text
def clear_text():
    # Clear the text input box
    text_input.delete("1.0", "end")
    # Clear the result label
    result_label.config(text="")
    # Clear the algorithm label
    algorithm_label.config(text="")


# Create the main window
root = tk.Tk()
root.title("AI/Human Text Classifier")

# Create a frame for the text input
input_frame = ttk.Frame(root)
input_frame.pack(pady=10)

# Create a scrolled text widget for input
text_input = scrolledtext.ScrolledText(input_frame, width=50, height=10)
text_input.pack(side="left")

# Create buttons for classification, clearing, and resetting threshold
classify_button = ttk.Button(input_frame, text="Classify", command=classify_text)
clear_button = ttk.Button(input_frame, text="Clear", command=clear_text)
reset_button = ttk.Button(input_frame, text="Reset Threshold", command=reset_threshold)
classify_button.pack(side="left", padx=5)
clear_button.pack(side="left", padx=5)
reset_button.pack(side="left", padx=5)

# Create a frame for the threshold slider
threshold_frame = ttk.Frame(root)
threshold_frame.pack(pady=5)

# Create labels for AI and Human
ai_label = ttk.Label(threshold_frame, text="AI")
ai_label.pack(side="left", padx=5)

# Create a label and slider for the threshold
threshold_slider = ttk.Scale(threshold_frame, from_=0, to=1, orient="horizontal")
threshold_slider.set(threshold)  # Set default threshold
threshold_slider.pack(side="left", padx=5)

human_label = ttk.Label(threshold_frame, text="Human")
human_label.pack(side="left", padx=5)

# Create a label to display the threshold value
threshold_display = ttk.Label(threshold_frame, text=f"Threshold: {threshold}")
threshold_display.pack(side="left", padx=5)

# Bind the threshold slider to the function handling its change
threshold_slider.bind("<ButtonRelease-1>", on_threshold_change)

# Create a label to display the classification result
result_label = ttk.Label(root, text="")
result_label.pack(pady=10)

# Create a label for the algorithm response
algorithm_label = ttk.Label(root, text="")
algorithm_label.pack(pady=10)

# Run the GUI
root.mainloop()