# import libraries
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from utils.chatbot_utils import chatbot_response
from utils.translation_utils import detect_lang, translate


def send():
    # get text input by user in chat box
    msg = EntryBox.get("1.0", "end-1c").strip()
    # clear input chat box
    EntryBox.delete("0.0", tk.END)

    if msg != "":
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: " + msg + "\n\n")
        print(f"User: {msg}")
        ChatLog.config(foreground="#FFFF66", font=("Verdana", 12))

        dest_lang = detect_lang(msg)
        msg = translate(msg, "en")
        print(f"     ({msg})")
        print(f"({dest_lang})")
        # get response from chatbot model for given user input
        res = chatbot_response(msg)

        bot_res = translate(res, dest_lang)
        ChatLog.insert(tk.END, "Bot: " + bot_res + "\n\n")
        print(f"Bot: {bot_res}")
        print(f"    ({res})\n")

        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)


# Creating GUI with tkinter
# invoke tk constructor to create top level root window
root = tk.Tk()
root.title("PIZZA PALACE")
# set dimensions of GUI
root.geometry("720x480")
# disable resize of GUI
root.resizable(width=False, height=False)

# load image for background
image = Image.open("./resources/pizza.png")
resized_image = image.resize((720, 480))
background_image = ImageTk.PhotoImage(resized_image)
# set background image
background_label = ttk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

# Create Chat window
ChatLog = tk.Text(root, bd=0, bg="black", height="9", width="0", font="Arial",)
ChatLog.config(state=tk.DISABLED)

# Bind scrollbar to Chat window
scrollbar = ttk.Scrollbar(root, command=ChatLog.yview)
ChatLog["yscrollcommand"] = scrollbar.set

# Create Button to send input
SendButton = tk.Button(
    root,
    font=("Verdana", 20, "bold"),
    text="Send",
    bd=0,
    bg="orange",
    activebackground="olive",
    fg="red",
    command=send,
)

# Create input chat box
EntryBox = tk.Text(
    root, bd=0, bg="white", width="31", height="3", font="Arial"
)

# Place all GUI components on window
scrollbar.place(x=700, y=20, height=360)
ChatLog.place(x=20, y=20, height=360, width=480)
EntryBox.place(x=20, y=400, height=60, width=560)
SendButton.place(x=600, y=400, height=60, width=100)

# run main of root
root.mainloop()
