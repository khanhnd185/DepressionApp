import threading
import webbrowser

from pytube import YouTube
from tkinter import *
from voicebot import voicebot

BG_TEXT = "#2C3E50"
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

# GUI class for the chat
class GUI:
    # constructor method
    def __init__(self):
 
        # chat window which is currently hidden
        self.Window = Tk()
        self.Window.withdraw()
 
        # login window
        self.login = Toplevel()
        self.login.title("Depression Recognition Application")
        self.login.resizable(width=False, height=False)
        self.login.configure(width=400, height=300)

        self.title = Label(self.login,
                         text="Depression Recognition Application",
                         justify=CENTER,
                         font=FONT)
 
        self.title.place(relheight=0.15, relx=0.2, rely=0.07)

        # create buttons
        self.but1 = Button(self.login,
                         text="YOUTUBE",
                         font=FONT_BOLD,
                         command=lambda: self.youtube())
        self.but1.place(relx=0.3, rely=0.55)

        self.but2 = Button(self.login,
                         text="INTERVIEW",
                         font=FONT_BOLD,
                         command=lambda: self.interview())
        self.but2.place(relx=0.6, rely=0.55)

        #Start main loop
        self.Window.mainloop()

    # Send function
    def start_interview(self):
        # Start the handling thread
        thread = threading.Thread(target=lambda: voicebot(self.chatscreen))
        thread.start()

    def interview(self):
        self.login.destroy()
        self.layout_interview()

    def layout_interview(self):
        # to show chat window
        self.Window.deiconify()
        self.Window.title("Depression Interview")
        self.Window.resizable(width=False,
                              height=False)
        self.Window.configure(width=470,
                              height=550,
                              bg=BG_COLOR)
        
        
        self.title_interview = Label(self.Window
                                , bg=BG_COLOR
                                , fg=TEXT_COLOR
                                , text="Depression Interview"
                                , font=FONT_BOLD
                                , pady=10, width=20
                                , height=1).grid(row=0)
 
        self.chatscreen = Text(self.Window, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=60)
        self.chatscreen.grid(row=1, column=0, columnspan=2)
        
        self.scrollbar = Scrollbar(self.chatscreen)
        self.scrollbar.place(relheight=1, relx=0.974)
        
        self.start_interview_but = Button(self.Window, text="Start", font=FONT_BOLD, bg=BG_GRAY,
                    command=self.start_interview).grid(row=2, column=1)


    def youtube(self):
        self.login.destroy()
        self.layout_youtube()

    def layout_youtube(self):
        # to show chat window
        self.link = None
        self.Window.deiconify()
        self.Window.title("Youtube Depression Recognition")
        self.Window.resizable(width=False,
                              height=False)
        self.Window.configure(width=470,
                              height=550,
                              bg=BG_COLOR)
        
        
        self.title_youtube = Label(self.Window
                                , bg=BG_COLOR
                                , fg=TEXT_COLOR
                                , text="Youtube Depression Recognition"
                                , font=FONT_BOLD
                                , pady=10, width=30
                                , height=1).grid(row=0)
 
        self.string_var = StringVar()
        self.video_title = Label(self.Window
                                , bg=BG_COLOR
                                , fg=TEXT_COLOR
                                , text="None"
                                , font=FONT
                                , pady=10
                                , height=1
                                , textvariable=self.string_var)
        self.video_title.grid(row=1)
        self.video_title.bind("<Button-1>", lambda e: self.callback(self.link))

        self.e = Entry(self.Window, bg=BG_TEXT, fg=TEXT_COLOR, font=FONT, width=55)
        self.e.grid(row=3, column=0)
        
        self.check_video_but = Button(self.Window, text="Check", font=FONT_BOLD, bg=BG_GRAY,
                    command=lambda: self.read_link(self.e.get())).grid(row=3, column=1)
        self.string_var.set("Input your YouTube URL")

    def callback(self, url):
        webbrowser.open_new(url)

    # function to basically start the thread for sending messages
    def read_link(self, link):
        self.link = link
        yt = YouTube(link, use_oauth=True, allow_oauth_cache=True)
        self.string_var.set("Processing...")
        yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(filename='a.mp4')
        self.string_var.set(yt.title)
        self.e.delete(0, END)


 
# create a GUI class object
g = GUI()