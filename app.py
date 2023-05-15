import cv2
import threading
import webbrowser

from pytube import YouTube
from tkinter import *
from voicebot import voicebot
from PIL import ImageTk, Image

BG_TEXT = "#2C3E50"
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"
FONT_TITLE = "Helvetica 14 bold"

# GUI class for the chat
class GUI:
    # constructor method
    def __init__(self):
 
        # chat window which is currently hidden
        self.Window = Tk()
        self.Window.withdraw()
        self.Window.bind('<Escape>', lambda e: self.Window.quit())
        self.init_homepage()

        #Start main loop
        self.Window.mainloop()

    def init_homepage(self):
        self.home = Toplevel()
        self.home.title("Depression Recognition Application")
        self.home.resizable(width=False, height=False)
        self.home.configure(width=450, height=450, bg=BG_COLOR)
        self.title = Label(self.home
                         , bg=BG_COLOR
                         , fg=TEXT_COLOR
                         , font=FONT_TITLE 
                         , text="Depression Recognition Application"
                         , justify=CENTER)
        self.title.place(anchor='center', relx=0.5, rely=0.1)

        self.frame = Frame(self.home, width=300, height=300)
        self.frame.pack()
        self.frame.place(anchor='center', relx=0.5, rely=0.5)
        self.img = ImageTk.PhotoImage(Image.open("logo.png"))
        self.logo = Label(self.frame, image = self.img)
        #self.logo.grid(row=1, column=0, columnspan=2)
        self.logo.pack()

        # create buttons
        self.but1 = Button(self.home
                        ,text="YOUTUBE"
                        ,font=FONT_BOLD
                        ,command=lambda: self.from_home_to_youtube())
        self.but1.place(anchor='center', relx=0.25, rely=0.93)

        self.but2 = Button(self.home
                        , text="INTERVIEW"
                        , font=FONT_BOLD
                        , command=lambda: self.from_home_to_interview())
        self.but2.place(anchor='center', relx=0.72, rely=0.93)

        self.but2 = Button(self.home
                        , text="CAMERA"
                        , font=FONT_BOLD
                        , command=lambda: self.from_home_to_camera())
        self.but2.place(anchor='center', relx=0.48, rely=0.93)

    def deinit_homepage(self):
        self.but2.destroy()
        self.but1.destroy()
        self.frame.destroy()
        self.logo.destroy()
        self.title.destroy()
        self.home.destroy()

    def from_home_to_interview(self):
        self.deinit_homepage()
        self.init_interview_page()

    def start_interview(self):
        # Start the handling thread
        thread = threading.Thread(target=lambda: voicebot(self.chatscreen))
        thread.start()

    def init_interview_page(self):
        # to show chat window
        self.interview = Toplevel()
        self.interview.title("Depression Interview")
        self.interview.resizable(width=False, height=False)
        self.interview.configure(width=470, height=550, bg=BG_COLOR)
        
        
        self.title_interview = Label(self.interview
                                , bg=BG_COLOR
                                , fg=TEXT_COLOR
                                , text="Depression Interview"
                                , font=FONT_BOLD
                                , pady=10, width=20
                                , height=1)
        self.title_interview.grid(row=0, column=0, columnspan=2)
 
        self.chatscreen = Text(self.interview, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=60)
        self.chatscreen.grid(row=1, column=0, columnspan=2)
        
        self.scrollbar = Scrollbar(self.chatscreen)
        self.scrollbar.place(relheight=1, relx=0.974)
    
        self.start_interview_but = Button(self.interview, text="Start", font=FONT_BOLD, bg=BG_GRAY,
                    command=self.start_interview)
        self.start_interview_but.grid(row=2, column=0)

        self.from_interview_to_home_but = Button(self.interview, text="Home", font=FONT_BOLD, bg=BG_GRAY,
                    command=self.from_interview_to_home)
        self.from_interview_to_home_but.grid(row=2, column=1)

    def deinit_interview_page(self):
        self.from_interview_to_home_but.destroy()
        self.start_interview_but.destroy()
        self.scrollbar.destroy()
        self.title_interview.destroy()
        self.interview.destroy()

    def from_interview_to_home(self):
        self.deinit_interview_page()
        self.init_homepage()


    def from_home_to_youtube(self):
        self.deinit_homepage()
        self.init_youtube_page()

    def init_youtube_page(self):
        # to show chat window
        self.link = None
        self.youtube = Toplevel()
        self.youtube.title("Youtube Depression Recognition")
        self.youtube.resizable(width=False, height=False)
        self.youtube.configure(width=470, height=550, bg=BG_COLOR)
        
        self.title_youtube = Label(self.youtube
                                , bg=BG_COLOR
                                , fg=TEXT_COLOR
                                , text="Youtube Depression Recognition"
                                , font=FONT_BOLD
                                , width=30
                                , height=1)
        self.title_youtube.grid(row=0, column=0, columnspan=2)
 
        self.string_var = StringVar()
        self.video_title = Label(self.youtube
                                , bg=BG_COLOR
                                , fg=TEXT_COLOR
                                , text="None"
                                , font=FONT
                                , pady=10
                                , height=1
                                , textvariable=self.string_var)
        self.video_title.grid(row=1, column=0, columnspan=2)
        self.video_title.bind("<Button-1>", lambda e: self.callback(self.link))

        self.e = Entry(self.youtube, bg=BG_TEXT, fg=TEXT_COLOR, font=FONT, width=55)
        self.e.grid(row=2, column=0, columnspan=2)

        self.check_video_but = Button(self.youtube, text="Check", font=FONT_BOLD, bg=BG_GRAY,
                    command=lambda: self.read_link(self.e.get()))
        self.check_video_but.grid(row=3, column=0)
        self.string_var.set("Input your YouTube URL")

        self.from_youtube_to_home_but = Button(self.youtube, text="Home", font=FONT_BOLD, bg=BG_GRAY,
                    command=self.from_youtube_to_home)
        self.from_youtube_to_home_but.grid(row=3, column=1)

    def deinit_youtube_page(self):
        self.from_youtube_to_home_but.destroy()
        self.check_video_but.destroy()
        self.e.destroy()
        self.title_youtube.destroy()
        self.youtube.destroy()

    def from_youtube_to_home(self):
        self.deinit_youtube_page()
        self.init_homepage()

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

    ## Test feature camera
    def from_home_to_camera(self):
        self.deinit_homepage()
        self.init_camera_page()

    def open_camera(self):
        _, frame = self.vid.read()
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)
        self.label_widget.photo_image = photo_image
        self.label_widget.configure(image=photo_image)
        self.label_widget.after(10, self.open_camera)

    def init_camera_page(self):
        
        self.vid = cv2.VideoCapture(0)
        # Declare the width and height in variables
        width, height = 800, 600
        
        # Set the width and height
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # to show chat window
        self.camera = Toplevel()
        self.camera.title("Depression Camera")

        self.label_widget = Label(self.camera)
        self.label_widget.pack()
    
        self.start_camera_but = Button(self.camera, text="Start", font=FONT_BOLD, bg=BG_GRAY,
                    command=self.open_camera)
        self.start_camera_but.pack()

        self.from_camera_to_home_but = Button(self.camera, text="Home", font=FONT_BOLD, bg=BG_GRAY,
                    command=self.from_camera_to_home)
        self.from_camera_to_home_but.pack()

    def deinit_camera_page(self):
        self.from_camera_to_home_but.destroy()
        self.start_camera_but.destroy()
        self.label_widget.destroy()
        self.camera.destroy()

    def from_camera_to_home(self):
        self.deinit_camera_page()
        self.init_homepage()


# create a GUI class object
g = GUI()
