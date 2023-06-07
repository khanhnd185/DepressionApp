import os
import cv2
import datetime
import threading
import webbrowser
import dlib
import functools
import moviepy.editor as mp

from imutils import face_utils
from pytube import YouTube
from tkinter import *
from tkinter import messagebox
from voicebot import voicebot, detector
from PIL import ImageTk, Image
from tkVideoPlayer import TkinterVideo

BG_TEXT = "#EAECEE"
BG_GRAY = "#E8E2E4"
BG_COLOR = "#E8E2E4"
TEXT_COLOR = "#000000"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"
FONT_TITLE = "Helvetica 24 bold"

# GUI class for the chat
class GUI:
    # constructor method
    def __init__(self):
        
        self.vid = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        # Declare the width and height in variables
        width, height = 600, 450
        
        # Set the width and height
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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
        self.home.configure(width=1280, height=720, bg=BG_COLOR)
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
        self.but1.place(anchor='center', relx=0.35, rely=0.93)

        self.but2 = Button(self.home
                        , text="ABOUT"
                        , font=FONT_BOLD
                        , command=lambda: self.about())
        self.but2.place(anchor='center', relx=0.65, rely=0.93)

        self.but2 = Button(self.home
                        , text="INTERVIEW"
                        , font=FONT_BOLD
                        , command=lambda: self.from_home_to_interview())
        self.but2.place(anchor='center', relx=0.50, rely=0.93)

    def about(self):
        messagebox.showinfo("Information","Developed by Dang-Khanh Nguyen. Email: khanhnd185@gmail.com")

    def deinit_homepage(self):
        self.but2.destroy()
        self.but1.destroy()
        self.frame.destroy()
        self.logo.destroy()
        self.title.destroy()
        self.home.destroy()

    def start_interview(self):
        # Start the handling thread
        thread = threading.Thread(target=lambda: voicebot(self.chatscreen))
        thread.start()


    def from_home_to_youtube(self):
        self.deinit_homepage()
        self.init_youtube_page()


    """"""""""""""""""""""""""
    """ Video player begin """
    """"""""""""""""""""""""""

    def load_video(self):
        file_path = "a.mp4"

        if file_path:
            self.vid_player.load(file_path)

            self.progress_slider.config(to=0, from_=0)
            self.play_pause_btn["text"] = "Play"
            self.progress_value.set(0)


    def seek(self, value):
        self.vid_player.seek(int(value))


    def skip(self, value):
        self.vid_player.seek(int(self.progress_slider.get())+value)
        self.progress_value.set(self.progress_slider.get() + value)


    def play_pause(self):
        if self.vid_player.is_paused():
            self.vid_player.play()
            self.play_pause_btn["text"] = "Pause"

        else:
            self.vid_player.pause()
            self.play_pause_btn["text"] = "Play"


    """"""""""""""""""""""""
    """ Video player end """
    """"""""""""""""""""""""

    def init_youtube_page(self):
        # to show chat window
        self.link = None
        self.youtube = Toplevel()
        self.youtube.title("Youtube Depression Recognition")
        self.youtube.resizable(width=False, height=False)
        self.youtube.configure(width=1280, height=720, bg=BG_COLOR)

        self.load_btn = Button(self.youtube, text="Load", command=lambda: self.load_video())
        self.load_btn.place(anchor='center', relx=0.4, rely=0.70)

        self.vid_player = TkinterVideo(scaled=True, master=self.youtube)
        self.vid_player.place(anchor='center', relx=0.5, rely=0.4, relwidth=0.5, relheight=0.5)

        self.play_pause_btn = Button(self.youtube, text="Play", command=self.play_pause)
        self.play_pause_btn.place(anchor='center', relx=0.6, rely=0.70)

        self.skip_minus_5sec = Button(self.youtube, text="Skip -5 sec", command=lambda: self.skip(-5))
        self.skip_minus_5sec.place(anchor='center', relx=0.3, rely=0.75)

        self.start_time = Label(self.youtube, text=str(datetime.timedelta(seconds=0)))
        self.start_time.place(anchor='center', relx=0.5, rely=0.73)

        self.progress_value = IntVar(self.youtube)

        self.progress_slider = Scale(self.youtube, variable=self.progress_value, from_=0, to=0, orient="horizontal", command=self.seek)
        self.progress_slider.place(anchor='center', relx=0.5, rely=0.75)

        self.end_time = Label(self.youtube, text=str(datetime.timedelta(seconds=0)))
        self.end_time.place(anchor='center', relx=0.5, rely=0.81)

        self.vid_player.bind("<<Duration>>", functools.partial(
            update_duration,
            self=self,
        ))
        self.vid_player.bind("<<SecondChanged>>", functools.partial(
            update_scale,
            self=self,
        ))
        self.vid_player.bind("<<Ended>>", functools.partial(
            video_ended,
            self=self,
        ))

        self.skip_plus_5sec = Button(self.youtube, text="Skip +5 sec", command=lambda: self.skip(5))
        self.skip_plus_5sec.place(anchor='center', relx=0.7, rely=0.75)

        self.title_youtube = Label(self.youtube
                                , bg=BG_COLOR
                                , fg=TEXT_COLOR
                                , text="Youtube Depression Recognition"
                                , font=FONT_TITLE
                                , width=30
                                , height=1)
        self.title_youtube.place(anchor='center', relx=0.5, rely=0.05)
 
        self.string_var = StringVar()
        self.video_title = Label(self.youtube
                                , bg=BG_COLOR
                                , fg=TEXT_COLOR
                                , text="None"
                                , font=FONT
                                , pady=10
                                , height=1
                                , textvariable=self.string_var)
        self.video_title.place(anchor='center', relx=0.5, rely=0.81)
        self.video_title.bind("<Button-1>", lambda e: self.callback(self.link))

        self.e = Entry(self.youtube, bg=BG_TEXT, fg=TEXT_COLOR, font=FONT, width=55)
        self.e.place(anchor='center', relx=0.5, rely=0.86)

        self.check_video_but = Button(self.youtube, text="Check", font=FONT_BOLD, bg=BG_GRAY,
                    command=lambda: self.read_link(self.e.get()))
        self.check_video_but.place(anchor='center', relx=0.47, rely=0.91)
        self.string_var.set("Input your YouTube URL")

        self.from_youtube_to_home_but = Button(self.youtube, text="Home", font=FONT_BOLD, bg=BG_GRAY,
                    command=self.from_youtube_to_home)
        self.from_youtube_to_home_but.place(anchor='center', relx=0.53, rely=0.91)

    def deinit_youtube_page(self):
        self.end_time.destroy()
        self.progress_slider.destroy()
        self.start_time.destroy()
        self.skip_plus_5sec.destroy()
        self.skip_minus_5sec.destroy()
        self.play_pause_btn.destroy()
        self.vid_player.destroy()
        self.load_btn.destroy()
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
        clip = mp.VideoFileClip("a.mp4")
        clip.audio.write_audiofile("a.wav")
        is_depress = detector.inference("a.wav")
        is_depress = "depressed" if is_depress else "normal"
        os.remove("a.wav")
        text = "{} - The subject is diagnosed to be {}".format(yt.title, is_depress)
        self.string_var.set(text)
        self.e.delete(0, END)

    ## Test feature interview
    def from_home_to_interview(self):
        self.deinit_homepage()
        self.init_interview_page()

    def open_camera(self):
        _, frame = self.vid.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)
        self.label_widget.photo_image = photo_image
        self.label_widget.configure(image=photo_image)
        self.label_widget.after(10, self.open_camera)

    def init_interview_page(self):
        # to show chat window
        self.interview = Toplevel()
        self.interview.title("Depression Camera")
        self.interview.resizable(width=False, height=False)
        self.interview.configure(width=1280, height=720, bg=BG_COLOR)

        self.title_interview = Label(self.interview
                                , bg=BG_COLOR
                                , fg=TEXT_COLOR
                                , text="Depression Interview"
                                , font=FONT_TITLE
                                , pady=10, width=20
                                , height=1)
        self.title_interview.place(anchor='center', relx=0.3, rely=0.1)

        self.chatscreen = Text(self.interview, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=40, height=28)
        self.chatscreen.place(anchor='center', relx=0.79, rely=0.5)

        self.scrollbar = Scrollbar(self.chatscreen)
        self.scrollbar.place(relheight=1, relx=0.974)

        self.label_widget = Label(self.interview)
        self.label_widget.pack()
        self.label_widget.place(anchor='center', relx=0.3, rely=0.5)
    
        self.start_interview_but = Button(self.interview, text="Start", font=FONT_BOLD, bg=BG_GRAY,
                    command=self.start_interview)
        self.start_interview_but.pack()
        self.start_interview_but.place(anchor='center', relx=0.47, rely=0.91)

        self.from_interview_to_home_but = Button(self.interview, text="Home", font=FONT_BOLD, bg=BG_GRAY,
                    command=self.from_interview_to_home)
        self.from_interview_to_home_but.pack()
        self.from_interview_to_home_but.place(anchor='center', relx=0.53, rely=0.91)

        self.open_camera()

    def deinit_interview_page(self):
        self.from_interview_to_home_but.destroy()
        self.start_interview_but.destroy()
        self.label_widget.destroy()
        self.title_interview.destroy()
        self.scrollbar.destroy()
        self.chatscreen.destroy()
        self.interview.destroy()

    def from_interview_to_home(self):
        self.deinit_interview_page()
        self.init_homepage()


def update_duration(event, self):
    duration = self.vid_player.video_info()["duration"]
    self.end_time["text"] = str(datetime.timedelta(seconds=duration))
    self.progress_slider["to"] = duration

def update_scale(event, self):
    self.progress_value.set(self.vid_player.current_duration())

def video_ended(event, self):
    self.progress_slider.set(self.progress_slider["to"])
    self.play_pause_btn["text"] = "Play"
    self.progress_slider.set(0)


# create a GUI class object
g = GUI()
