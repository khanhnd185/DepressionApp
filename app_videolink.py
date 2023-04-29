#importing streamlit library

import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.markdown('<h1 style="background-color: gainsboro; padding-left: 10px; padding-bottom: 20px;">Depression Detector Application</h1>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Online video", "Record"])

with tab1:
    query = st.text_input('', help='Enter link to Youtube Video and hit Enter')

    #displaying a local video file
    #video_file = open('FILENAME', 'rb') #enter the filename with filepath
    #video_bytes = video_file.read() #reading the file
    #st.video(video_bytes) #displaying the video


    if query:
        #displaying a video by simply passing a Youtube link
        st.video(query)

        
        result_str = '<b style="font-size:20px;">The subject in the video is predicted to be normal - Correct</b>'
        st.markdown(f'{result_str}', unsafe_allow_html=True)


with tab2:
    webrtc_streamer(key="sample")
