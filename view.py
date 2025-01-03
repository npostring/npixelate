import cv2
import numpy as np
import streamlit as st

import model


# set page config
st.set_page_config(
    page_title='NPixelate',
    page_icon='💠',
    layout="centered",
    initial_sidebar_state="collapsed"
    )


def initialize():
    # initialize session state
    if 'mode' not in st.session_state:
        st.session_state.mode = 'k-means'
    if 'pixel_size' not in st.session_state:
        st.session_state.pixel_size = 4
    if 'color_num' not in st.session_state:
        st.session_state.color_num = 8
    if 'color_thresh' not in st.session_state:
        st.session_state.color_thresh = 10
    if 'is_noise_reduction' not in st.session_state:
        st.session_state.is_noise_reduction = False

    # hidden topbar and adjust width
    st.markdown("""
        <style>
        header.stAppHeader {
            background-color: transparent;
        }
        section.stMain .block-container {
            padding-top: 0rem;
            z-index: 1;
            max-width: 800px;
        }
        </style>
        """, unsafe_allow_html=True)

    # apply font css
    with open('style.css') as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)


def main():
    # sidebar
    with st.sidebar:
        st.header('使い方')
        st.markdown('1. 「Browse files」をクリックし画像をアップロード')
        st.markdown('2. パラメーターを設定')
        st.markdown('3. 「ドット化」ボタンをクリックしドット化')
        st.markdown('4. 「ダウンロード」ボタンをクリックしダウンロード')
        st.header('管理人')
        st.write('npostring')
        st.link_button('Website', 'https://www.dozingwhale.net', use_container_width=True)
        st.link_button('GitHub', 'https://github.com/npostring', use_container_width=True)
        st.link_button('Twitter', 'https://x.com/npostring', use_container_width=True)
        st.write('Ⓒ 2019-2025 [DOZING WHALE](https://www.dozingwhale.net)')

    # title
    st.markdown("<h1 style='text-align: center; padding-top: 0rem'>NPixelate (ドット絵変換ツール)</h1>", unsafe_allow_html=True)

    # upload image
    uploaded_img = st.file_uploader('画像をアップロード(5MBまで)', type=['jpg', 'jpeg', 'png'])
    if uploaded_img:
        st.session_state.uploaded_img = uploaded_img
    else:
        if 'uploaded_img' in st.session_state:
            del st.session_state.uploaded_img
        if 'pixelated_img' in st.session_state:
            del st.session_state.pixelated_img

    # set parameter
    with st.container(border=True, height=350):
        st.write('パラメーター設定')
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox('変換モード', options=['k-means', 'mda', 'sampling', 'ndcd'], key='mode', index=0,
                         help='・k-means：イラストにおすすめ。\n\n・mda：イラストにおすすめ(遅いけどくっきり)。\n\n・sampling：写真におすすめ。\n\n・ndcd：写真におすすめ(遅いけど自然)。')
            st.checkbox('ノイズ除去', key='is_noise_reduction', help='ドット化前にわずかにぼかすことでノイズを除去する。')
        with col2:
            st.slider('ドットサイズ', 2, 32, int(st.session_state.pixel_size), step=1, key='pixel_size')
            if st.session_state.mode == 'sampling':
                pass
            if st.session_state.mode == 'k-means':
                st.slider('色数', 2, 32, int(st.session_state.color_num), step=1, key='color_num')
            if st.session_state.mode == 'mda':
                st.slider('最大色数', 2, 32, int(st.session_state.color_num), step=1, key='color_num', help='最大で設定値まで色数が割り当てられる。')
                st.slider('色閾値', 2, 32, int(st.session_state.color_thresh), step=1, key='color_thresh', help='設定値を下げるほど色数が増える。')
            if st.session_state.mode == 'ndcd':
                pass

    # pixelate button
    pixelate_button = st.button('ドット化', use_container_width=True, disabled='uploaded_img' not in st.session_state)
    if pixelate_button:
        # convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(st.session_state.uploaded_img.read()), dtype=np.uint8)
        cv_img = cv2.imdecode(file_bytes, 1)

        # pixelate
        st.session_state.pixelated_img = model.convert(cv_img,
                                                       mode=st.session_state.mode,
                                                       pixel_size=int(st.session_state.pixel_size),
                                                       color_num=int(st.session_state.color_num),
                                                       color_thresh=int(st.session_state.color_thresh),
                                                       is_noise_reduction=st.session_state.is_noise_reduction)


    image_container = st.container(border=True, height=450)
    with image_container:
        col1, col2 = st.columns(2)
        with col1:
            st.write('ドット化前')
            if 'uploaded_img' in st.session_state:
                st.image(st.session_state.uploaded_img)
            else:
                st.write('画像をアップロードしてください。')
        with col2:
            st.write('ドット化後')
            if 'pixelated_img' in st.session_state:
                st.image(st.session_state.pixelated_img, channels='BGR')
            else:
                st.write('画像をドット化してください。')

    # download button
    buf = '' # dummy buffer
    if 'pixelated_img' in st.session_state:
        _, buf = cv2.imencode('.png', st.session_state.pixelated_img)
        buf = buf.tobytes()
    st.download_button(label='ダウンロード',
                       data=buf,
                       file_name='ドット化後画像.png', 
                       mime='image/png',
                       use_container_width=True,
                       disabled='pixelated_img' not in st.session_state)


if __name__ == '__main__':
    initialize()
    main()
