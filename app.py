from enum import auto
from pathlib import WindowsPath
import streamlit as st
import tensorflow as tf
import random
import pickle

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
hs = ["အဲ့လိုမပြောရဘူးနော်", "အမုန်းစကားကြီး", "နောက်ဆို ဒီလိုမပြောရဘူးနော်"]
nhs = ["အမုန်းစကား မပြောရဘူးနော်", "ရေး!!!!", "လိမ္မာလိုက်တာ"]
st.cache(allow_output_mutation=True)
def loadModel():
    
    global model, tokenizer

    model = keras.models.load_model("modelv3.h5")
    with open('tokenizer.pickle', "rb") as file:
        tokenizer = pickle.load(file)

    # st.write("Loaded")


def pred(text):
    #st.write(text)
    X_pred = tokenizer.texts_to_sequences([text])
    X_pred = pad_sequences(X_pred, padding='post', maxlen=100)
    # print(X_pred)
    res = model.predict(X_pred)
    i = random.randint(0, 2)
    if(res<0.5):
        st.write(f"{text} is not a HateSpeech.")
        st.write(nhs[i])
        st.balloons()
    else:
        st.write(f"{text} is a HateSpeech.")
        st.write(hs[i])


h2 = "<h2 style='text-align: center;'>Hello, Mingalar Pr Symbols!!!</h2>"
p = "<p>&nbsp&nbsp&nbsp&nbspWe are IDK from Símbolo AI course!</p>"

loadModel()
c1, c2, c3, c4, c5 = st.columns(5)
with c3:
    st.image("logo.png",width=100, use_column_width=auto)

st.markdown(h2, unsafe_allow_html=True)
st.markdown(p, unsafe_allow_html=True)

st.text("Hatespeech Detection lay ko San kyi pay pr onn")
text = st.text_input("Enter something: ")
if text:
    pred(text)
