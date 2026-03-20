import random, pickle
import streamlit as st
import tensorflow as tf

from enum import auto

hs = ["အမုန်းစကားကြီး အဲ့လိုမပြောရဘူးနော်", "အမုန်းစကားကြီး စိတ်ဆိုးတယ်", 
        "ဟမ် အမုန်းစကားကြီး", "နောက်ဆို ဒီလိုမပြောရဘူးနော်", "အမုန်းစကားကြီး စိတ်ဆိုးလိုက်တာ"]
nhs = ["အမုန်းစကားဆို မပြောရဘူးနော်", "ချစ်လိုက်တာ", "လိမ္မာလိုက်တာ", 
        "ဒီလိုဘဲပြောရမယ်နော်", "နောက်လည်း ဒီလိုဘဲပြောရမယ်နော်"]


@st.cache_resource
def loadModel():
    # Cache the model assets so Streamlit reruns do not reload them every time.
    # Load without recompiling because this app only performs inference.
    model = tf.keras.models.load_model("modelv3.h5", compile=False)
    with open('tokenizer.pickle', "rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer
    

def pred(text):
    # Preprocess the input exactly like the training pipeline expects.
    X_pred = tokenizer.texts_to_sequences([text])
    X_pred = tf.keras.preprocessing.sequence.pad_sequences(X_pred, padding='post', maxlen=100)
    res = model.predict(X_pred, verbose=0)

    # Use a 0.5 threshold to choose between non-toxic and toxic responses.
    if(res<0.5):
        st.write(nhs[random.randint(0, len(nhs)-1)] + "\t :heart:")
    else:
        st.write(hs[random.randint(0, len(hs)-1)] +"\t :broken_heart:")


greeting = "<h1 style='text-align: center;'>Hello, Mingalar Pr Symbols!!!</h1>"
whower = "<h4 style='text-align: center;'>We are <i>IDK</i> from Símbolo AI course!</h4>"
stment = "<p style='text-align: center;'>This is demo Toxic Detection to detect a word is <b>Toxic or Not</b> in Myanglish.<br><i>Have Fun!</i></p>"
cr = "<p style='text-align: center; color: #ed5d5a'>Made with &hearts; </p>"

model, tokenizer = loadModel()
c1, c2, c3, c4, c5 = st.columns(5)
with c3:
    # Put the logo in the middle column to keep the header visually centered.
    st.image("logo.png",width=150)

st.markdown(greeting, unsafe_allow_html=True)
st.markdown(whower, unsafe_allow_html=True)
st.markdown(stment, unsafe_allow_html=True)

text = st.text_input("Type somethings in Myanglish")
if text:
    pred(text.lower())

st.markdown(cr, unsafe_allow_html=True)
