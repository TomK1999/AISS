import streamlit as st

st.title("IKEA Assembly Assistant")
st.markdown("---")
st.markdown("## How to use the app")

st.write("Our project aims to assist individuals in assembling IKEA nightstands. We have developed an application that utilizes image recognition technology to detect the required parts for each assembly step. By scanning the assembly area, the application automatically identifies the necessary parts and displays to the user which parts are needed for the current step.")
st.write("If you need help with the assembly process, you can click on the link below to find detailed information on how to assemble the nightstand.")
st.image('Pictures/knarrevik-nightstand-black__AA-2032369-5-100-2.png', width=600)
