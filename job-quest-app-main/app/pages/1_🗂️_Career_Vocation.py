import streamlit as st
import requests

st.set_page_config(page_title="🗂️ Career Vocation")

st.title("What's your calling?")

query = st.text_area(
    label="Describe your interests to discover the field that suits you the best.",
    placeholder="I enjoy creative projects, analyzing data..."
)

st.markdown(
    """<style>
div[class*="stTextArea"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px;
}
    </style>
    """, unsafe_allow_html=True)

submit_button = st.button(label = "Predict 🔮")


if submit_button:
    # Process the inputs
    url = f'{st.secrets["api_url"]}/predict_ml'
    payload = {
        'query':query
    }
    r = requests.post(url, params=payload)
    if r:
        response = r.json()
        if response == "Management":
            st.subheader(response)
            st.image('./app/public/management.jpg')

        elif response == "Engineer":
            st.subheader('Engineering')
            st.image('./app/public/engineering.jpg')

        elif response == "Consultant":
            st.subheader('Consulting')
            st.image('./app/public/consulting.jpg')

        elif response == "Development":
            st.subheader(response)
            st.image('./app/public/development.jpg')

        elif response == "Sales":
            st.subheader(response)
            st.image('./app/public/sales.jpg')

        elif response == "Data":
            st.subheader(response)
            st.image('./app/public/data.jpg')

        elif response == "Product":
            st.subheader(response)
            st.image('./app/public/product.jpg')

    with st.expander("Learn More"):
        st.write(response)



with st.expander("Demo"):
    st.markdown("""
        #### Engineer:
        - Technical experts with a strong background in engineering and technology.
        - Problem solvers who design, build, and maintain various systems.
        - Proficient in using engineering principles to create solutions.

        #### Sales:
        - Sales-oriented professionals with excellent communication skills.
        - Goal-driven individuals who promote and sell products or services.
        - Build relationships with clients and meet sales targets.

        #### Data:
        - Analytical thinkers who work with data to extract insights.
        - Proficient in data analysis tools and techniques.
        - Responsible for gathering, processing, and interpreting data.

        #### Development:
        - Creative individuals involved in software development.
        - Focus on code, design, and innovate to create new products or services.
        - Collaborate with teams to bring ideas to life.
""")
