import streamlit as st
import requests

st.set_page_config(page_title="🪄 Skills Matcher")

st.title("Skills Matcher")

query = st.text_area(
    label="Describe what you like to do.",
    placeholder="I enjoy creative projects, analyzing data...")


st.markdown(
    """<style>
div[class*="stTextArea"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px;
}
    </style>
    """, unsafe_allow_html=True)

submit_button = st.button("Predict 🔮")

if submit_button:
    # Process the inputs
    url = f'{st.secrets["api_url"]}/predict'
    payload = {
        'type': 'dense',
        'query':query
    }
    r = requests.post(url, params=payload)

    ID_jobs = {
        "Software Engineer".lower() : "SWE",
        "Project Manager".lower() : "PJM",
        "Product Manager".lower() : "PDM",
        "Business Analyst".lower() : "BSA",
        "Customer Success Manager".lower() : "CSM",
        "Data Analyst".lower() : "DAN",
        "Data Scientist".lower() : "DSC",
        "Account Executive".lower() : "ACE",
        "Data Engineer".lower() : "DEN",
        "Sales Representative".lower() : "SRS",
        "Quality Engineer".lower() : "QEN",
        "Solutions Engineer".lower() : "SEN",
        "Marketing Manager".lower() : "MKT",
        "DevOps Engineer".lower() : "DEV",
        "System Engineer".lower() : "SYS",
    }

    job_images = {
        "software engineer": './app/public/swe.gif',
        "project manager": './app/public/pjm.jpg',
        "product manager": './app/public/pdm.png',
        "business analyst": './app/public/bsa.jpg',
        "customer success manager": './app/public/csm.png',
        "data analyst": './app/public/dan.png',
        "data scientist": './app/public/dsc.png',
        "account executive": './app/public/ace.jpg',
        "data engineer": './app/public/den.jpg',
        "sales representative": './app/public/srs.jpg',
        "quality engineer": './app/public/qen.png',
        "solutions engineer": './app/public/sen.png',
        "marketing manager": './app/public/mkt.png',
        "devops engineer": './app/public/dev.jpg',
        "system engineer": './app/public/sys.jpg',
    }

    if r:
        response = r.json()
        prefix = "You would be a phenomenal "

        # Remove the prefix from the string
        job_title = list(response)[0][len(prefix):].lower()
        #job_title = list(response)[0].lower()

        if job_title in job_images:
            st.header(job_title.capitalize())
            st.image(job_images[job_title])
        else:
            st.header("Error 404, No Job Found 🥲")
        # job_id = response['top_match']['id']

        # if "Software engineer" in job_title:
        #     st.header(job_title.capitalize())
        #     st.image('./app/public/swe.gif')

        # elif "Customer success manager" in job_title:
        #     st.header(job_title.capitalize())
        #     st.image('./app/public/pjm.jpg')

        # elif job_id == "PDM":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/pdm.png')

        # elif job_id == "BSA":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/bsa.jpg')

        # elif job_id == "CSM":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/csm.png')

        # elif job_id == "DAN":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/dan.png')

        # elif job_id == "DSC":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/dsc.png')

        # elif job_id == "ACE":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/ace.jpg')

        # elif job_id == "DEN":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/den.jpg')

        # elif job_id == "SRS":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/srs.jpg')

        # elif job_id == "QEN":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/qen.png')

        # elif job_id == "SEN":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/sen.png')

        # elif job_id == "MKT":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/mkt.png')

        # elif job_id == "DEV":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/dev.jpg')

        # elif job_id == "SYS":
        #     st.header(response['top_match']['title'])
        #     st.image('./app/public/sys.jpg')

        # else :
        #     st.header('Error 404, No Job Found 🥲')

    with st.expander("Learn More"):
        st.write(response)

with st.expander("Demo"):
    st.markdown("""
    #### Software Engineer:
    - Proficiency in programming languages (e.g., Java, Python, C++)
    - Software development and coding skills
    - Problem-solving and debugging abilities
    - Knowledge of software development methodologies

    #### Project Manager:
    - Leadership and team management
    - Project planning and organization
    - Communication and interpersonal skills
    - Risk management and problem-solving

    #### Business Analyst:
    - Data analysis and interpretation
    - Requirements gathering and documentation
    - Business process modeling
    - Strong analytical and communication skills

    #### Data Engineer:
    - Manages data storage and ETL (Extract, Transform, Load) processes effectively.
    - Proficient in database administration and SQL query optimization.
    - Designs and develops data pipelines to ensure smooth data flow.
    - Skilled in programming languages such as Python and Java for data-related tasks.

    #### Data Analyst:
    - Proficient in using statistical analysis and data visualization techniques to extract insights from data.
    - Skilled in data preprocessing and cleaning to ensure data accuracy.
    - Familiar with data analysis tools like Python and R for conducting data-related tasks.
    - Possesses strong attention to detail and analytical thinking, enabling effective data interpretation and reporting.

    #### Data Scientist:
    - An expert in applying statistical analysis, machine learning, and deep learning models to uncover valuable insights within data.
    - Proficient in data preprocessing and cleaning to ensure data quality and reliability.
    - Skilled in utilizing programming languages such as Python and R, along with machine learning libraries like Scikit-Learn, TensorFlow, and PyTorch, for advanced data analysis and modeling.
    - Actively engages in data experimentation and model development using tools like Jupyter Notebook to derive actionable insights and make data-driven decisions.

    #### Product Manager:
    - A visionary leader responsible for driving the development and success of a product or product line.
    - Masters the art of market research, customer needs analysis, and product strategy formulation.
    - Collaborates extensively with cross-functional teams to transform ideas into innovative and market-ready products.
    - Employs strategic thinking and strong communication skills to guide the product's entire lifecycle, from inception to launch and beyond.

    #### Customer Success Manager:
    - Expert in building and maintaining client relationships for enhanced loyalty and satisfaction.
    - Skilled in onboarding, training, and supporting clients throughout their lifecycle.
    - Adept at identifying and addressing at-risk accounts and improving service offerings.
    - Strong in analytics, tracking customer engagement, and improving services.
    - Committed to driving product adoption, reducing churn, and identifying upsell opportunities.
""")
