import streamlit as st
import process_data as pda
import pandas as pd
import pca_chart as pc
import matplotlib.pyplot as plt
import word_similarity
import pickle
import re
from io import StringIO
import pdfplumber
import os
import speech_recognition as sr
import ffmpeg
import pandas as pd
import sys
from moviepy.editor import *
from pydub import AudioSegment
from pydub.silence import split_on_silence
from fuzzywuzzy import fuzz
import webbrowser
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth
import plotly.express as px
import plotly.graph_objects as go
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import time
import io
import random
from tika import parser
import numpy as np
from PIL import Image, ImageDraw
import streamlit.components.v1 as components
from Apps import home,heatmap,upload# import your app modules here

from sklearn.metrics.pairwise import linear_kernel
import sqlite3 
import hashlib


#Authentification
#-------------------------------------------------------------------------------------------------------
selected=""
st.set_page_config(page_title='AlumnAI', page_icon='🧠')
conn = sqlite3.connect('data.db')
c = conn.cursor()


    # Security
    #passlib,hashlib,bcrypt,scrypt
   
def make_hashes(password):
        return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
        if make_hashes(password) == hashed_text:
            return hashed_text
        return False






def create_usertable():
        c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
        c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
        conn.commit()

def login_user(username,password):
        c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
        data = c.fetchall()
        return data
def delete_user(username,password):
        c.execute('DELETE FROM userstable WHERE username =? AND password = ?',(username,password))
        data = c.fetchall()
        return data

def view_all_users():
        c.execute('SELECT * FROM userstable')
        data = c.fetchall()
        return data


menu = ["Login","SignUp"]
choice = st.sidebar.selectbox("Login Or SignUp?",menu)

result=""                

st.markdown("<h1 style='text-align: center;'>Welcome to AlumnAI </h1><br>", unsafe_allow_html=True)
if choice == "Login":
		
        
		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
        
		if st.sidebar.checkbox("Login"):
			# if password == '12345':
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			

    
                        

                
               
if result:
                #-------------------------------------------------------------------------------------------------------
#Variables and Dependecies
    #delete_user("","")
    os.environ["PATH"]+=":/opt/homebrew/Cellar/ffmpeg/5.0.1/bin"
    data =""
    Data=[""]
    user_input=""
    list_jobs=[]
    final_new_list=[" "]
                    #-------------------------------------------------------------------------------------------------------

                    #Introduce App

    if username=="eya":  
                    selected = option_menu(None, ["Partners","Dash"], 
                    icons=['activity','sliders'], 
                    menu_icon="cast", default_index=0, orientation="horizontal")
                    if selected=="Dash":
                        
                        
                        webbrowser.open_new_tab("https://app.powerbi.com/reportEmbed?reportId=8ee8fe4a-8635-4741-b4ff-7a570e3879ee&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLW5vcnRoLWV1cm9wZS1pLXByaW1hcnktcmVkaXJlY3QuYW5hbHlzaXMud2luZG93cy5uZXQvIn0%3D")
                        
                    if selected=="Partners":
                    #-------------------------------------------------------------------------------------------------------
                            #PARTNERS RECOMMANDATION
                        option = st.selectbox(
                                    'Which Specialization would you like to compare to to see Esprit Potential Partners?',
                                ('Choose a Specialization','Information Technology and Services','Software','Computer networks','Data Science','civil engineering','electromechanical engineering'))
                        def user_input() :
                            cossim=st.slider('similarity greater than : ',0.0000000000000002,1.0000000000000002,0.5000000000000002)
                            data={'cossim':cossim
                            }
                            param=pd.DataFrame(data,index=[0])
                            return param

                        input_partner=user_input()



                        df_partners = pd.read_csv("Company2.csv",header=None)
                        df_partners=df_partners.rename(columns=df_partners.iloc[0])
                        df_partners.drop(df_partners.index[0],inplace=True)
                        df_partners.reset_index(drop=True, inplace=True)
                    
                        def cosine_sim(text1, text2):
                        #Calculate the similarity btween 2 texts
                            vectorizer = TfidfVectorizer(
                                ngram_range=(1,1))
                            tfidf = vectorizer.fit_transform([text1, text2])
                            return ((tfidf * tfidf.T).A)[0,1]



                        for i in range(len(df_partners)):
                        #print(cosine_sim(df['description'][i],j))
                         if( cosine_sim(df_partners['description'][i],option)  >= input_partner['cossim'][0]):
                            #image = Image.open(df['logoUrl'][i])
                            if st.image(df_partners['logoUrl'][i]) and st.button(df_partners['companyName'][i]) :
                                webbrowser.open_new_tab(df_partners['companyUrl'][i])
                    
                    
    else:        

                 
                    
                    col1, col2, col3 = st.columns(3)

                    # with col1:
                    #     st.write(' ')

                    # with col2:
                    #    st.image('logo_alumnAI.png', width=200)

                    # with col3:
                    #     st.write(' ')
                    selected = option_menu(None, ["Jobs","Profiling",'Company','Dash'], 
                        icons=['kanban', "list-task",'sliders'], 
                        menu_icon="cast", default_index=0, orientation="horizontal")
                   
                      
                    if selected=="Dash":
                        
                        
                        webbrowser.open_new_tab("https://app.powerbi.com/reportEmbed?reportId=cfe7590a-4e7c-4825-a374-c3b3a9d3a24a&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLW5vcnRoLWV1cm9wZS1pLXByaW1hcnktcmVkaXJlY3QuYW5hbHlzaXMud2luZG93cy5uZXQvIn0%3D")    
                        
                        
                    # with st.sidebar:
                            
                    #         st.subheader('Help Us Improve our Application')
                            # col1, col2, col3 = st.columns(3)

                            # with col1:
                            #       st.write(' ')

                            # with col2:
                            #       st.image('logo_alumnAI.png', width=100)

                            # with col3:
                            #       st.write(' ')

                            # with st.form("my_form"):
                            #     add_type = st.selectbox(
                            #     "Are You Currently ?",
                            #     ("A Student", "Alumni", "Esprit Administration"))
                            #     add_age = st.number_input('How old Are you?',0, 85)
                            #     add_experience=st.select_slider('How much Are you satisfied with your work experience ?', options=['low','medium','high'])   
                            #     submitted =st.form_submit_button("Submit")
                            
                            # st.markdown('#')
                            # st.markdown('#')
                            # col1, col2, col3 = st.columns(3)

                            # with col1:
                            #       st.write(' ')

                            # with col2:
                            #        authenticator.logout('Logout', 'main')

                            # with col3:
                            #       st.write(' ')
                                
                        
                        
                   
                        
                        
                

                        

                    if selected=="Profiling":
                            
                    


                            #-------------------------------------------------------------------------------------------------------

                            #PDF CV
                            def extract_data(feed):
                                
                                with pdfplumber.open(feed) as pdf:
                                    pages = pdf.pages
                                    for p in pages:
                                        Data.append(p.extract_tables())
                                return Data
                            user_input=""
                            uploaded_file_pdf = st.file_uploader("Please put your CV (PDF)", "pdf")
                            path = "Exemple_Resumes"
                            os.chmod(path, 0o777)
                            for f in os.listdir(path):
                                os.remove(os.path.join(path, f)) 
                            if uploaded_file_pdf is not None:
                                Data = extract_data(uploaded_file_pdf)
                                user_input = ' '.join(map(str, Data))
                                with open(os.path.join("Exemple_Resumes",uploaded_file_pdf.name),"wb") as f:
                                    f.write(uploaded_file_pdf.getbuffer())




                            #-------------------------------------------------------------------------------------------------------

                            #CV DIGITAL
                            r = sr.Recognizer()

                            def get_large_audio_transcription(path):
                                """
                                Splitting the large audio file into chunks
                                and apply speech recognition on each of these chunks
                                """
                            
                                sound = AudioSegment.from_wav(path)  
                            
                                chunks = split_on_silence(sound,
                                    
                                    min_silence_len = 500,
                                    
                                    silence_thresh = sound.dBFS-14,
                                    
                                    keep_silence=500,
                                )
                                folder_name = "audio-chunks"
                            
                                if not os.path.isdir(folder_name):
                                    os.mkdir(folder_name)
                                whole_text = ""
                                
                                for i, audio_chunk in enumerate(chunks, start=1):
                                    chunk_filename = os.path.join(folder_name, f"speech{i}.wav")
                                    audio_chunk.export(chunk_filename, format="wav")
                                    with sr.AudioFile(chunk_filename) as source:
                                        audio_listened = r.record(source)
                                        try:
                                            text = r.recognize_google(audio_listened)
                                        except sr.UnknownValueError as e:
                                            print("Error:", str(e))
                                        else:
                                            text = f"{text.capitalize()}. "
                                            print(chunk_filename, ":", text)
                                            whole_text += text
                                return whole_text
                            uploaded_file = st.file_uploader("Please put your Video CV (Digital CV)", type=["mp4", "mpeg"])

                            if uploaded_file is not None:
                                video = VideoFileClip(os.path.join("digitalCV/cvexemple.mp4"))
                                video.audio.write_audiofile(os.path.join("digitalCV/speech.mp3"))
                                sound = AudioSegment.from_mp3("digitalCV/speech.mp3")
                                sound.export("digitalCV/speech.wav", format="wav")
                                result=get_large_audio_transcription("digitalCV/speech.wav")
                                with open('FinalRecognized.txt',mode ='w') as file: 
                                    file.write(result) 
                                    print("Transcriped")
                                with open('FinalRecognized.txt', 'r') as file:
                                    data = file.read().rstrip()
                            
                                user_input = re.sub('[^a-zA-Z0-9\.]', ' ', data)
                                user_input = data.lower()
                            



                            #-------------------------------------------------------------------------------------------------------

                            #STUDY WORK CONTACT
                            st.subheader('Find your Study-Work Contact')

                            def get_ratio(row1,row2):
                                name = row1
                                name1 = row2
                                return fuzz.token_set_ratio(name, name1)

                            df_Match=pd.read_csv('Profile_Meriem')
                            for i in df_Match['headlinee']:    
                                if(get_ratio(i,user_input) > 30):
                                    index=df_Match.index[df_Match['headlinee']==i]
                                    list_jobs.append(df_Match['fullName'][index].values[0])
                                    final_new_list=list(set(list_jobs))
                                    final_new_list.insert(0,'Choose your Study-Work Contact')
                            candidate =st.selectbox("Select A Study-Work Contact",options=[opt for opt in final_new_list])
                            for i in df_Match['fullName']:    
                                if (i==candidate):
                                    index=df_Match.index[df_Match['fullName']==i]
                                    webbrowser.open_new_tab(df_Match["linkedinProfile"][index].values[0])  

                            #-------------------------------------------------------------------------------------------------------
                            #JOB PROFILLING PERCENTAGE BAR PLOT

                            user_input = pd.Series(user_input)
                            st.subheader('Find your Job Profiling')
                            topic_model = pickle.load(open('topic_model.sav', 'rb'))
                            classifier = pickle.load(open('classification_model.sav', 'rb'))
                            vec = pickle.load(open('job_vec.sav', 'rb'))

                            classes, prob = pda.main(user_input, topic_model, classifier, vec)

                            data = pd.DataFrame(zip(classes.T, prob.T), columns = ['jobs', 'probability'])
                            def plot_user_probability():
                                #plt.figure(figsize = (2.5,2.5))
                            
                                plt.barh(data['jobs'], data['probability'], color = 'r')
                                plt.title('Percent Match of Job Type')
                                st.pyplot()


                            #-------------------------------------------------------------------------------------------------------
                            #JOB PROFILLING DOT/CLUSTER
                            def plot_clusters():
                                st.markdown('This chart uses PCA to show you where you fit among the different job archetypes.')
                                X_train, pca_train, y_train, y_vals, pca_model = pc.create_clusters()
                                for i, val in enumerate(y_train.unique()):
                                    y_train = y_train.apply(lambda x: i if x == val else x)
                                example = user_input
                                doc = pc.transform_user_resume(pca_model, example)

                                pc.plot_PCA_2D(pca_train, y_train, y_vals, doc)
                                st.pyplot()

                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            plot_user_probability()
                            st.subheader('Representation Among Job Types')
                            plot_clusters()
                            #-------------------------------------------------------------------------------------------------------
                            #KEYWORDS CV

                            st.subheader('Find your the Keywords for your CV')

                            option = st.selectbox(
                                'Which job would you like to compare to?',
                            ('ux,designer', 'data,analyst', 'project,manager', 'product,manager', 'account,manager', 'consultant', 'marketing', 'sales',
                            'data,scientist'))

                            st.write('You selected:', option)
                            matches, misses = word_similarity.resume_reader(user_input, option)
                            match_string = ' '.join(matches)
                            misses_string = ' '.join(misses)

                            st.markdown('Matching Words:')
                            st.markdown(match_string)
                            st.markdown('Missing Words:')
                            st.markdown(misses_string)
                            
                
                #-------------------------------------------------------------------------------------------------------
                            #JOBS RECOMMANDATION            
                    if selected=="Jobs":    
                        df1 = pd.read_csv('jobss.csv')
                        df=df1.head(1000)
                        df=df.dropna()
                        df=df.fillna("nan")
                        def extract_data(feed):
                                    
                                    with pdfplumber.open(feed) as pdf:
                                        pages = pdf.pages
                                        for p in pages:
                                            Data.append(p.extract_tables())
                                    return Data
                        user_input=""
                        uploaded_file_pdf = st.file_uploader("Please put your CV (PDF)", "pdf")
                        path = "Exemple_Resumes"
                        os.chmod(path, 0o777)
                        for f in os.listdir(path):
                            os.remove(os.path.join(path, f)) 
                        if uploaded_file_pdf is not None:
                            Data = extract_data(uploaded_file_pdf)
                            user_input = ' '.join(map(str, Data))
                            with open(os.path.join("Exemple_Resumes",uploaded_file_pdf.name),"wb") as f:
                              f.write(uploaded_file_pdf.getbuffer())

                        #Elyes PDF
                        df2 = pd.DataFrame()
                        df2.insert(0,"Description",False)
                        df2.insert(1, "Skills", False)
                        df2.insert(2, "Education", False)
                        df2.insert(3, "Activities", False)
                        profiles_voc = ['PERSONALPROFILE','PROFILE' , 'SUMMARY' , 'EXECUTIVEPROFILE']
                        education_vocab = ['EDUCATION', 'ACADEMICPROFILE']
                        activities_vocab = ['ACTIVITIESANDAWARDS','OTHERACTIVITIES','ACTIVITIES','OTHERQUALIFICATION', 'PROJECTS']
                        skills_vocab = ['EXPERIENCE', 'SKILLS','WORKBACKGROUND', 'WORKEXPERIENCE']
                        directory = "Exemple_Resumes"
                        index=0
                        for filename in os.listdir(directory):
                            index+=1
                            f = os.path.join(directory,filename)
                            row={
                            'description':' ',
                            'skills':' ',
                            'education':' ',
                            'activities':' '}

                        # checking if it is a file
                            if os.path.isfile(f):
                                raw = parser.from_file(f)
                                lines= raw['content'].splitlines()
                                while("" in lines) :
                                    lines.remove("")
                                name_candidates= []
                                i=0
                                found=False
                                start=False
                                feature='name'
                                found=False
                                skip=False
                                for line in lines:
                                    for word in profiles_voc:
                                        if(line.upper().replace(" ","").count(word)!=0):
                                            start=True
                                            feature='description'
                                            found=True
                                            skip=True
                                    if(True):
                                        for word in education_vocab:
                                            if(line.upper().replace(" ","").count(word)!=0):
                                                start=True
                                                feature='education'
                                                found=True
                                                skip=True
                                    if(True):
                                        for word in activities_vocab:
                                            if(line.upper().replace(" ","").count(word)!=0):
                                                start=True
                                                feature='activities'
                                                found=True
                                                skip=True
                                    if(True):
                                        for word in skills_vocab:
                                            if(line.upper().replace(" ","").count(word)!=0):
                                                start=True
                                                feature='skills'
                                                found=True  
                                                skip=True
                                    if(start):
                                        if(skip):
                                            skip=False
                                        else:
                                            row[feature] = row[feature] + ' ' + line
                                df2.loc[index] = [row['description'],row['skills'],row['education'],row['activities']]
                            
                        

                                
                        def mapp(df):

                            data = df[["Longitude", "Latitude"]]
                            apps = [
                            {"func": home.app(data), "title": "Home", "icon": "house"},
                            {"func": heatmap.app, "title": "Heatmap", "icon": "map"},
                            {"func": upload.app, "title": "Upload", "icon": "cloud-upload"},
                            ]

                            titles = [app["title"] for app in apps]
                            titles_lower = [title.lower() for title in titles]
                            icons = [app["icon"] for app in apps]

                            params = st.experimental_get_query_params()

                            if "page" in params:
                                default_index = int(titles_lower.index(params["page"][0].lower()))
                            else:
                                default_index = 0








                        def cosineSimilarity(name1,name2):
                            cv= CountVectorizer()
                            count_matrix = cv.fit_transform([name1,name2])
                            matchPercentage = cosine_similarity(count_matrix)[0][1]*100
                            matchPercentage = round(matchPercentage,2)
                            return matchPercentage

                        skills  = "manual testing| test engineering| test cases"
                        Job_Experience = "5 yrs"
                        arr=[]





                        if uploaded_file_pdf:
                            nb=0
                            df3=pd.DataFrame()
                            st.subheader("Your profile")
                            st.write(df2.iloc[0])
                            st.subheader("Your potentiels offers")
                            
                            for i in df['Key Skills']:
                                a = cosineSimilarity(i,df2['Skills'].iloc[0])
                                #indexx1=df[df['Key Skills']==i].index[0]
                                #b =cosineSimilarity(Job_Experience,df["Job Experience Required"].iloc[indexx1]) 
                                if (a > 10 and nb!=5):
                                    indexx1=df[df['Key Skills']==i].index[0]
                                    df3 = df3.append(df.iloc[indexx1])
                                    st.subheader(df["Job Title"].iloc[indexx1])     
                                    st.write("Role needed : "+str(df["Role"].iloc[indexx1]))
                                    st.write("Experience required : "+df["Job Experience Required"].iloc[indexx1])
                                    st.write("Key skills : "+df["Key Skills"].iloc[indexx1])
                                    st.write("Avarage salary : "+str(df["sal"].iloc[indexx1]))
                                    st.write("Location  : "+str(df["Location"].iloc[indexx1]))
                                    nb+=1
                                    components.html("""<hr style="height:1px;border:none;color:#FF4B4B;background-color:#FF4B4B;" /> """)
                            mapp(df3)
                        else:
                            mapp(df)
                            for i in df["Job Title"].head(10) :
                                indexx1=df[df['Job Title']==i].index[0]
                                if(df["Job Title"].iloc[indexx1] != "Nan"):
                                    st.subheader(i)     
                                    st.write("Role needed : "+str(df["Role"].iloc[indexx1]))
                                    st.write("Experience required : "+df["Job Experience Required"].iloc[indexx1])
                                    st.write("Key skills : "+df["Key Skills"].iloc[indexx1])
                                    st.write("Avarage salary : "+str(df["sal"].iloc[indexx1]))
                                    st.write("Location  : "+str(df["Location"].iloc[indexx1]))
                                    components.html("""<hr style="height:1px;border:none;color:#FF4B4B;background-color:#FF4B4B;" /> """)
                        col1, col2, col3 = st.columns(3)

                        with col1:
                                  st.write(' ')

                        with col2:
                            if st.button("See More Info On Job Offers"):
                              webbrowser.open_new_tab("https://app.powerbi.com/reportEmbed?reportId=094f0681-9bdc-4b4a-b75c-12f5412a9add&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLW5vcnRoLWV1cm9wZS1pLXByaW1hcnktcmVkaXJlY3QuYW5hbHlzaXMud2luZG93cy5uZXQvIn0%3D")

                        with col3:
                                  st.write(' ')   
                        
                    #-------------------------------------------------------------------------------------------------------
                    #Companies RECOMMANDATION

                    if selected == "Company" :
                        def extract_data(feed):
                                    
                                    with pdfplumber.open(feed) as pdf:
                                        pages = pdf.pages
                                        for p in pages:
                                            Data.append(p.extract_tables())
                                    return Data
                        user_input=""
                        uploaded_file_pdf = st.file_uploader("Please put your CV (PDF)", "pdf")
                        path = "Exemple_Resumes"
                        os.chmod(path, 0o777)
                        for f in os.listdir(path):
                            os.remove(os.path.join(path, f)) 
                        if uploaded_file_pdf is not None:
                            Data = extract_data(uploaded_file_pdf)
                            user_input = ' '.join(map(str, Data))
                            with open(os.path.join("Exemple_Resumes",uploaded_file_pdf.name),"wb") as f:
                              f.write(uploaded_file_pdf.getbuffer())
                        
                        Company=pd.read_csv("CompanyStartups.csv")
                        Company.drop(index=Company[Company['name'] == 'vide'].index, inplace=True)
                        
                        #Profile=pd.read_csv("Profile.csv")
                        
                        # tf = TfidfVectorizer(analyzer='word',ngram_range=(0, 3),min_df=0, stop_words='english')
                        # Company['spécialisations'] = Company['spécialisations'].replace("empty","")
                        # tfidf_matrix = tf.fit_transform(Company['spécialisations'],user_input)
                        
                        # cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
                        
                        # indices = pd.Series(Company.index, index=Company['spécialisations']).drop_duplicates()
                        
                    
                        # def get_recommendations(cosine_sim=cosine_sim):
                        #     idx = indices[title]
                        #     #print (idx)
                        #     sim_scores = list(enumerate(cosine_sim[idx]))
                        #     #print (sim_scores)
                        #     sim_scores = sorted(sim_scores, key=lambda x: x[:], reverse=True)
                        #     sim_scores = sim_scores[1:11]
                        #     Company_indices = [i[0] for i in sim_scores]
                        
                        
                        #     return Company['name'].iloc[Company_indices]


                        # Companies = get_recommendations()
                        # st.subheader("Companies in this field :")
                        # Companies
                        # for i in Companies:
                            
                        #     st.button(i):
                        #              
                        def get_ratio(row1,row2):
                                    name = row1
                                    name1 = row2
                                    return fuzz.token_set_ratio(name, name1)
                        list_company=[]
                        final_new_list_company=[]
                        for i in Company['description']:    
                            if(get_ratio(i,user_input) > 30):
                                        index=Company.index[Company['description']==i]
                                        list_company.append(Company['name'][index].values[0])
                                        final_new_list_company=list(set(list_company))
                                        final_new_list_company.insert(0,'Choose your Company')
                        companySelection =st.selectbox("Select A Company",options=[opt for opt in final_new_list_company])
                        for i in Company['name']:    
                            if (i==companySelection):
                                index=Company.index[Company['name']==i]
                                webbrowser.open_new_tab(Company['companyUrl'][index].values[0])  
                        col1, col2, col3 = st.columns(3)

                        with col1:
                                  st.write(' ')

                        with col2:
                            if st.button("See More Infos On Companies"):
                              webbrowser.open_new_tab("https://app.powerbi.com/reportEmbed?reportId=f3db6fa9-58cc-4f95-b229-938fd4634a9d&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLW5vcnRoLWV1cm9wZS1pLXByaW1hcnktcmVkaXJlY3QuYW5hbHlzaXMud2luZG93cy5uZXQvIn0%3D")

                        with col3:
                                  st.write(' ')        
                        
                                    
elif choice == "SignUp":
		st.subheader("New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")

                   
               





           
    
    
    


