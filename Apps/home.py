import streamlit as st
import leafmap.foliumap as leafmap
import folium
import pandas as pd
import leafmap as L

def app(data):
    st.title("All the job offers")

    st.markdown("  "
        
    )
    

    m = leafmap.Map(locate_control=True)
    #data = 'https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_cities.csv'
    m.add_points_from_xy(data, x="Longitude", y="Latitude")
    m.add_basemap("ROADMAP")
    m.to_streamlit(height=700)
  