import streamlit as st
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from functions import *
from optimum.onnxruntime import ORTModelForSequenceClassification
import validators
import textwrap
from PIL import Image
from streamlit import source_util
from streamlit.runtime.scriptrunner import RerunData, RerunException

# replace logo
image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
st.set_page_config(page_title="Sentiment",page_icon=image_logo)

footer="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
st.markdown(footer, unsafe_allow_html=True)
if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    st.session_state['authenticator'].logout("Logout", "sidebar")
    st.sidebar.header("Sentiment Analysis using AI")
    st.markdown("## Sentiment Analysis")

    if "title" not in st.session_state:
        st.session_state.title = ''   
    try:
        if ("passages" in st.session_state) and (len(st.session_state['passages']) > 0):
                sentiment, sentences = sentiment_pipe(st.session_state['passages'])
                
                # Save to a dataframe for ease of visualization
                sen_df = pd.DataFrame(sentiment)
                sen_df['text'] = sentences
                grouped = pd.DataFrame(sen_df['label'].value_counts()).reset_index()
                grouped.columns = ['sentiment','count']
                st.session_state['sen_df'] = sen_df
                
                # Display number of positive, negative and neutral sentiments
                fig = px.bar(grouped, x='sentiment', y='count', color='sentiment', color_discrete_map={"Negative":"firebrick","Neutral":\
                                                                                                    "navajowhite","Positive":"darkgreen"},\
                                                                                                    title='Sentiment Analysis')
                fig.update_layout(
                    showlegend=False,
                    autosize=True,
                    margin=dict(
                        l=25,
                        r=25,
                        b=25,
                        t=50,
                        pad=2
                    )
                )
                
                st.plotly_chart(fig)
                
            # Display sentiment score
            # try:
                sentiment_score = 0.0
                _sc_pos = 0.0
                _sc_neg = 0.0
                if 'Positive' in grouped['sentiment'].to_list():
                    _sc_pos = grouped[grouped['sentiment']=='Positive']['count'].iloc[0]
                
                if 'Negative' in grouped['sentiment'].to_list():
                    _sc_neg = grouped[grouped['sentiment']=='Negative']['count'].iloc[0]
                sentiment_score = (_sc_pos - _sc_neg) * 100/len(sen_df)
                            
                fig_1 = go.Figure()
                
                fig_1.add_trace(go.Indicator(
                    mode = "delta",
                    value = sentiment_score,
                    domain = {'row': 1, 'column': 1}))
                
                fig_1.update_layout(
                    template = {'data' : {'indicator': [{
                        'title': {'text': "Sentiment Score"},
                        'mode' : "number+delta+gauge",
                        'delta' : {'reference': 1}}]
                                        }},
                    autosize=False,
                    width=250,
                    height=250,
                    margin=dict(
                        l=5,
                        r=5,
                        b=5,
                        pad=2
                    )
                )
            
                with st.sidebar:
                
                    st.plotly_chart(fig_1)
                hd = sen_df.text.apply(lambda txt: '<br>'.join(textwrap.wrap(txt, width=70)))
                # Display negative sentence locations
                fig = px.scatter(sen_df, y='label', color='label', size='score', hover_data=[hd], color_discrete_map={"Negative":"firebrick","Neutral":"navajowhite","Positive":"darkgreen"}, title='Sentiment Score Distribution')
                fig.update_layout(
                    showlegend=False,
                    autosize=True,
                    width=800,
                    height=350,
                    margin=dict(
                        b=5,
                        t=50,
                        pad=4
                    )
                )
            
                st.plotly_chart(fig)
            # except Exception as e:
            #     print(e)
            #     st.error('The input audio is too short. Cannot provide a Sentiment Analysis!')
            
        else:
            st.write("No YouTube URL or file upload detected")
            
    except (AttributeError, TypeError):
        st.write("No YouTube URL or file upload detected")        
else:
    st.subheader("Authentication Required...")
    pages = source_util.get_pages('HOME.py')
    for page_hash, config in pages.items():
        if config["page_name"] == 'HOME':
            page_hash_home = page_hash

    if st.button("Return to Home to login"):
        raise RerunException(
            RerunData(
                page_script_hash=page_hash_home,
                page_name='HOME',
            )
        )
        raise ValueError(f"Could not find page HOME. Must be one of HOME")