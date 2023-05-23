import streamlit as st
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from functions import *
import validators
from PIL import Image
from streamlit import source_util
from streamlit.runtime.scriptrunner import RerunData, RerunException

import textwrap
# replace logo
image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
st.set_page_config(page_title="TakeNote",page_icon=image_logo)

language_code = {'ab': 'Abkhaz',
'aa': 'Afar',
'af': 'Afrikaans',
'ak': 'Akan',
'sq': 'Albanian',
'am': 'Amharic',
'ar': 'Arabic',
'an': 'Aragonese',
'hy': 'Armenian',
'as': 'Assamese',
'av': 'Avaric',
'ae': 'Avestan',
'ay': 'Aymara',
'az': 'Azerbaijani',
'bm': 'Bambara',
'ba': 'Bashkir',
'eu': 'Basque',
'be': 'Belarusian',
'bn': 'Bengali',
'bh': 'Bihari',
'bi': 'Bislama',
'bs': 'Bosnian',
'br': 'Breton',
'bg': 'Bulgarian',
'my': 'Burmese',
'ca': 'Catalan; Valencian',
'ch': 'Chamorro',
'ce': 'Chechen',
'ny': 'Chichewa; Chewa; Nyanja',
'zh': 'Chinese',
'cv': 'Chuvash',
'kw': 'Cornish',
'co': 'Corsican',
'cr': 'Cree',
'hr': 'Croatian',
'cs': 'Czech',
'da': 'Danish',
'dv': 'Divehi; Maldivian;',
'nl': 'Dutch',
'dz': 'Dzongkha',
'en': 'English',
'eo': 'Esperanto',
'et': 'Estonian',
'ee': 'Ewe',
'fo': 'Faroese',
'fj': 'Fijian',
'fi': 'Finnish',
'fr': 'French',
'ff': 'Fula',
'gl': 'Galician',
'ka': 'Georgian',
'de': 'German',
'el': 'Greek, Modern',
'gn': 'Guaraní',
'gu': 'Gujarati',
'ht': 'Haitian',
'ha': 'Hausa',
'he': 'Hebrew (modern)',
'hz': 'Herero',
'hi': 'Hindi',
'ho': 'Hiri Motu',
'hu': 'Hungarian',
'ia': 'Interlingua',
'id': 'Indonesian',
'ie': 'Interlingue',
'ga': 'Irish',
'ig': 'Igbo',
'ik': 'Inupiaq',
'io': 'Ido',
'is': 'Icelandic',
'it': 'Italian',
'iu': 'Inuktitut',
'ja': 'Japanese',
'jv': 'Javanese',
'kl': 'Kalaallisut',
'kn': 'Kannada',
'kr': 'Kanuri',
'ks': 'Kashmiri',
'kk': 'Kazakh',
'km': 'Khmer',
'ki': 'Kikuyu, Gikuyu',
'rw': 'Kinyarwanda',
'ky': 'Kirghiz, Kyrgyz',
'kv': 'Komi',
'kg': 'Kongo',
'ko': 'Korean',
'ku': 'Kurdish',
'kj': 'Kwanyama, Kuanyama',
'la': 'Latin',
'lb': 'Luxembourgish',
'lg': 'Luganda',
'li': 'Limburgish',
'ln': 'Lingala',
'lo': 'Lao',
'lt': 'Lithuanian',
'lu': 'Luba-Katanga',
'lv': 'Latvian',
'gv': 'Manx',
'mk': 'Macedonian',
'mg': 'Malagasy',
'ms': 'Malay',
'ml': 'Malayalam',
'mt': 'Maltese',
'mi': 'Māori',
'mr': 'Marathi (Marāṭhī)',
'mh': 'Marshallese',
'mn': 'Mongolian',
'na': 'Nauru',
'nv': 'Navajo, Navaho',
'nb': 'Norwegian Bokmål',
'nd': 'North Ndebele',
'ne': 'Nepali',
'ng': 'Ndonga',
'nn': 'Norwegian Nynorsk',
'no': 'Norwegian',
'ii': 'Nuosu',
'nr': 'South Ndebele',
'oc': 'Occitan',
'oj': 'Ojibwe, Ojibwa',
'cu': 'Old Church Slavonic',
'om': 'Oromo',
'or': 'Oriya',
'os': 'Ossetian, Ossetic',
'pa': 'Panjabi, Punjabi',
'pi': 'Pāli',
'fa': 'Persian',
'pl': 'Polish',
'ps': 'Pashto, Pushto',
'pt': 'Portuguese',
'qu': 'Quechua',
'rm': 'Romansh',
'rn': 'Kirundi',
'ro': 'Romanian, Moldavan',
'ru': 'Russian',
'sa': 'Sanskrit (Saṁskṛta)',
'sc': 'Sardinian',
'sd': 'Sindhi',
'se': 'Northern Sami',
'sm': 'Samoan',
'sg': 'Sango',
'sr': 'Serbian',
'gd': 'Scottish Gaelic',
'sn': 'Shona',
'si': 'Sinhala, Sinhalese',
'sk': 'Slovak',
'sl': 'Slovene',
'so': 'Somali',
'st': 'Southern Sotho',
'es': 'Spanish; Castilian',
'su': 'Sundanese',
'sw': 'Swahili',
'ss': 'Swati',
'sv': 'Swedish',
'ta': 'Tamil',
'te': 'Telugu',
'tg': 'Tajik',
'th': 'Thai',
'ti': 'Tigrinya',
'bo': 'Tibetan',
'tk': 'Turkmen',
'tl': 'Tagalog',
'tn': 'Tswana',
'to': 'Tonga',
'tr': 'Turkish',
'ts': 'Tsonga',
'tt': 'Tatar',
'tw': 'Twi',
'ty': 'Tahitian',
'ug': 'Uighur, Uyghur',
'uk': 'Ukrainian',
'ur': 'Urdu',
'uz': 'Uzbek',
've': 'Venda',
'vi': 'Vietnamese',
'vo': 'Volapük',
'wa': 'Walloon',
'cy': 'Welsh',
'wo': 'Wolof',
'fy': 'Western Frisian',
'xh': 'Xhosa',
'yi': 'Yiddish',
'yo': 'Yoruba',
'za': 'Zhuang, Chuang',
'zu': 'Zulu'}

footer="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
# print'state = ', st.session_state)
import json
with open('segment.json', 'w') as f:
    json.dump(st.session_state['segments'],f, indent=4)
    
st.markdown(footer, unsafe_allow_html=True)
if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    st.session_state['authenticator'].logout("Logout", "sidebar")
    st.sidebar.header("Transcription")
    st.markdown("## Transcription Audio")

    if "title" not in st.session_state:
        st.session_state.title = ''   
    if 'passages' not in st.session_state:
        st.session_state['passages'] = ""
    if "language" not in st.session_state:
        st.session_state['language'] = ""
    if "running_time_transcript" not in st.session_state:
        st.session_state['running_time_transcript'] = 0.0
        
    passages = st.session_state['passages']
    title = st.session_state['title']
    language = st.session_state['language']
    running_time = st.session_state['running_time_transcript']
    if running_time / 3600 > 1.0:
        running_time = str(running_time / 3600).format("{:. 2f}")[:4] + ' hours'
    elif running_time / 60 > 1.0:
        running_time = str(running_time / 60).format("{:. 2f}")[:4] + ' minutes'
    else:
        running_time = str(running_time).format("{:. 2f}")[:4] + ' seconds'
        
    if len(passages) > 0:
        write_data = 'running_time: ' + str(running_time) + '\n' + passages
        transcript_downloader(write_data, "Download transcription", header="transcription", user_name=st.session_state['username'])

    # sentiment, sentences = sentiment_pipe(passages)
    # with st.expander("See Transcribed Text"):
    if (language is not None) and len(language) > 0:
        st.write(f'detected language: {language_code[language]}')
    if running_time is not None:
        st.write(f'running time: {running_time}')
                 
    st.write(f"Number of words: {len(passages.split())}")
    st.subheader(f'Title: {title}')
    st.write(passages)        
            
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