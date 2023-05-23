import streamlit as st
from functions import *
from PIL import Image
from streamlit import source_util
from streamlit.runtime.scriptrunner import RerunData, RerunException

emb_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl') 
# replace logo
image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
st.set_page_config(page_title="TakeNote",page_icon=image_logo)

# st.set_page_config(page_title="Question/Answering", page_icon="🔎")
st.sidebar.header("Semantic Search")
st.markdown("##  Semantic Search")

def gen_sentiment(text):
    '''Generate sentiment of given text'''
    return sent_pipe(text)[0]['label']

def gen_annotated_text(df):
    '''Generate annotated text'''
    
    tag_list=[]
    for row in df.itertuples():
        label = row[2]
        text = row[1]
        if label == 'Positive':
            tag_list.append((text,label,'#8fce00'))
        elif label == 'Negative':
            tag_list.append((text,label,'#f44336'))
        else:
            tag_list.append((text,label,'#000000'))
        
    return tag_list

if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    st.session_state['authenticator'].logout("Logout", "sidebar")

    bi_enc_dict = {'mpnet-base-v2':"all-mpnet-base-v2",
                'instructor-base': 'hkunlp/instructor-base'}

    search_input = st.text_input(
            label='Enter Your Search Query',value= "What key challenges did the business face?", key='search')
            
    sbert_model_name = st.sidebar.selectbox("Embedding Model", options=list(bi_enc_dict.keys()), key='sbox')
            
    chunk_size = st.sidebar.slider("Number of Chars per Chunk of Text",min_value=500,max_value=2000,value=1000)
    overlap_size = st.sidebar.slider("Number of Overlap Chars in Search Response",min_value=50,max_value=300,value=50)

    try:

        if search_input:
            
            if "sen_df" in st.session_state and "passages" in st.session_state:
            
                ## Save to a dataframe for ease of visualization
                sen_df = st.session_state['sen_df']

                title = st.session_state['title']

                embedding_model = bi_enc_dict[sbert_model_name]
                                
                with st.spinner(
                    text=f"Loading {embedding_model} embedding model and Generating Response..."
                ):
                    
                    docsearch = process_corpus(st.session_state['passages'],title, embedding_model)
                    st.write(docsearch)

                    result = embed_text(search_input,title,embedding_model,emb_tokenizer, docsearch)
                    st.write(result)

                references = [doc.page_content for doc in result['source_documents']]

                answer = result['result']

                sentiment_label = gen_sentiment(answer)
                    
                ##### Sematic Search #####
                
                df = pd.DataFrame.from_dict({'Text':[answer],'Sentiment':[sentiment_label]})
                
                
                text_annotations = gen_annotated_text(df)[0]            
                
                with st.expander(label='Query Result', expanded=True):
                    annotated_text(text_annotations)
                    
                with st.expander(label='References from Corpus used to Generate Result'):
                    for ref in references:
                        st.write(ref)
                    
            else:
                
                st.write('Please ensure you have entered the YouTube URL or uploaded the Earnings Call file')
                
        else:
        
            st.write('Please ensure you have entered the YouTube URL or uploaded the Earnings Call file')  
            
    except RuntimeError:
    
        st.write('Please ensure you have entered the YouTube URL or uploaded the Earnings Call file')  
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