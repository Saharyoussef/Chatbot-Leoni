import random
import pandas as pd
import streamlit as st
from datetime import datetime
from qa_model import find_most_similar_document, find_most_similar_question, answer_question

def refresh_suggestions():
    st.session_state.random_questions = random.sample(
        list(st.session_state.qa_base.keys()), min(4, len(st.session_state.qa_base))
    )

def get_answer_for_question(question, model_name, history_path):
    answer = None
    source = None
    if question:
        most_similar_question, similarity_score = find_most_similar_question(st.session_state.qa_base, question, model_name=model_name)
        if similarity_score > 0.8:
            answer = st.session_state.qa_base[most_similar_question]
            source = 'QA Base'
            #st.session_state: This is a Streamlit feature that provides a way to store and manage state across multiple runs of a Streamlit app. 
            #It acts like a dictionary where you can store and retrieve values that persist throughout the user's interaction with the app.
        
        else:
            if len(st.session_state.file_data) == 0:
                st.write("Aucun document trouvé")
                #st.session_state.file_data is a variable used to store data related to files within the Streamlit app’s session state.
            else:
                try:
                    most_similar_document = find_most_similar_document(st.session_state.file_data, question)
                    if most_similar_document is None:
                        suggestions = []
                        for section in st.session_state.file_data:
                            if question in section['section_text']:
                                suggestions.append(section)
                        if suggestions:
                            st.write("Suggestions où vous pouvez trouver l'information souhaitée:")
                            for suggestion in suggestions:
                                st.write(f"Document: {suggestion['filename']}, Section: {suggestion['section_number']}")
                                st.write(suggestion['section_text'])
                                st.write("=" * 50)
                            source = 'Suggestions'
                        else:
                            st.write("Veuillez contacter email: xxxx@leoni.com")
                    else:
                        context = most_similar_document['section_text']
                        answer = answer_question(context, question)
                        source = f"Document: {most_similar_document['filename']}"
                except Exception as e:
                    st.write(f"Erreur - {str(e)}")
        if answer is None:
            answer = "Aucun document similaire trouvé. Veuillez nous contacter pour plus d'informations sur l'email xxxx@leoni.com."

            source = 'None'
        new_entry = pd.DataFrame({
            'date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'question': [question],
            'response': [answer],
            'source': [source]
        })
        st.session_state.history_df = pd.concat([st.session_state.history_df, new_entry], ignore_index=True)
        st.session_state.history_df.to_json(history_path, orient='records', date_format='iso')
        refresh_suggestions()
    return answer, source
