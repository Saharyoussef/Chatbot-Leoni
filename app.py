import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
from qa_model import load_model
from data_management import load_qa_base, load_file_data, save_qa_base, log_change, update_history, load_password
from utils import get_answer_for_question, refresh_suggestions
from transformers import AutoTokenizer

# Paths
qa_base_path = r"C:\Users\Sahar Y\OneDrive\Bureau\stage leoni\ChatLeoni\qa_base.json"
# JSON file for the question-answer base
file_path = r"C:\Users\Sahar Y\OneDrive\Bureau\stage leoni\ChatLeoni\file.json"
# JSON file for PDF data
history_path = r"C:\Users\Sahar Y\OneDrive\Bureau\stage leoni\ChatLeoni\history.json"
# JSON file for history
history_log_path = r"C:\Users\Sahar Y\OneDrive\Bureau\stage leoni\ChatLeoni\history_log.json"
# JSON file for history log
password_file_path = r"C:\Users\Sahar Y\OneDrive\Bureau\stage leoni\ChatLeoni\password.txt"
# Text file for storing the password


model_name = 'mrm8488/bert-multi-cased-finetuned-xquadv1'

@st.cache_resource
# @st.cache_resource is a decorator used in Streamlit to optimize the performance of an application 
# by caching the result of a function.

def load_data():
    qa_base = load_qa_base(qa_base_path)
    file_data = load_file_data(file_path)
    return qa_base, file_data
#load_qa_base et load_file_data sont 2 fonctions définis dans data_management.py

@st.cache_resource
def load_bert_model_and_tokenizer(model_name):
    model = load_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def save_to_history(question, answer, source):
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as file:
            history_data = json.load(file)
    else:
        history_data = []

    history_data.append({
        'date': datetime.now().isoformat(),
        'question': question,
        'response': answer,
        'source': source,
        'status': 'Original'
    })

    with open(history_path, 'w', encoding='utf-8') as file:
        json.dump(history_data, file, ensure_ascii=False, indent=4)


#The save_to_history function: Records a question, its answer, and the source in a JSON history file.
    
    #- Loads the existing history if available, otherwise initializes a new one.
    #- Adds a new entry with the current date, the question, the answer, the source, and a status "Original".
    #- Saves the updated history to a JSON file.

    #Parameters:
    #- question (str): The question asked.
    #- answer (str): The answer provided.
    #- source (str): The source of the answer.

def update_history_log(history_df):
    history_log_data = []
    if os.path.exists(history_log_path):
        with open(history_log_path, 'r', encoding='utf-8') as file:
            history_log_data = json.load(file)

    for index, row in history_df.iterrows():
        existing_entry = next((entry for entry in history_log_data if 'index' in entry and entry['index'] == index), None)
        if existing_entry:
            existing_entry['changes'] += 1
            existing_entry['date'] = datetime.now().isoformat()
        else:
            history_log_data.append({
                'index': index,
                'date': datetime.now().isoformat(),
                'changes': 1
            })

    with open(history_log_path, 'w', encoding='utf-8') as file:
        json.dump(history_log_data, file, ensure_ascii=False, indent=4)

if 'qa_base' not in st.session_state:
    try:
        st.session_state.qa_base, st.session_state.file_data = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")

if 'model' not in st.session_state:
    try:
        st.session_state.model, st.session_state.tokenizer = load_bert_model_and_tokenizer(model_name)
    except Exception as e:
        st.error(f"Error loading model: {e}")

if os.path.exists(history_log_path):
    with open(history_log_path, 'r', encoding='utf-8') as file:
        history_log_data = json.load(file)
else:
    history_log_data = []

if os.path.exists(history_path):
    with open(history_path, 'r', encoding='utf-8') as file:
        history_data = json.load(file)
        for entry in history_data:
            if 'status' not in entry:
                entry['status'] = 'Original'
        st.session_state.history_df = pd.DataFrame(history_data)
else:
    st.session_state.history_df = pd.DataFrame(columns=['date', 'question', 'response', 'source', 'status'])

st.sidebar.image(r'C:\Users\Sahar Y\OneDrive\Bureau\stage leoni\ChatLeoni\img-removebg-preview.png', use_column_width=True)
st.sidebar.title("Information")
st.sidebar.markdown("""
    Bienvenue à l'application de questions-réponses.
    Posez vos questions et obtenez des réponses basées sur notre base de données et documents.
    Pour toute assistance, contactez email : xxxx@leoni.com
""")
page = st.sidebar.radio("Go to", ["Q&A", "Admin"])

def ui_setup():
    st.title("Question-Réponse")

def clear_input():
    st.session_state.input_question = ""
    refresh_suggestions()

if page == "Q&A":
    ui_setup()

    if 'qa_base' in st.session_state and len(st.session_state.qa_base) > 0:
        if 'random_questions' not in st.session_state:
            refresh_suggestions()

        st.write("Suggestions de questions:")
        for q in st.session_state.random_questions:
            if st.session_state.get('input_question'):
                st.button(q, disabled=True)
            else:
                if st.button(q):
                    question = q
                    with st.spinner("Recherche de la réponse..."):
                        answer, source = get_answer_for_question(question, model_name, history_path)
                    st.write(f"Question: {question}")
                    st.write(f"Réponse: {answer}")
                    st.write(f"Source: {source}")
                    save_to_history(question, answer, source)
                    break

    question = st.text_input("Posez votre question:", key='input_question')
    if st.button("Poser la question"):
        with st.spinner("Recherche de la réponse..."):
            answer, source = get_answer_for_question(question, model_name, history_path)
        st.write(f"Question: {question}")
        st.write(f"Réponse: {answer}")
        st.write(f"Source: {source}")
        save_to_history(question, answer, source)

    st.button("Effacer la question", on_click=clear_input)

elif page == "Admin":
    st.title("Administration des Questions et Réponses")
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    def authenticate(password):
        if password == load_password(password_file_path):
            st.session_state.authenticated = True
        else:
            st.error("Mot de passe incorrect")

    if not st.session_state.authenticated:
        password = st.text_input("Mot de passe", type="password")
        if st.button("Valider"):
            authenticate(password)

    if st.session_state.authenticated:
        # Search through history
        search_query = st.text_input("Rechercher dans l'historique", "")
        if search_query:
            filtered_df = st.session_state.history_df[st.session_state.history_df['question'].str.contains(search_query, case=False, na=False)]
        else:
            filtered_df = st.session_state.history_df

        # Filter by date range
        start_date = st.date_input("Date de début", value=datetime(2000, 1, 1))
        end_date = st.date_input("Date de fin", value=datetime.now())
        filtered_df = filtered_df[(pd.to_datetime(filtered_df['date'], format='ISO8601') >= pd.to_datetime(start_date)) & (pd.to_datetime(filtered_df['date'], format='ISO8601') <= pd.to_datetime(end_date))]

        # Display filtered history with highlights and status
        def highlight_changes(row):
            color = 'background-color: #A80000;' if row['status'] == 'Changed' else ''
            return [color for _ in row]

        st.dataframe(filtered_df.style.apply(highlight_changes, axis=1))

        # Add question-response pair
        st.subheader("Ajouter une nouvelle question et réponse")
        new_question = st.text_area("Nouvelle Question")
        new_response = st.text_area("Nouvelle Réponse")
        if st.button("Ajouter à la base QA"):
            if new_question and new_response:
                st.session_state.qa_base[new_question] = new_response
                save_qa_base(qa_base_path, st.session_state.qa_base)
                st.success("Nouvelle question et réponse ajoutées à la base QA")

        # Delete history entry
        selected_indices = st.multiselect("Sélectionnez les indices des entrées à supprimer", filtered_df.index)
        if st.button("Supprimer l'entrée sélectionnée"):
            if selected_indices:
                for index in selected_indices:
                    log_change(
                        index,
                        filtered_df.at[index, 'question'],
                        filtered_df.at[index, 'response'],
                        "Deleted",
                        history_log_path
                    )
                st.session_state.history_df.drop(selected_indices, inplace=True)
                update_history(st.session_state.history_df, history_path)
                st.success("Entrée(s) supprimée(s)")

        # Edit history entry
        selected_index = st.selectbox("Sélectionnez l'indice de l'entrée à modifier", filtered_df.index)
        if selected_index is not None:
            new_question = st.text_area("Modifier la question", filtered_df.at[selected_index, 'question'])
            new_response = st.text_area("Modifier la réponse", filtered_df.at[selected_index, 'response'])
            if st.button("Enregistrer les modifications"):
                old_response = filtered_df.at[selected_index, 'response']
                st.session_state.history_df.at[selected_index, 'question'] = new_question
                st.session_state.history_df.at[selected_index, 'response'] = new_response
                st.session_state.history_df.at[selected_index, 'status'] = 'Changed'
                log_change(
                    selected_index,
                    new_question,
                    old_response,
                    new_response,
                    history_log_path
                )
                update_history(st.session_state.history_df, history_path)
                st.success("Entrée modifiée")

        # Validate question and add to QA base
        selected_index = st.selectbox("Sélectionnez l'indice de l'entrée à valider", filtered_df.index, key="validate_index")
        if selected_index is not None:
            if st.button("Valider la question sélectionnée et l'ajouter à la base de données QA"):
                question = filtered_df.at[selected_index, 'question']
                response = filtered_df.at[selected_index, 'response']
                if question not in st.session_state.qa_base:
                    st.session_state.qa_base[question] = response
                    save_qa_base(qa_base_path, st.session_state.qa_base)
                    st.success("Question validée et ajoutée à la base de données QA")

        if st.button("Télécharger l'historique en JSON"):
            history_json = st.session_state.history_df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="Télécharger l'historique en JSON",
                data=history_json,
                file_name='history_log.json',
                mime='application/json'
            )
