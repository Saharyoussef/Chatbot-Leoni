import json
import os
import pandas as pd
from datetime import datetime

def load_password(file_path):
    # Reads and returns the password from a text file.
    with open(file_path, 'r') as file:
        password = file.read().strip()
        #removes any leading or trailing whitespace (including newlines)
    return password

def log_change(index, question, old_response, new_response, history_log_path):
    # Logs a change in responses, adding an entry to a JSON file that tracks changes over time.
    if os.path.exists(history_log_path):
        with open(history_log_path, 'r', encoding='utf-8') as file:
            history_log_data = json.load(file)
    else:
        history_log_data = []

    change_entry = {
        'index': index,
        'date': datetime.now().isoformat(),
        'question': question,
        'old_response': old_response,
        'last_correction': new_response
    }

    history_log_data.append(change_entry)

    with open(history_log_path, 'w', encoding='utf-8') as file:
        json.dump(history_log_data, file, ensure_ascii=False, indent=4)
        #json.dump is a function used to convert a Python object into a JSON string and write it directly to a file.

def update_history(history_df, history_path):
    # Saves the history DataFrame to a JSON file, preserving records of changes.
    history_df.to_json(history_path, orient='records', date_format='iso')

def save_qa_base(qa_base_path, qa_base):
    # Saves the current QA base (question-answer pairs) to a JSON file.
    try:
        with open(qa_base_path, 'w', encoding='utf-8') as file:
            json.dump(qa_base, file, ensure_ascii=False, indent=4)
        print(f"QA base saved to {qa_base_path}")
    except Exception as e:
        print(f"Error saving QA base to {qa_base_path}: {e}")

def load_qa_base(json_path):
    # Loads the QA base from a JSON file or creates an empty base if the file doesn't exist.
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                qa_base = json.load(file)
            print(f"Loaded QA base from {json_path}")
        except Exception as e:
            print(f"Error loading QA base from {json_path}: {e}")
            qa_base = {}
    else:
        qa_base = {}
        print(f"No QA base found at {json_path}, starting with an empty QA base")
    print(f"Total QA pairs loaded: {len(qa_base)}")
    return qa_base

def load_file_data(file_path):
    # Loads data from a JSON file, returning an empty list if the file doesn't exist.
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                file_data = json.load(file)
            print(f"Loaded file data from {file_path}")
        except Exception as e:
            print(f"Error loading file data from {file_path}: {e}")
            file_data = []
    else:
        file_data = []
        print(f"No file data found at {file_path}, starting with an empty file data")
    print(f"Total documents loaded: {len(file_data)}")
    return file_data

def load_history(history_path):
    # Loads the history of changes from a JSON file into a pandas DataFrame, adding a 'status' column if missing.
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as file:
            history_data = json.load(file)
            for entry in history_data:
                if 'status' not in entry:
                    entry['status'] = 'Original'
            return pd.DataFrame(history_data)
    return pd.DataFrame(columns=['date', 'question', 'response', 'source', 'status'])
