from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance

def load_model(model_name='mrm8488/bert-multi-cased-finetuned-xquadv1'):
    # Load and return a pre-trained model specified by the model_name.
    model = AutoModel.from_pretrained(model_name)
    return model

def find_most_similar_document(documents, question, threshold=0.001):
    #documents: List of document dictionaries, where each dictionary has a 'section_text' key.
    #question: The question to compare with the documents.
    #threshold: Similarity threshold for considering a document as relevant.
    
    if not documents:
        return None
    vectorizer = TfidfVectorizer()
    #Creates an instance of TfidfVectorizer, which converts text data into numerical features based on TF-IDF (Term Frequency-Inverse Document Frequency).
    document_texts = [document['section_text'] for document in documents]
    document_texts.append(question)
    document_vectors = vectorizer.fit_transform(document_texts)
    #Converts the list of document texts (including the question) into numerical vectors using TF-IDF.
    similarities = cosine_similarity(document_vectors[-1], document_vectors[:-1])[0]
    #The cosine_similarity function returns a similarity score for each document compared to the question.
    max_similarity = similarities.max()
    if max_similarity >= threshold:
        max_similarity_index = similarities.argmax()
        return documents[max_similarity_index]
    else:
        return None
    
    # Returns the most similar document to the given question based on TF-IDF similarity.
    # If no documents are provided or if the highest similarity is below the threshold, returns None.

def find_most_similar_question(qa_base, question, model_name='mrm8488/bert-multi-cased-finetuned-xquadv1', threshold_bert=0.9, threshold_levenshtein=3):
    #threshold_bert: Cosine similarity threshold for considering a question as similar based on BERT embeddings.
    #threshold_levenshtein: Levenshtein distance threshold for considering a question as similar based on textual similarity.
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    questions = list(qa_base.keys())
    question_texts = [question] + questions

    encoded_inputs = tokenizer(question_texts, padding=True, truncation=True, return_tensors="pt")
    #Tokenizes the question_texts using the tokenizer. The padding and truncation ensure that all sequences are of the same length.
    #Converts the tokenized texts into PyTorch tensors.
    
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        question_embeddings = outputs.last_hidden_state[:, 0, :]
        
        #Passes the tokenized inputs through the model to get the embeddings.
        #outputs.last_hidden_state[:, 0, :] extracts the embeddings for the [CLS] token, which is often used to represent the entire sequence.

    question_embedding = question_embeddings[0]
    questions_embeddings = question_embeddings[1:]
    similarities = torch.cosine_similarity(question_embedding.unsqueeze(0), questions_embeddings)
    max_similarity_index = similarities.argmax()
    max_similarity = similarities[max_similarity_index]

    best_question = None
    min_distance = float('inf')

    for base_question in qa_base:
        current_distance = levenshtein_distance(question, base_question)
        if current_distance < min_distance:
            min_distance = current_distance
            best_question = base_question

    if min_distance <= threshold_levenshtein or max_similarity > threshold_bert:
        return best_question, max_similarity.item()
    else:
        return None, 0.0
    
# Finds the most similar question in the QA base to the given question using both BERT embeddings and Levenshtein distance.
# Uses BERT embeddings to compute cosine similarity and Levenshtein distance for textual similarity.
# Returns the most similar question and its similarity score if either measure exceeds the specified thresholds, otherwise returns None and 0.0.

def answer_question(context, question):
    return context

