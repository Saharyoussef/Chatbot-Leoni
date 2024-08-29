from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance

def load_model(model_name='mrm8488/bert-multi-cased-finetuned-xquadv1'):
    # Load and return a pre-trained model specified by the model_name.
    model = AutoModel.from_pretrained(model_name)
    return model

#Change: Enhance doc search by changing Threshold flexibility
def find_most_similar_document(documents, question):
    if not documents:
        return None

    vectorizer = TfidfVectorizer()
    document_texts = [document['section_text'] for document in documents]
    document_texts.append(question)
    document_vectors = vectorizer.fit_transform(document_texts)

    similarities = cosine_similarity(document_vectors[-1], document_vectors[:-1])[0]
    max_similarity = similarities.max()

    # Allow dynamic adjustment of the threshold
    current_threshold = adjust_threshold(max_similarity, question)

    if max_similarity >= current_threshold:
        max_similarity_index = similarities.argmax()
        return documents[max_similarity_index]
    else:
        return None
   
    # Returns the most similar document to the given question based on TF-IDF similarity.
    # If no documents are provided or if the highest similarity is below the threshold, returns None.

def adjust_threshold(similarity, question):
    # Example logic: Increase threshold if similarity is low, or based on question length
    if similarity < 0.5:
        return 0.003  # Lower the threshold for more lenient matching
    elif len(question.split()) > 10:
        return 0.004  # Higher threshold for more precise matching in longer queries
    else:
        return 0.004  # Default threshold


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