from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
from fuzzywuzzy import fuzz

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


#Change 2: Add fuzzy matching for enhancing handling of Typos and Misspellings
def find_most_similar_question(qa_base, question, model_name='mrm8488/bert-multi-cased-finetuned-xquadv1', threshold_bert=0.85, threshold_levenshtein=4):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    questions = list(qa_base.keys())

    # Use fuzzy matching to find the most similar question
    best_fuzz_match = None
    highest_fuzz_score = 0
    for q in questions:
        fuzz_score = fuzz.ratio(question, q)
        #compares two strings and returns a similarity score between 0 and 100, where 100 means the strings are identical.
        if fuzz_score > highest_fuzz_score:
            highest_fuzz_score = fuzz_score
            best_fuzz_match = q

    if highest_fuzz_score > 80:  # Adjust based on your preference
        return best_fuzz_match, 1.0

    # Fallback to BERT-based similarity if fuzzy matching doesn't suffice
    question_texts = [question] + questions
    encoded_inputs = tokenizer(question_texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        question_embeddings = outputs.last_hidden_state[:, 0, :]
        #The input question and all questions from the qa_base are tokenized and passed through the BERT model to obtain embeddings.
        #The embeddings represent the semantic meaning of the text.
        
    question_embedding = question_embeddings[0]
    #question_embeddings[0] refers to the first element in the tensor, which is the embedding for the input question (the one you want to find a match for).
    questions_embeddings = question_embeddings[1:]
    #extracts the embeddings of all the other questions from the qa_base, excluding the input question.
    similarities = torch.cosine_similarity(question_embedding.unsqueeze(0), questions_embeddings)
    max_similarity_index = similarities.argmax()
    max_similarity = similarities[max_similarity_index]

    if max_similarity > threshold_bert:
        return questions[max_similarity_index], max_similarity.item()
    else:
        return best_fuzz_match, highest_fuzz_score / 100

def answer_question(context, question):
    return context

#Fuzzy Matching: This method is fast and works well for small typos or minor variations in the input text, where the actual characters are mostly the same but arranged differently.
#BERT-based Similarity: This method is more powerful and captures the semantic meaning of the text, which allows it to recognize that different words or phrases might mean the same thing.