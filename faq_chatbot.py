import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample FAQs
faqs = [
    {"question": "What is the duration of the internship?", "answer": "It is 1 month long."},
    {"question": "Is this internship paid?", "answer": "No, it's unpaid."},
    {"question": "Do I get a certificate?", "answer": "Yes, after completing tasks."},
    {"question": "What is the best things that I can get?", "answer": "You will gain experience and knowledge about projects."}
]

# User input
user_input = input("Ask your question: ")

# Match logic
questions = [faq["question"] for faq in faqs]
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(questions + [user_input])
similarity = cosine_similarity(vectors[-1], vectors[:-1])
index = similarity.argmax()
print("Answer:", faqs[index]["answer"])
