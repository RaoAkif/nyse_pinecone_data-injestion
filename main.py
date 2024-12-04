from stock_info import get_stock_info
from embeddings import cosine_similarity_between_sentences

# Example usage for cosine similarity between two sentences
# sentence1 = "I like walking to the park"
# sentence2 = "I like walking to the office"
# similarity = cosine_similarity_between_sentences(sentence1, sentence2)

# Fetch stock information and print it
aapl_info = get_stock_info('AAPL')
print(aapl_info)

# Compare descriptions
aapl_description = aapl_info['Business Summary']
company_description = "I want to find companies that make food and are headquartered in California"

similarity = cosine_similarity_between_sentences(aapl_description, company_description)
