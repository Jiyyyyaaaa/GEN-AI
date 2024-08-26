# Website Scraping
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin
from transformers import T5ForConditionalGeneration, T5Tokenizer

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]
    return links

def save_links_to_file(links, filename='links.json'):
    with open(filename, 'w') as file:
        json.dump(links, file)

# Example usage
website_url = 'https://www.geeksforgeeks.org/'
links = scrape_website(website_url)
save_links_to_file(links)


# Retrieve and Save Webpage Content

def fetch_and_save_content(links, filename='content.json'):
    content_dict = {}
    for link in links:
        try:
            response = requests.get(link)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ')
            content_dict[link] = text.strip()
        except Exception as e:
            print(f"Failed to fetch {link}: {e}")
    
    with open(filename, 'w') as file:
        json.dump(content_dict, file)

# Example usage
fetch_and_save_content(links)




# Load the pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def generate_questions(text, max_length=80):
    input_text = "generate questions: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    
    # Generate output using the model
    outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=10)
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

# Example usage
content = "Artificial intelligence (AI) is the simulation of human intelligence in machines..."
questions = generate_questions(content[:512])  # T5 has a token limit; truncate content if necessary
for i, question in enumerate(questions, 1):
    print(f"Question {i}: {question}")





