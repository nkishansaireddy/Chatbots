from nltk.chat.util import Chat, reflections
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import wikipediaapi

# Initialize the sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Create a Wikipedia API object
wiki_api = wikipediaapi.Wikipedia(user_agent='my-application/1.0')

# Define patterns and responses for the chatbot
patterns = [
    (r'(hi|hello|hey)', ['Hello!', 'Hi there!', 'Hey!']),
    (r'how are you\?', ['I am doing well, thank you!', 'I\'m good, thanks for asking.']),
    (r'whats your name\?', ['You can call me Assistant.', 'I\'m your virtual assistant.']),
    (r'my name is .*', ['Nice to meet you!', 'Hello!', 'Hi!']),
    (r'quit', ['Bye! Take care.', 'Goodbye!']),
    (r'i love (.*)', ['That\'s great to hear!', 'Love is a beautiful thing.']),
    (r'what can you do\?', ['I can help you with various tasks like providing information, answering questions, etc.']),
    (r'(tell me|what|who is) (.*)', ['Let me find some information about {}.', 'Sure, let me look up information about {}.']),
    (r'(more|give me more content|give me more|give extra content|provide extra information|give more data)', 
     ['Sure, here is more information.', 'Here is additional content.', 'Let me give you more details.', 'I can provide extra information.']),
]

# Define a function to extract keywords
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    keywords = [word.lower() for word in words if word.lower() not in stop_words]
    return keywords

# Define a function to get information from Wikipedia
def fetch_wikipedia_summary(topic):
    try:
        page = wiki_api.page(topic)
        if page.exists():
            summary = page.summary
            # Split summary into paragraphs
            paragraphs = summary.split('\n')
            return paragraphs
        else:
            return None
    except wikipediaapi.exceptions.WikipediaException as e:
        return None
    
# Define a function to perform intent recognition
def recognize_intent(user_input):
    if any(keyword in user_input for keyword in ['tell me about', 'what is', 'who is']):
        return 'WikipediaSummary'
    elif any(keyword in user_input for keyword in ['more info','still','more','provide more info','give more info','give me more info','provide more data','give more data','please elaborate on','expand on','can you tell me more about','give me more content about','give me more','give extra content about','provide extra information about','give extra information about','give more data about','provide additional details on','I want to know more about','please give me more insights on','could you expand on','an you elaborate further on','i need more information on','provide more details about']):
        return 'MoreContent'
    else:
        return 'Default'

# Create a chatbot using the defined patterns
chatbot = Chat(patterns, reflections)

# Initialize remaining paragraphs as an empty list
remaining_paragraphs = []

# Define a function to interact with the chatbot
def get_chatbot_response(user_input, user_name):
    global remaining_paragraphs
    keywords = extract_keywords(user_input)
    # Sentiment analysis
    sentiment_score = sentiment_analyzer.polarity_scores(user_input)['compound']
    if sentiment_score >= 0.2:
        response = "It sounds like you're feeling positive. "
    elif sentiment_score <= -0.2:
        response = "It sounds like you're feeling negative. "
    else:
        response = ""

    # Recognize user intent
    intent = recognize_intent(user_input)
    
    # Perform actions based on intent
    if intent == 'WikipediaSummary':
        for keyword in ['tell me about', 'what is', 'who is']:
            if keyword in user_input:
                topic = user_input.split(keyword)[-1].strip()
                break
        summary = fetch_wikipedia_summary(topic)
        if summary:
            # Save remaining paragraphs
            remaining_paragraphs = summary[1:]
            response += summary[0]
        else:
            response += f"Sorry, I couldn't find information about {topic}."
    elif intent == 'MoreContent':
        if remaining_paragraphs:
            response += remaining_paragraphs.pop(0)
        else:
            response += "The summary is completed."
    else:
        response_from_chatbot = chatbot.respond(user_input)
        response += response_from_chatbot if response_from_chatbot else "I'm not sure how to respond to that."

    return response

# Main function to run the chatbot
def run_chatbot():
    print("Hello! I'm Assistant. What's your name?")
    user_name = input("You: ")
    print(f"Nice to meet you, {user_name}!")
    print("How can I help you today?")
    while True:
        user_input = input(f"{user_name}: ").lower()
        if user_input == 'quit':
            print(get_chatbot_response(user_input, user_name))
            break
        else:
            print("Assistant:", get_chatbot_response(user_input, user_name))

if __name__ == "__main__":
    run_chatbot()
