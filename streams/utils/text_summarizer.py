import os
import re
from transformers import pipeline

# Set environment variable to handle tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the summarization pipeline with a specific model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def clean_text(text):
    # Remove filler words and clean up the text
    filler_words = ['um', 'uh', 'like', 'you know', 'I mean']
    for word in filler_words:
        text = re.sub(r'\b' + word + r'\b', '', text, flags=re.IGNORECASE)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

def summarize_text(text, max_length=150):
    """
    Summarize the given text using the Hugging Face Transformers summarization pipeline.
    Args:
    text (str): The text to summarize.
    max_length (int): The maximum length of the summary.
    Returns:
    str: The summarized text.
    """
    # Clean the input text
    cleaned_text = clean_text(text)
    
    # Check if the cleaned text is long enough to summarize
    if len(cleaned_text.split()) < 20:
        return "Text too short for meaningful summarization."

    # Call the summarizer with adjusted parameters
    summary = summarizer(cleaned_text, 
                         max_length=max_length, 
                         min_length=min(30, len(cleaned_text.split()) // 2),  # Adjust min_length based on input
                         do_sample=False,
                         num_beams=4,  # Use beam search for potentially better results
                         early_stopping=True)
    
    summary_text = summary[0]['summary_text']
    
    # Post-process: ensure the summary is not just a repetition of the input
    if summary_text.lower() in cleaned_text.lower():
        return f"Summary generation failed. Input snippet: {cleaned_text[:50]}..."
    
    return summary_text

# Function to accumulate text over time
accumulated_text = ""
def process_and_summarize(new_text, force_summarize=False):
    global accumulated_text
    accumulated_text += new_text + " "
    
    # Only summarize if we have accumulated enough text or if forced
    if len(accumulated_text.split()) >= 100 or force_summarize:
        summary = summarize_text(accumulated_text, max_length=200)
        accumulated_text = ""  # Reset accumulated text
        return summary
    return None