from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

def summarize_text(text, max_length=150):
    """
    Summarize the given text using the Hugging Face Transformers summarization pipeline.
    
    Args:
    text (str): The text to summarize.
    max_length (int): The maximum length of the summary.
    
    Returns:
    str: The summarized text.
    """
    # Call the summarizer with the provided max_length
    summary = summarizer(text, max_length=max_length, min_length=5, do_sample=False)
    return summary[0]['summary_text']
