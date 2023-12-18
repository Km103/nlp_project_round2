import spacy
from spacy import displacy
# Load the English language model
nlp = spacy.load("en_core_web_sm")

def paraEntities(text):

    para1 = nlp(text)
    return para1