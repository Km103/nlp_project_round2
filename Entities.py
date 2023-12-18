import spacy

def generateEntities(text):

    # Load the English language model
    nlp = spacy.load("en_core_web_sm")


    # Process the text with spaCy
    book_ner = nlp(text)

    # Iterate through entities and print
    for ent in book_ner.ents:
        print(f"Entity: {ent.text}, Type: {ent.label_}")

