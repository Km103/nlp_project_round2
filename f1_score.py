def calculatescore(manually_labeled_entities,entities):


    spacy_predicted_entities = [(ent.text, ent.label_) for ent in entities.ents]


    # Calculate evaluation metrics
    true_positives = set(spacy_predicted_entities) & set(manually_labeled_entities)
    false_positives = set(spacy_predicted_entities) - set(manually_labeled_entities)
    false_negatives = set(manually_labeled_entities) - set(spacy_predicted_entities)



    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0
    f1_score = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    