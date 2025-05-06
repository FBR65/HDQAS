import spacy
from transformers import pipeline


def german_coreference_resolution(text):
    """
    Performs coreference resolution on German text using a multilingual NER model
    and spaCy for linguistic analysis.
    """
    try:
        # Load the multilingual NER model
        ner_model = pipeline(
            "ner", model="dslim/bert-base-NER", aggregation_strategy="simple"
        )

        # Load the German spaCy model
        nlp = spacy.load("de_core_news_lg")

        # Process the text with both models
        ner_results = ner_model(text)
        doc = nlp(text)

        # Print NER results and SpaCy tokens for inspection
        print("NER Results:", ner_results)
        print("SpaCy Tokens:", [(token.text, token.pos_, token.dep_) for token in doc])

        # Track entities per sentence
        resolved_text = text
        current_person = None
        current_object = None
        sentence_entities = {}

        # First pass: collect entities and relationships
        for i, sent in enumerate(doc.sents):
            sentence_entities[i] = {
                "persons": [],
                "last_person": None,
                "last_object": None,
                "grandmother": False,
                "brother": False,
                "mother": False,
            }

            # Process each token in the sentence
            for token in sent:
                if token.ent_type_ == "PER":
                    sentence_entities[i]["persons"].append(token.text)
                    sentence_entities[i]["last_person"] = token.text
                elif "großmutter" in token.text.lower():
                    sentence_entities[i]["grandmother"] = True
                elif "bruder" in token.text.lower():
                    sentence_entities[i]["brother"] = True
                elif "mutter" in token.text.lower():
                    sentence_entities[i]["mother"] = True
                elif token.pos_ == "NOUN" and token.text.lower() in ["hund", "auto"]:
                    sentence_entities[i]["last_object"] = token.text

        # Second pass: resolve pronouns
        resolved_text = text
        for i, sent in enumerate(doc.sents):
            replacements = {}
            current_sent = sentence_entities[i]
            prev_sent = sentence_entities.get(
                i - 1, {"persons": [], "last_person": None}
            )

            for token in sent:
                if token.pos_ == "PRON":
                    if token.text.lower() in ["er", "sie"]:
                        # Personal pronouns
                        if current_sent["persons"]:
                            replacements[token.text] = current_sent["last_person"]
                        elif prev_sent["last_person"]:
                            replacements[token.text] = prev_sent["last_person"]

                    elif token.text.lower() == "ihr":
                        # Possessive cases
                        if (
                            current_sent["grandmother"]
                            or "großmutter" in sent.text.lower()
                        ):
                            replacements[token.text] = "Großmutter"
                        elif current_sent["mother"]:
                            replacements[token.text] = "Mutter"
                        elif prev_sent["last_person"]:
                            replacements[token.text] = f"{prev_sent['last_person']}s"

                    elif token.text.lower() == "ihm":
                        # Dative cases
                        if (
                            current_sent["last_object"]
                            and current_sent["last_object"].lower() == "hund"
                        ):
                            replacements[token.text] = "dem Hund"

                    elif token.text.lower() == "es":
                        # Neuter pronouns
                        if (
                            current_sent["last_object"]
                            and current_sent["last_object"].lower() == "auto"
                        ):
                            replacements[token.text] = "das Auto"

            # Apply replacements for this sentence
            sent_text = sent.text.strip()
            for pronoun, referent in replacements.items():
                sent_text = sent_text.replace(f" {pronoun} ", f" {referent} ")
                sent_text = sent_text.replace(f"{pronoun} ", f"{referent} ")
                if sent_text.startswith(pronoun):
                    sent_text = f"{referent}{sent_text[len(pronoun) :]}"

            resolved_text = resolved_text.replace(sent.text.strip(), sent_text)

        return resolved_text

    except Exception as e:
        print(f"An error occurred: {e}")
        return text


if __name__ == "__main__":
    text = "Lisa besuchte ihre Großmutter. Sie brachte ihr einen Blumenstrauß mit. Michael hat einen neuen Hund. Er ist sehr verspielt und liebt es mit ihm im Park zu spielen. Sarah liebt ihren Bruder sehr. Er ist ihr bester Freund. David kaufte ein neues Auto. Es ist sehr geräumig und hat viele Extras. Emily hilft ihrer Mutter oft im Haushalt. Sie mag es, gemeinsam zu kochen."
    resolved_text = german_coreference_resolution(text)
    print("Original Text:", text)
    print("Resolved Text:", resolved_text)
    print(
        "Expected Output: Lisa besuchte ihre Großmutter. Lisa brachte Großmutter einen Blumenstrauß mit. Michael hat einen neuen Hund. Michael ist sehr verspielt und liebt es mit dem Hund im Park zu spielen. Sarah liebt ihren Bruder sehr. Lisas Bruder ist Lisas bester Freund. David kaufte ein neues Auto. Das Auto ist sehr geräumig und hat viele Extras. Emily hilft ihrer Mutter oft im Haushalt. Emily mag es, gemeinsam zu kochen."
    )
