import streamlit as st
import subprocess  # Added import statement
from transformers import BertTokenizer, BertForTokenClassification
import torch
import spacy
from spacy import displacy

def install_requirements():
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

def recognize_entities(text_input):
    model_name = "dslim/bert-base-NER"
    model = BertForTokenClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    inputs = tokenizer(text_input, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

    entities = []
    for i, token in enumerate(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])):
        if predictions[0][i].item() != 0:  # Ignore 'O' (Outside) tokens
            entities.append((token, model.config.id2label[predictions[0][i].item()]))

    return entities

def main():
    st.title("BERT-based Named Entity Recognition (NER) App")

    text_input = st.text_area("Enter a sentence:")

    if st.button("Recognize Entities"):
        entities = recognize_entities(text_input)

        # Create a spacy Doc for visualization
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text_input)
        
        # Highlight recognized entities
        for ent in entities:
            token_start = text_input.find(ent[0])
            token_end = token_start + len(ent[0])
            doc.ents += (spacy.tokens.Span(doc, token_start, token_end, label=ent[1]),)
        
        # Display the sentence with entity highlighting
        html_output = displacy.render(doc, style="ent", page=True)
        st.write(html_output, unsafe_allow_html=True)

if __name__ == "__main__":
    install_requirements()  # Call install_requirements before running the Streamlit app
    main()
