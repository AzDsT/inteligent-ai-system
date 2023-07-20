#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import requests


# In[ ]:


dataset = pd.read_csv("e_c_data.csv") 

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text) 
    text = re.sub(r"\d+", "", text)  
    text = re.sub(r"[^\w\s]", "", text)  
    text = text.lower()  
    text = " ".join(word for word in text.split() if word not in stopwords.words("english"))  
    return text

dataset["cleaned_text"] = dataset["text"].apply(clean_text)
dataset["tokens"] = dataset["cleaned_text"].apply(word_tokenize)
dataset.to_csv("preprocessed_dataset.csv", index=False)


# In[ ]:


class IntentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(IntentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.dropout(pooled_output)
        output = self.fc(output)
        return output
num_classes = 5  
model = IntentClassifier(num_classes)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "Can you help me track my order"
input_ids = tokenizer.encode(text, add_special_tokens=True)
attention_mask = [1] * len(input_ids)
input_ids = torch.tensor([input_ids])
attention_mask = torch.tensor([attention_mask])
outputs = model(input_ids, attention_mask)
predicted_intent = torch.argmax(outputs, dim=1)

print("Predicted Intent:", predicted_intent.item())


# In[ ]:


model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

while True:
    user_input = input("User: ")
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    with torch.no_grad():
        response = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    model_response = tokenizer.decode(response[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Assistant:", model_response)
    
    if user_input.lower() == 'exit':
        break


# In[ ]:


def send_inquiry(inquiry_text):
    response = requests.post("https://api.aiassistant.com/inquiry", json={"text": inquiry_text})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("AI assistant request failed. Status code: {}".format(response.status_code))

def test_ai_assistant():
    inquiries = [
        "I have a problem with my order.",
        "How can I track my package?",
        "What is your return policy?",
        "Can I change my shipping address?"
    ]
    for inquiry in inquiries:
        response = send_inquiry(inquiry)
        print("Customer Inquiry:", inquiry)
        print("AI Assistant Response:", response)
        print()
test_ai_assistant()


# In[ ]:


model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def fine_tune_model(train_data):
    train_texts = train_data['text'].tolist()
    train_labels = train_data['label'].tolist()
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                                  torch.tensor(train_encodings['attention_mask']),
                                                  torch.tensor(train_labels))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):  
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

def analyze_feedback(feedback_data):
    feedback_sentiments = analyze_sentiments(feedback_data['comments'])
    additional_labeled_data = create_labeled_data(feedback_data['comments'], feedback_sentiments)
    fine_tune_model(model, labeled_data + additional_labeled_data)
    
feedback_data = pd.read_csv("customer_feedback.csv") 
labeled_data = pd.read_csv("labeled_data.csv")  
analyze_feedback(feedback_data)


# In[ ]:


def deploy_ai_assistant():
    customer_service_system = connect_to_customer_service_system()
    ai_assistant = AIAssistant()
    customer_service_system.configure_assistant(ai_assistant)
    customer_service_system.deploy_assistant()

def train_customer_service_agents():
    training_data = load_training_data()
    for data in training_data:
        agent = CustomerServiceAgent(data)
        agent.train()

deploy_ai_assistant()
train_customer_service_agents()


# In[ ]:


def evaluate_performance():
    evaluation_data = pd.read_csv("evaluation_data.csv")  
    accuracy = calculate_accuracy(evaluation_data)
    customer_satisfaction = calculate_customer_satisfaction(evaluation_data)
    agent_workload_reduction = calculate_workload_reduction(evaluation_data)
    
    print("Accuracy:", accuracy)
    print("Customer Satisfaction:", customer_satisfaction)
    print("Agent Workload Reduction:", agent_workload_reduction)

def iterate_ai_assistant():
    feedback_data = pd.read_csv("feedback_data.csv")  
    areas_for_improvement = analyze_feedback(feedback_data)
    implement_improvements(areas_for_improvement)

evaluate_performance()
iterate_ai_assistant()


# In[ ]:




