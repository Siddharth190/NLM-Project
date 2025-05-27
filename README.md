import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import string
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class NewsDataset(Dataset):
    """Custom Dataset class for news articles"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class NewsClassifier:
    """Main class for news article categorization"""
    
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_to_id = {}
        self.id_to_label = {}
        self.tfidf_vectorizer = None
        self.logistic_model = None
        
    def preprocess_text(self, text):
        """Preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub('<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def load_sample_data(self):
        """Create sample data for demonstration"""
        sample_data = [
            ("The stock market reached new highs today as technology companies posted strong earnings", "business"),
            ("The championship game was decided in overtime with a spectacular goal", "sport"),
            ("New artificial intelligence breakthrough promises to revolutionize healthcare", "tech"),
            ("Political leaders gathered for emergency summit on climate change", "politics"),
            ("Celebrity couple announces engagement at film premiere", "entertainment"),
            ("GDP growth exceeded expectations in the third quarter", "business"),
            ("Olympic athlete breaks world record in swimming event", "sport"),
            ("Smartphone manufacturer unveils latest flagship device", "tech"),
            ("Election results show tight race in key battleground states", "politics"),
            ("Box office numbers reveal summer blockbuster success", "entertainment"),
            ("Federal Reserve announces interest rate decision", "business"),
            ("Tennis tournament sees upset victory in finals", "sport"),
            ("Space agency launches mission to Mars", "tech"),
            ("Healthcare reform bill passes first reading", "politics"),
            ("Streaming service announces new original series", "entertainment")
        ]
        
        texts, labels = zip(*sample_data)
        return list(texts), list(labels)
    
    def prepare_data(self, texts, labels):
        """Prepare and preprocess data"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create label mappings
        unique_labels = sorted(list(set(labels)))
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        
        # Convert labels to ids
        label_ids = [self.label_to_id[label] for label in labels]
        
        return processed_texts, label_ids
    
    def train_baseline_model(self, texts, labels):
        """Train baseline logistic regression model"""
        print("Training baseline model (TF-IDF + Logistic Regression)...")
        
        # Create TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        X = self.tfidf_vectorizer.fit_transform(texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train logistic regression
        self.logistic_model = LogisticRegression(random_state=42, max_iter=1000)
        self.logistic_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.logistic_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        print(f"Baseline Model Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return accuracy, precision, recall, f1
    
    def train_transformer_model(self, texts, labels, epochs=3, batch_size=16):
        """Train transformer-based model"""
        print(f"Training transformer model ({self.model_name})...")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        num_labels = len(self.label_to_id)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = NewsDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = NewsDataset(val_texts, val_labels, self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        
        # Train model
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        
        # Manual evaluation for additional metrics
        predictions = trainer.predict(val_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = val_labels
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        print(f"Transformer Model Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=list(self.id_to_label.values())))
        
        return accuracy, precision, recall, f1
    
    def predict_baseline(self, text):
        """Predict using baseline model"""
        if self.tfidf_vectorizer is None or self.logistic_model is None:
            return "Model not trained"
        
        processed_text = self.preprocess_text(text)
        features = self.tfidf_vectorizer.transform([processed_text])
        prediction = self.logistic_model.predict(features)[0]
        confidence = self.logistic_model.predict_proba(features)[0].max()
        
        return self.id_to_label[prediction], confidence
    
    def predict_transformer(self, text):
        """Predict using transformer model"""
        if self.model is None or self.tokenizer is None:
            return "Model not trained"
        
        processed_text = self.preprocess_text(text)
        
        # Tokenize and encode
        inputs = self.tokenizer(
            processed_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = predictions.argmax().item()
            confidence = predictions[0][predicted_class_id].item()
        
        return self.id_to_label[predicted_class_id], confidence

def create_gradio_interface(classifier):
    """Create Gradio web interface"""
    
    def predict_category(text, model_type):
        """Prediction function for Gradio"""
        if not text.strip():
            return "Please enter some text", 0.0
        
        try:
            if model_type == "Transformer (BERT)":
                category, confidence = classifier.predict_transformer(text)
            else:  # Baseline
                category, confidence = classifier.predict_baseline(text)
            
            return f"Category: {category.upper()}", f"Confidence: {confidence:.2%}"
        except Exception as e:
            return f"Error: {str(e)}", "0%"
    
    # Create interface
    interface = gr.Interface(
        fn=predict_category,
        inputs=[
            gr.Textbox(
                label="News Article Text",
                placeholder="Enter news article text here...",
                lines=5
            ),
            gr.Radio(
                choices=["Transformer (BERT)", "Baseline (TF-IDF + LogReg)"],
                label="Model Type",
                value="Transformer (BERT)"
            )
        ],
        outputs=[
            gr.Textbox(label="Predicted Category"),
            gr.Textbox(label="Confidence Score")
        ],
        title="📰 News Article Categorization System",
        description="Enter a news article text to classify it into categories: Business, Sports, Technology, Politics, or Entertainment",
        examples=[
            ["The Federal Reserve announced a new interest rate policy that will affect mortgage rates nationwide", "Transformer (BERT)"],
            ["The championship match ended with a stunning victory in the final minutes", "Transformer (BERT)"],
            ["Scientists have developed a revolutionary AI system that can predict weather patterns", "Transformer (BERT)"],
            ["The election results showed a close race between the leading candidates", "Transformer (BERT)"],
            ["The blockbuster movie broke box office records in its opening weekend", "Transformer (BERT)"]
        ]
    )
    
    return interface

def main():
    """Main function to run the complete pipeline"""
    print("🚀 Starting News Article Categorization System")
    print("=" * 50)
    
    # Initialize classifier
    classifier = NewsClassifier(model_name='bert-base-uncased')
    
    # Load sample data
    print("📊 Loading sample data...")
    texts, labels = classifier.load_sample_data()
    
    # Prepare data
    print("🔧 Preprocessing data...")
    processed_texts, processed_labels = classifier.prepare_data(texts, labels)
    
    print(f"Dataset info:")
    print(f"- Total samples: {len(processed_texts)}")
    print(f"- Categories: {list(classifier.label_to_id.keys())}")
    print(f"- Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Train baseline model
    print("\n" + "=" * 50)
    classifier.train_baseline_model(processed_texts, processed_labels)
    
    # Train transformer model
    print("\n" + "=" * 50)
    classifier.train_transformer_model(processed_texts, processed_labels, epochs=2, batch_size=8)
    
    # Create and launch Gradio interface
    print("\n" + "=" * 50)
    print("🌐 Launching web interface...")
    
    interface = create_gradio_interface(classifier)
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )

if __name__ == "__main__":
    main()
