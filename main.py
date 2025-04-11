import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# Topic categories
TOPICS = ['动力', '价格', '内饰', '配置', '安全性', '外观', '操控', '油耗', '空间', '舒适性']

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Function to extract topics and sentiment values from text
def extract_labels_from_text(text):
    # Initialize an empty topic vector (one-hot encoding for 10 topics)
    topic_vector = [0] * len(TOPICS)

    # Initialize sentiment sum for the text
    sentiment_sum = 0

    # Find all occurrences of topic#sentiment_value pattern
    pattern = r'([^\s#]+)#(-?\d+)'
    matches = re.findall(pattern, text)

    for topic, sentiment_value in matches:
        sentiment_value = int(sentiment_value)
        sentiment_sum += sentiment_value

        # Find the index of the topic in our predefined list
        try:
            topic_idx = TOPICS.index(topic)
            topic_vector[topic_idx] = 1  # Mark the topic as present
        except ValueError:
            # If topic not in our predefined list, ignore it
            pass

    # Convert sentiment sum to sentiment class (0: neutral, 1: positive, -1: negative)
    if sentiment_sum > 0:
        sentiment_class = 1  # positive
    elif sentiment_sum < 0:
        sentiment_class = -1  # negative
    else:
        sentiment_class = 0  # neutral

    # Clean the text by removing the topic#sentiment patterns
    clean_text = re.sub(r'[^\s#]+#-?\d+', '', text).strip()

    return clean_text, sentiment_class, topic_vector


# Load the dataset
def load_data(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Check if the dataset has only one column (text)
    if len(df.columns) == 1:
        text_column = df.columns[0]
        df.rename(columns={text_column: 'text'}, inplace=True)

    return df


# Custom dataset
class AutoReviewDataset(Dataset):
    def __init__(self, texts, sentiment_labels, topic_labels, tokenizer, max_len=128):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.topic_labels = topic_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        sentiment_label = self.sentiment_labels[idx]
        topic_label = self.topic_labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment_label': torch.tensor(sentiment_label, dtype=torch.long),
            'topic_label': torch.tensor(topic_label, dtype=torch.float)
        }


# BERT for Sentiment Classification (Single-label)
class BERTSentimentClassifier(nn.Module):
    def __init__(self, bert_model_name="bert-base-chinese", num_classes=3):
        super(BERTSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits


# BERT for Topic Classification (Multi-label)
class BERTTopicClassifier(nn.Module):
    def __init__(self, bert_model_name="bert-base-chinese", num_labels=len(TOPICS)):
        super(BERTTopicClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return self.sigmoid(logits)


# Training function
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    predictions = []
    actual_labels = []

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        if isinstance(model, BERTSentimentClassifier):
            # Map sentiment labels from [-1, 0, 1] to [0, 1, 2]
            labels_mapped = batch['sentiment_label'].clone()
            labels_mapped[labels_mapped == -1] = 0
            labels_mapped[labels_mapped == 0] = 1
            labels_mapped[labels_mapped == 1] = 2
            labels = labels_mapped.to(device)
        else:  # BERTTopicClassifier
            labels = batch['topic_label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if isinstance(model, BERTSentimentClassifier):
            _, preds = torch.max(outputs, dim=1)
            # Map predictions back from [0, 1, 2] to [-1, 0, 1]
            preds_mapped = preds.clone()
            preds_mapped[preds == 0] = -1
            preds_mapped[preds == 1] = 0
            preds_mapped[preds == 2] = 1
            predictions.extend(preds_mapped.cpu().tolist())
            actual_labels.extend(batch['sentiment_label'].cpu().tolist())
        else:  # BERTTopicClassifier
            preds = (outputs > 0.5).float()
            predictions.extend(preds.detach().cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    if isinstance(model, BERTSentimentClassifier):
        accuracy = accuracy_score(actual_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_labels, predictions, average='macro', zero_division=0
        )
    else:  # BERTTopicClassifier
        # For multi-label, calculate accuracy as exact match ratio
        predictions = np.array(predictions)
        actual_labels = np.array(actual_labels)
        accuracy = np.mean(np.all(predictions == actual_labels, axis=1))

        # Calculate precision, recall, and F1 score per class and average
        precision, recall, f1 = 0, 0, 0
        for i in range(len(TOPICS)):
            class_precision = precision_score_binary(actual_labels[:, i], predictions[:, i])
            class_recall = recall_score_binary(actual_labels[:, i], predictions[:, i])
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall + 1e-8)

            precision += class_precision
            recall += class_recall
            f1 += class_f1

        precision /= len(TOPICS)
        recall /= len(TOPICS)
        f1 /= len(TOPICS)

    return avg_loss, accuracy, precision, recall, f1, predictions, actual_labels


# Evaluation function
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            if isinstance(model, BERTSentimentClassifier):
                # Map sentiment labels from [-1, 0, 1] to [0, 1, 2]
                labels_mapped = batch['sentiment_label'].clone()
                labels_mapped[labels_mapped == -1] = 0
                labels_mapped[labels_mapped == 0] = 1
                labels_mapped[labels_mapped == 1] = 2
                labels = labels_mapped.to(device)
            else:  # BERTTopicClassifier
                labels = batch['topic_label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            if isinstance(model, BERTSentimentClassifier):
                _, preds = torch.max(outputs, dim=1)
                # Map predictions back from [0, 1, 2] to [-1, 0, 1]
                preds_mapped = preds.clone()
                preds_mapped[preds == 0] = -1
                preds_mapped[preds == 1] = 0
                preds_mapped[preds == 2] = 1
                predictions.extend(preds_mapped.cpu().tolist())
                actual_labels.extend(batch['sentiment_label'].cpu().tolist())
            else:  # BERTTopicClassifier
                preds = (outputs > 0.5).float()
                predictions.extend(preds.detach().cpu().numpy())
                actual_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    if isinstance(model, BERTSentimentClassifier):
        accuracy = accuracy_score(actual_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_labels, predictions, average='macro', zero_division=0
        )
    else:  # BERTTopicClassifier
        # For multi-label, calculate accuracy as exact match ratio
        predictions = np.array(predictions)
        actual_labels = np.array(actual_labels)
        accuracy = np.mean(np.all(predictions == actual_labels, axis=1))

        # Calculate precision, recall, and F1 score per class and average
        precision, recall, f1 = 0, 0, 0
        for i in range(len(TOPICS)):
            class_precision = precision_score_binary(actual_labels[:, i], predictions[:, i])
            class_recall = recall_score_binary(actual_labels[:, i], predictions[:, i])
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall + 1e-8)

            precision += class_precision
            recall += class_recall
            f1 += class_f1

        precision /= len(TOPICS)
        recall /= len(TOPICS)
        f1 /= len(TOPICS)

    return avg_loss, accuracy, precision, recall, f1, predictions, actual_labels


# Helper functions for binary classification metrics
def precision_score_binary(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positive = np.sum(y_pred == 1)
    return true_positive / (predicted_positive + 1e-8)


def recall_score_binary(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    actual_positive = np.sum(y_true == 1)
    return true_positive / (actual_positive + 1e-8)


# Plot training and validation curves
def plot_curves(train_metrics, val_metrics, metric_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f'Training {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric_name.lower()}_curve.png')
    plt.show()


# Main function
def main():
    # Load data from the provided file path
    # Replace with the actual path from the QuarkDrive link
    file_path = "auto_reviews.csv"

    try:
        df = load_data(file_path)
        print(f"Loaded dataset with {len(df)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating a sample dataset for demonstration")

        # Create a sample dataset for demonstration
        sample_texts = [
            '一直92，偶尔出去了不了解当地油品加95（97）。5万公里从没遇到问题，省油，动力也充足，加95也没感觉有啥不同。油耗#1动力#1',
            '外观挺好看的，但是内饰太差了，塑料感很强，而且空间也不够大。外观#1内饰#-1空间#-1',
            '价格便宜，但是安全性能不太好，配置也一般。价格#1安全性#-1配置#0'
        ]
        df = pd.DataFrame({'text': sample_texts})

    # Extract labels from text
    print("Extracting labels from text...")
    clean_texts = []
    sentiment_labels = []
    topic_labels = []

    for text in tqdm(df['text'], desc="Processing"):
        clean_text, sentiment, topics = extract_labels_from_text(text)
        clean_texts.append(clean_text)
        sentiment_labels.append(sentiment)
        topic_labels.append(topics)

    # Create a new DataFrame with extracted labels
    processed_df = pd.DataFrame({
        'text': clean_texts,
        'sentiment': sentiment_labels,
        'topics': topic_labels
    })

    # Display some statistics
    sentiment_counts = processed_df['sentiment'].value_counts()
    print("\nSentiment Distribution:")
    print(f"Positive (1): {sentiment_counts.get(1, 0)}")
    print(f"Neutral (0): {sentiment_counts.get(0, 0)}")
    print(f"Negative (-1): {sentiment_counts.get(-1, 0)}")

    topic_distribution = np.sum(np.array(topic_labels), axis=0)
    print("\nTopic Distribution:")
    for i, topic in enumerate(TOPICS):
        print(f"{topic}: {topic_distribution[i]}")

    # Split into train and test sets
    X_train, X_test, y_sentiment_train, y_sentiment_test, y_topics_train, y_topics_test = train_test_split(
        processed_df['text'].tolist(),
        processed_df['sentiment'].tolist(),
        processed_df['topics'].tolist(),
        test_size=0.2,
        random_state=42
    )

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # Create datasets
    train_dataset = AutoReviewDataset(
        texts=X_train,
        sentiment_labels=y_sentiment_train,
        topic_labels=y_topics_train,
        tokenizer=tokenizer
    )

    test_dataset = AutoReviewDataset(
        texts=X_test,
        sentiment_labels=y_sentiment_test,
        topic_labels=y_topics_test,
        tokenizer=tokenizer
    )

    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize models
    sentiment_model = BERTSentimentClassifier(num_classes=3).to(device)
    topic_model = BERTTopicClassifier().to(device)

    # Define loss functions and optimizers
    sentiment_criterion = nn.CrossEntropyLoss()
    topic_criterion = nn.BCELoss()

    sentiment_optimizer = torch.optim.AdamW(sentiment_model.parameters(), lr=2e-5)
    topic_optimizer = torch.optim.AdamW(topic_model.parameters(), lr=2e-5)

    # Training parameters
    num_epochs = 5

    # Training and evaluation metrics
    sentiment_train_losses = []
    sentiment_train_accuracies = []
    sentiment_val_losses = []
    sentiment_val_accuracies = []

    topic_train_losses = []
    topic_train_accuracies = []
    topic_val_losses = []
    topic_val_accuracies = []

    # Train and evaluate sentiment model
    print("Training Sentiment Model...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1, _, _ = train_model(
            sentiment_model, train_loader, sentiment_optimizer, sentiment_criterion, device
        )
        sentiment_train_losses.append(train_loss)
        sentiment_train_accuracies.append(train_acc)

        # Evaluate
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate_model(
            sentiment_model, test_loader, sentiment_criterion, device
        )
        sentiment_val_losses.append(val_loss)
        sentiment_val_accuracies.append(val_acc)

        print(
            f"Sentiment - Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}")
        print(
            f"Sentiment - Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")

    # Train and evaluate topic model
    print("\nTraining Topic Model...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1, _, _ = train_model(
            topic_model, train_loader, topic_optimizer, topic_criterion, device
        )
        topic_train_losses.append(train_loss)
        topic_train_accuracies.append(train_acc)

        # Evaluate
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate_model(
            topic_model, test_loader, topic_criterion, device
        )
        topic_val_losses.append(val_loss)
        topic_val_accuracies.append(val_acc)

        print(
            f"Topic - Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}")
        print(
            f"Topic - Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")

    # Plot learning curves
    plot_curves(sentiment_train_losses, sentiment_val_losses, "Sentiment Loss")
    plot_curves(sentiment_train_accuracies, sentiment_val_accuracies, "Sentiment Accuracy")
    plot_curves(topic_train_losses, topic_val_losses, "Topic Loss")
    plot_curves(topic_train_accuracies, topic_val_accuracies, "Topic Accuracy")

    # Final evaluation on test set
    _, sentiment_acc, sentiment_prec, sentiment_rec, sentiment_f1, sentiment_preds, sentiment_labels = evaluate_model(
        sentiment_model, test_loader, sentiment_criterion, device
    )

    _, topic_acc, topic_prec, topic_rec, topic_f1, topic_preds, topic_labels = evaluate_model(
        topic_model, test_loader, topic_criterion, device
    )

    print("\nFinal Test Results:")
    print(
        f"Sentiment - Accuracy: {sentiment_acc:.4f}, Precision: {sentiment_prec:.4f}, Recall: {sentiment_rec:.4f}, F1: {sentiment_f1:.4f}")
    print(
        f"Topic - Accuracy: {topic_acc:.4f}, Precision: {topic_prec:.4f}, Recall: {topic_rec:.4f}, F1: {topic_f1:.4f}")

    # Save models
    torch.save(sentiment_model.state_dict(), 'sentiment_model.pth')
    torch.save(topic_model.state_dict(), 'topic_model.pth')

    print("Models saved successfully!")


# Function to process new text
def predict_sentiment_and_topics(text, sentiment_model, topic_model, tokenizer):
    sentiment_model.eval()
    topic_model.eval()

    # Extract labels from the text (for comparison with predictions)
    clean_text, true_sentiment, true_topics = extract_labels_from_text(text)

    # Tokenize
    encoding = tokenizer.encode_plus(
        clean_text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Predict sentiment
    with torch.no_grad():
        sentiment_outputs = sentiment_model(input_ids, attention_mask)
        _, sentiment_pred = torch.max(sentiment_outputs, dim=1)
        # Map predictions from [0, 1, 2] to [-1, 0, 1]
        sentiment_mapped = sentiment_pred.clone()
        sentiment_mapped[sentiment_pred == 0] = -1
        sentiment_mapped[sentiment_pred == 1] = 0
        sentiment_mapped[sentiment_pred == 2] = 1
        sentiment = sentiment_mapped.item()

        # Map sentiment
        sentiment_map = {0: "中立", 1: "正向", -1: "负向"}
        sentiment_label = sentiment_map[sentiment]

        # Predict topics
        topic_outputs = topic_model(input_ids, attention_mask)
        topic_preds = (topic_outputs > 0.5).float().cpu().numpy()[0]

        # Get topic labels
        topic_labels = [TOPICS[i] for i in range(len(TOPICS)) if topic_preds[i] == 1]

    # Compare with extracted true labels
    true_sentiment_label = sentiment_map[true_sentiment]
    true_topic_labels = [TOPICS[i] for i in range(len(TOPICS)) if true_topics[i] == 1]

    return {
        'text': text,
        'clean_text': clean_text,
        'predicted_sentiment': sentiment_label,
        'true_sentiment': true_sentiment_label,
        'predicted_topics': topic_labels,
        'true_topics': true_topic_labels
    }


if __name__ == "__main__":
    main()