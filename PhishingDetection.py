import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def clean_data(input_file='urlset.csv', output_file='cleaned_urlset.csv'):
    """
    Clean the input CSV file to handle decoding errors.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the cleaned output CSV file.

    Returns:
        None
    """
    clean_lines = []
    with open(input_file, 'rb') as file:
        for line in file:
            try:
                clean_lines.append(line.decode('utf-8'))
            except UnicodeDecodeError:
                pass

    with open(output_file, 'w', encoding='utf-8') as clean_file:
        clean_file.writelines(clean_lines)

def load_english_words(filename='words_alpha.txt'):
    """
    Load English words from a file.

    Parameters:
        filename (str): Path to the file containing English words.

    Returns:
        set: Set containing English words.
    """
    with open(filename, 'r') as file:
        english_words = set(word.strip().lower() for word in file)
    return english_words

def extract_domain(url):
    """
    Extract the main domain from a URL.

    Parameters:
        url (str): The URL to extract the domain from.

    Returns:
        str: The main domain extracted from the URL.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    domain = domain.replace('www.', '')
    domain_parts = domain.split('.')
    if len(domain_parts) > 2:
        main_domain = domain_parts[-2]
    elif len(domain_parts) == 2:
        main_domain = domain_parts[0]
    else:
        main_domain = domain
    return main_domain

def extract_features(url, english_words):
    """
    Extract features from a URL.

    Parameters:
        url (str): The URL to extract features from.
        english_words (set): Set of English words for spelling check.

    Returns:
        list: List of extracted features from the URL.
    """
    num_slashes = url.count('/')
    num_dots = url.count('.')
    length_of_url = len(url)
    main_domain = extract_domain(url)
    is_english_word = 1 if main_domain in english_words else 0
    
    path_parts = url.split('/')[3:]
    avg_word_length = sum(len(part) for part in path_parts) / (len(path_parts) + 1e-5)
    
    return [num_slashes, num_dots, length_of_url, is_english_word, avg_word_length]

# Clean the data
clean_data()

# Load English words for spelling check
english_words = load_english_words()

# Load the dataset
df = pd.read_csv('cleaned_urlset.csv')  

# Remove unnecessary columns
df = df[['domain', 'label']]

# Apply feature extraction to each URL
features = df['domain'].apply(lambda url: extract_features(url, english_words))

# Convert features and labels into a tensor
features_tensor = torch.tensor(features.tolist(), dtype=torch.float32)
labels_tensor = torch.tensor(df['label'].values, dtype=torch.float32)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features_tensor, labels_tensor, test_size=0.2, random_state=42)

# DataLoader
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Neural Network Model
class URLClassifier(nn.Module):
    def __init__(self):
        """
        Define the architecture of the URL classifier neural network.
        """
        super(URLClassifier, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        """
        Forward pass through the neural network.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = URLClassifier()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(10):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(torch.float32), labels.to(torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Make predictions on the test set
model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze()
    predictions = torch.round(predictions)

# Convert predictions and true labels to CPU numpy arrays for evaluation
predictions = predictions.cpu().numpy()
y_test_np = y_test.cpu().numpy()

# Calculate metrics
accuracy = accuracy_score(y_test_np, predictions)
precision = precision_score(y_test_np, predictions)
recall = recall_score(y_test_np, predictions)
f1 = f1_score(y_test_np, predictions)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')