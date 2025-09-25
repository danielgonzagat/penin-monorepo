"""
Neural Language Model - Advanced NLP model for PENIN system
Transformer-based model with fine-tuning capabilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoConfig,
        TrainingArguments, Trainer,
        BertModel, BertTokenizer,
        GPT2Model, GPT2Tokenizer
    )
    from config.config_manager import get_config
    from penin.logging.logger import get_logger
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    get_config = lambda key, default=None: default
    get_logger = lambda: None

logger = get_logger("ml_models") if get_logger("ml_models") else None

@dataclass
class ModelConfig:
    """Configuração do modelo"""
    model_type: str = "bert"
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_layers: int = 12
    dropout: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01

class TextDataset(Dataset):
    """Dataset para treinamento de modelos de linguagem"""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, 
                 tokenizer=None, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            item = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'text': text
            }
            
            if self.labels is not None:
                item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
            return item
        else:
            # Fallback without tokenizer
            return {
                'text': text,
                'label': torch.tensor(self.labels[idx] if self.labels else 0, dtype=torch.long)
            }

class NeuralLanguageModel(nn.Module):
    """
    Modelo de linguagem neural avançado com capacidades de:
    - Compreensão de texto
    - Geração de texto
    - Classificação
    - Análise de sentimento
    - Extração de entidades
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        if TRANSFORMERS_AVAILABLE:
            self._init_transformer_model()
        else:
            self._init_fallback_model()
        
        # Task-specific heads
        self.classification_head = nn.Linear(config.hidden_size, 10)  # 10 classes
        self.sentiment_head = nn.Linear(config.hidden_size, 3)  # pos/neg/neutral
        self.generation_head = nn.Linear(config.hidden_size, 30000)  # vocab size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        if logger:
            logger.info(f"Neural Language Model initialized with {config.model_type}")
    
    def _init_transformer_model(self):
        """Inicializa modelo transformer"""
        if self.config.model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(self.config.model_name)
            self.base_model = BertModel.from_pretrained(self.config.model_name)
        elif self.config.model_type == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
            self.base_model = GPT2Model.from_pretrained(self.config.model_name)
            # Add padding token for GPT-2
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            # Generic auto model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.base_model = AutoModel.from_pretrained(self.config.model_name)
    
    def _init_fallback_model(self):
        """Inicializa modelo simples quando transformers não está disponível"""
        self.embedding = nn.Embedding(30000, self.config.hidden_size)
        self.lstm = nn.LSTM(
            self.config.hidden_size, 
            self.config.hidden_size, 
            num_layers=2, 
            batch_first=True,
            dropout=self.config.dropout
        )
        self.tokenizer = None
        self.base_model = None
        
        if logger:
            logger.warning("Using fallback model - transformers not available")
    
    def forward(self, input_ids=None, attention_mask=None, task="classification"):
        """Forward pass com suporte a múltiplas tarefas"""
        
        if self.base_model:
            # Transformer-based forward pass
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                # BERT-style pooled output
                hidden_states = outputs.pooler_output
            else:
                # Use last hidden state and pool
                last_hidden = outputs.last_hidden_state
                hidden_states = torch.mean(last_hidden, dim=1)  # Average pooling
        else:
            # Fallback LSTM model
            embedded = self.embedding(input_ids)
            lstm_out, (hidden, _) = self.lstm(embedded)
            hidden_states = hidden[-1]  # Use last hidden state
        
        hidden_states = self.dropout(hidden_states)
        
        # Task-specific outputs
        if task == "classification":
            return self.classification_head(hidden_states)
        elif task == "sentiment":
            return self.sentiment_head(hidden_states)
        elif task == "generation":
            return self.generation_head(hidden_states)
        else:
            return hidden_states
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Codifica texto em representação vetorial"""
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                hidden_states = self.forward(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask'],
                    task="embedding"
                )
            
            return hidden_states
        else:
            # Simple fallback encoding
            words = text.split()[:self.config.max_length]
            # Convert to simple hash-based encoding
            word_ids = [hash(word) % 30000 for word in words]
            input_ids = torch.tensor([word_ids], dtype=torch.long)
            
            with torch.no_grad():
                hidden_states = self.forward(input_ids=input_ids, task="embedding")
            
            return hidden_states
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """Classifica texto"""
        encoding = self.encode_text(text)
        
        with torch.no_grad():
            if self.tokenizer:
                tokenized = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
                
                logits = self.forward(
                    input_ids=tokenized['input_ids'],
                    attention_mask=tokenized['attention_mask'],
                    task="classification"
                )
            else:
                words = text.split()[:self.config.max_length]
                word_ids = [hash(word) % 30000 for word in words]
                input_ids = torch.tensor([word_ids], dtype=torch.long)
                logits = self.forward(input_ids=input_ids, task="classification")
            
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
        
        return {
            'predicted_class': predicted_class.item(),
            'confidence': torch.max(probabilities).item(),
            'probabilities': probabilities.squeeze().tolist()
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analisa sentimento do texto"""
        with torch.no_grad():
            if self.tokenizer:
                tokenized = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
                
                logits = self.forward(
                    input_ids=tokenized['input_ids'],
                    attention_mask=tokenized['attention_mask'],
                    task="sentiment"
                )
            else:
                words = text.split()[:self.config.max_length]
                word_ids = [hash(word) % 30000 for word in words]
                input_ids = torch.tensor([word_ids], dtype=torch.long)
                logits = self.forward(input_ids=input_ids, task="sentiment")
            
            probabilities = torch.softmax(logits, dim=-1)
            sentiment_labels = ['negative', 'neutral', 'positive']
            predicted_sentiment = torch.argmax(probabilities, dim=-1)
        
        return {
            'sentiment': sentiment_labels[predicted_sentiment.item()],
            'confidence': torch.max(probabilities).item(),
            'scores': {
                label: prob.item() 
                for label, prob in zip(sentiment_labels, probabilities.squeeze())
            }
        }
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Gera texto baseado em prompt (implementação simplificada)"""
        # This is a simplified implementation
        # In a real scenario, you'd use proper text generation techniques
        
        if logger:
            logger.info(f"Generating text from prompt: {prompt[:50]}...")
        
        # Simple template-based generation for demonstration
        templates = [
            f"{prompt} This is an important consideration for the system.",
            f"Based on {prompt}, we can conclude that the analysis shows significant patterns.",
            f"The input '{prompt}' suggests that further investigation is needed.",
            f"Regarding {prompt}, the neural processing indicates multiple possibilities."
        ]
        
        # Select based on prompt hash for consistency
        template_idx = hash(prompt) % len(templates)
        generated = templates[template_idx]
        
        return generated[:max_length]
    
    def save_model(self, save_path: str):
        """Salva modelo"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), save_dir / "model.pt")
        
        # Save config
        with open(save_dir / "config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save tokenizer if available
        if self.tokenizer and hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(save_dir / "tokenizer")
        
        if logger:
            logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str):
        """Carrega modelo"""
        load_dir = Path(load_path)
        
        # Load config
        with open(load_dir / "config.json", 'r') as f:
            config_dict = json.load(f)
        
        config = ModelConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load state
        model.load_state_dict(torch.load(load_dir / "model.pt"))
        
        if logger:
            logger.info(f"Model loaded from {load_path}")
        
        return model

class ModelTrainer:
    """Treinador de modelos com suporte a fine-tuning"""
    
    def __init__(self, model: NeuralLanguageModel, config: ModelConfig):
        self.model = model
        self.config = config
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.training_history = []
    
    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Treina o modelo"""
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                if 'input_ids' in batch:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        task="classification"
                    )
                else:
                    # Fallback for simple text
                    continue  # Skip if no proper tokenization
                
                if 'label' in batch:
                    loss = self.criterion(outputs, batch['label'])
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            
            # Validation
            val_loss = 0.0
            if val_loader:
                val_loss = self._validate(val_loader)
            
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            self.training_history.append(epoch_metrics)
            
            if logger:
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                          f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return {
            'training_history': self.training_history,
            'final_train_loss': self.training_history[-1]['train_loss'] if self.training_history else 0.0,
            'final_val_loss': self.training_history[-1]['val_loss'] if self.training_history else 0.0
        }
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Valida o modelo"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if 'input_ids' in batch and 'label' in batch:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        task="classification"
                    )
                    loss = self.criterion(outputs, batch['label'])
                    total_loss += loss.item()
                    num_batches += 1
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    def evaluate(self, test_dataset: Dataset) -> Dict[str, Any]:
        """Avalia o modelo"""
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if 'input_ids' in batch and 'label' in batch:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        task="classification"
                    )
                    
                    loss = self.criterion(outputs, batch['label'])
                    total_loss += loss.item()
                    
                    predictions = torch.argmax(outputs, dim=-1)
                    correct_predictions += (predictions == batch['label']).sum().item()
                    total_predictions += batch['label'].size(0)
        
        accuracy = correct_predictions / max(total_predictions, 1)
        avg_loss = total_loss / len(test_loader)
        
        return {
            'test_loss': avg_loss,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }

# Factory functions
def create_language_model(model_type: str = "bert", **kwargs) -> NeuralLanguageModel:
    """Cria modelo de linguagem"""
    config = ModelConfig(model_type=model_type, **kwargs)
    return NeuralLanguageModel(config)

def create_trainer(model: NeuralLanguageModel, **kwargs) -> ModelTrainer:
    """Cria treinador de modelo"""
    config = ModelConfig(**kwargs)
    return ModelTrainer(model, config)

# Example usage and testing
if __name__ == "__main__":
    # Test the model
    print("Testing Neural Language Model...")
    
    # Create model
    model = create_language_model("bert" if TRANSFORMERS_AVAILABLE else "simple")
    
    # Test text classification
    test_text = "This is a great product! I love it."
    
    classification_result = model.classify_text(test_text)
    print("Classification result:", classification_result)
    
    sentiment_result = model.analyze_sentiment(test_text)
    print("Sentiment result:", sentiment_result)
    
    generated_text = model.generate_text("The future of AI is")
    print("Generated text:", generated_text)
    
    # Test encoding
    encoding = model.encode_text(test_text)
    print("Text encoding shape:", encoding.shape)
    
    print("Neural Language Model test completed!")