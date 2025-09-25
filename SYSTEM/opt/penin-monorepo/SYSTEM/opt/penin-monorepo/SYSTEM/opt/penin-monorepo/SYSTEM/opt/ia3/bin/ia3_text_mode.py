#!/usr/bin/env python3
# /opt/ia3/bin/ia3_text_mode.py
# Modo texto para o sistema IA³ usando TinyStories e outros datasets

import os
import json
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import random

try:
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️ datasets/tokenizers não disponíveis, usando modo sintético")

class TextProcessor:
    """Processador de texto para tarefas de linguagem no sistema IA³"""
    
    def __init__(self, dataset_name="roneneldan/TinyStories", vocab_size=1000, seq_len=64):
        self.dataset_name = dataset_name
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.tokenizer = None
        self.corpus = []
        
        self._setup_dataset()
    
    def _setup_dataset(self):
        """Configura dataset e tokenizer"""
        if not DATASETS_AVAILABLE:
            self._create_synthetic_corpus()
            return
        
        try:
            # Carregar TinyStories
            print(f"📖 Carregando {self.dataset_name}...")
            dataset = load_dataset(self.dataset_name, split="train[:1%]")  # Pequena amostra para CPU
            
            # Extrair textos
            self.corpus = [item["text"] for item in dataset if "text" in item][:1000]  # Limite para CPU
            print(f"   ✅ {len(self.corpus)} histórias carregadas")
            
            # Treinar tokenizer BPE simples
            self._train_tokenizer()
            
        except Exception as e:
            print(f"⚠️ Erro ao carregar dataset: {e}")
            print("   Usando corpus sintético como fallback...")
            self._create_synthetic_corpus()
    
    def _create_synthetic_corpus(self):
        """Cria corpus sintético para teste"""
        templates = [
            "The {animal} {verb} {adverb} in the {place}.",
            "A {color} {object} {verb} near the {place}.",
            "The little {animal} {verb} with the {object}.",
            "Once upon a time, a {animal} lived in a {place}.",
            "The {color} {object} made the {animal} {verb}."
        ]
        
        words = {
            "animal": ["cat", "dog", "bird", "fish", "mouse", "rabbit", "bear", "fox"],
            "verb": ["runs", "jumps", "flies", "swims", "plays", "sleeps", "eats", "looks"],
            "adverb": ["quickly", "slowly", "happily", "sadly", "quietly", "loudly"],
            "place": ["forest", "house", "garden", "river", "mountain", "cave"],
            "color": ["red", "blue", "green", "yellow", "purple", "orange"],
            "object": ["ball", "book", "flower", "stone", "tree", "star"]
        }
        
        self.corpus = []
        for _ in range(500):  # 500 sentenças sintéticas
            template = random.choice(templates)
            sentence = template
            for key, values in words.items():
                sentence = sentence.replace(f"{{{key}}}", random.choice(values))
            self.corpus.append(sentence)
        
        print(f"   ✅ {len(self.corpus)} sentenças sintéticas geradas")
        self._create_simple_tokenizer()
    
    def _train_tokenizer(self):
        """Treina tokenizer BPE nos textos"""
        print("🔧 Treinando tokenizer BPE...")
        
        # Criar tokenizer BPE
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.BPE()
        
        # Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
        )
        
        # Treinar
        tokenizer.train_from_iterator(self.corpus, trainer=trainer)
        self.tokenizer = tokenizer
        
        print(f"   ✅ Tokenizer treinado com {len(self.tokenizer.get_vocab())} tokens")
    
    def _create_simple_tokenizer(self):
        """Cria tokenizer simples baseado em palavras para corpus sintético"""
        # Vocabulário baseado em palavras únicas
        all_words = set()
        for text in self.corpus:
            words = text.lower().replace('.', ' .').replace(',', ' ,').split()
            all_words.update(words)
        
        # Criar mapeamento
        vocab = ["<pad>", "<unk>", "<bos>", "<eos>"] + sorted(list(all_words))[:self.vocab_size-4]
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}
        self.vocab_size = len(vocab)
        
        print(f"   ✅ Tokenizer simples criado com {self.vocab_size} tokens")
    
    def encode_text(self, text: str) -> List[int]:
        """Codifica texto em tokens"""
        if self.tokenizer:
            # BPE tokenizer
            encoding = self.tokenizer.encode(text)
            return encoding.ids[:self.seq_len]
        else:
            # Tokenizer simples
            words = text.lower().replace('.', ' .').replace(',', ' ,').split()
            tokens = [self.word_to_id.get(word, self.word_to_id["<unk>"]) for word in words]
            return tokens[:self.seq_len]
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decodifica tokens em texto"""
        if self.tokenizer:
            return self.tokenizer.decode(token_ids)
        else:
            words = [self.id_to_word.get(id, "<unk>") for id in token_ids]
            return " ".join(words)
    
    def create_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cria batch de entrada e alvo para treino de linguagem"""
        inputs = []
        targets = []
        
        for _ in range(batch_size):
            # Escolher texto aleatório
            text = random.choice(self.corpus)
            
            # Tokenizar
            tokens = self.encode_text(text)
            
            # Pad se necessário
            if len(tokens) < self.seq_len + 1:
                tokens = tokens + [self.word_to_id.get("<pad>", 0)] * (self.seq_len + 1 - len(tokens))
            
            # Input: tokens[:-1], Target: tokens[1:]
            input_seq = tokens[:self.seq_len]
            target_seq = tokens[1:self.seq_len + 1]
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
        return torch.tensor(inputs), torch.tensor(targets)
    
    def create_ood_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cria batch OOD para teste de adaptação"""
        # Para OOD, misturar textos ou criar perturbações
        inputs = []
        targets = []
        
        for _ in range(batch_size):
            # Misturar dois textos diferentes
            text1 = random.choice(self.corpus)
            text2 = random.choice(self.corpus)
            
            # Criar híbrido (primeiras palavras de text1 + últimas de text2)
            words1 = text1.split()[:self.seq_len//2]
            words2 = text2.split()[self.seq_len//2:]
            
            hybrid_text = " ".join(words1 + words2)
            tokens = self.encode_text(hybrid_text)
            
            # Pad se necessário
            if len(tokens) < self.seq_len + 1:
                tokens = tokens + [self.word_to_id.get("<pad>", 0)] * (self.seq_len + 1 - len(tokens))
            
            input_seq = tokens[:self.seq_len]
            target_seq = tokens[1:self.seq_len + 1]
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
        return torch.tensor(inputs), torch.tensor(targets)

# Instância global
TEXT_PROCESSOR = None

def initialize_text_mode(dataset_name="roneneldan/TinyStories", vocab_size=1000, seq_len=64):
    """Inicializa modo texto globalmente"""
    global TEXT_PROCESSOR
    TEXT_PROCESSOR = TextProcessor(dataset_name, vocab_size, seq_len)
    return TEXT_PROCESSOR

def text_batch(batch_size=32, ood=False):
    """Interface compatível com o sistema principal"""
    global TEXT_PROCESSOR
    
    if TEXT_PROCESSOR is None:
        TEXT_PROCESSOR = initialize_text_mode()
    
    if ood:
        return TEXT_PROCESSOR.create_ood_batch(batch_size)
    else:
        return TEXT_PROCESSOR.create_batch(batch_size)

def get_vocab_size():
    """Retorna tamanho do vocabulário"""
    global TEXT_PROCESSOR
    if TEXT_PROCESSOR is None:
        TEXT_PROCESSOR = initialize_text_mode()
    return TEXT_PROCESSOR.vocab_size

if __name__ == "__main__":
    # Teste do processador de texto
    print("🧪 Testando processador de texto IA³...")
    
    processor = initialize_text_mode()
    
    # Teste de batch
    inputs, targets = processor.create_batch(4)
    print(f"\nBatch shape: inputs={inputs.shape}, targets={targets.shape}")
    
    # Decode exemplo
    for i in range(2):
        input_text = processor.decode_tokens(inputs[i].tolist())
        target_text = processor.decode_tokens(targets[i].tolist())
        print(f"\nExemplo {i+1}:")
        print(f"  Input:  {input_text}")
        print(f"  Target: {target_text}")
    
    # Teste OOD
    ood_inputs, ood_targets = processor.create_ood_batch(2)
    print(f"\nOOD batch shape: {ood_inputs.shape}")
    
    print("\n✅ Processador de texto funcionando!")