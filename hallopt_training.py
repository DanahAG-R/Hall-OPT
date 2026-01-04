import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
from hallopt_model import HALLOPTTransformer, AdaptiveKnowledgeDistillation, EdgeOptimizationLayer


class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if 'question' in item:
            text = item['question'] + ' ' + item.get('context', '')
            label = 1 if 'answer' in item and item['answer'] else 0
        else:
            text = item.get('text', '')
            label = item.get('label', 0)
        
        tokens = self.tokenizer(text, max_length=self.max_length)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        if len(input_ids) < self.max_length:
            padding = torch.zeros(self.max_length - len(input_ids), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
        else:
            input_ids = input_ids[:self.max_length]
        
        return {'input_ids': input_ids, 'label': label}


class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3}
        self.idx = 4
    
    def build_vocab(self, texts, max_vocab=10000):
        word_freq = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:max_vocab - 4]:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx += 1
    
    def __call__(self, text, max_length=512):
        words = text.lower().split()
        tokens = []
        for word in words:
            if word in self.word2idx:
                tokens.append(self.word2idx[word])
            else:
                tokens.append(self.word2idx['<UNK>'])
        
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        return tokens


class HALLOPTTrainer:
    def __init__(self, teacher_model, student_model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        
        self.teacher_model.to(device)
        self.student_model.to(device)
        
        self.akd = AdaptiveKnowledgeDistillation(
            student_model.hidden_size, 
            student_model.num_layers
        ).to(device)
        
        self.optimizer = optim.Adam(
            list(self.student_model.parameters()) + 
            list(self.akd.parameters()),
            lr=3e-5
        )
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=100
        )
    
    def train_epoch(self, train_loader, lambda_task=1.0, lambda_hall=0.5, lambda_feat=0.3, 
                    temperature=4.0, target_retention=0.45):
        self.student_model.train()
        self.teacher_model.eval()
        
        total_loss = 0
        
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            
            with torch.no_grad():
                teacher_logits, teacher_all_hidden, _, _ = self.teacher_model(
                    input_ids, target_retention=target_retention, return_all_hiddens=True
                )
            
            student_logits, student_all_hidden, _, student_hall_scores = self.student_model(
                input_ids, target_retention=target_retention, return_all_hiddens=True
            )
            
            loss_distill = self.akd.compute_distillation_loss(
                teacher_logits, student_logits, temperature
            )
            
            loss_hall = self.akd.compute_hallucination_aware_loss(
                student_logits, teacher_logits, student_hall_scores
            )
            
            loss_feat = self.akd.compute_feature_distillation_loss(
                teacher_all_hidden[1:], student_all_hidden[1:]
            )
            
            cross_entropy = nn.CrossEntropyLoss()(
                student_logits.view(-1, student_logits.shape[-1]),
                labels.view(-1)
            )
            
            total_loss_batch = (
                loss_distill + 
                lambda_task * cross_entropy + 
                lambda_hall * loss_hall + 
                lambda_feat * loss_feat
            )
            
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        self.scheduler.step()
        return total_loss / len(train_loader)
    
    def evaluate(self, eval_loader, target_retention=0.45):
        self.student_model.eval()
        
        total_accuracy = 0
        total_samples = 0
        hall_accuracy = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                student_logits, hall_scores = self.student_model(
                    input_ids, target_retention=target_retention
                )
                
                predictions = student_logits.argmax(dim=-1)
                accuracy = (predictions.view(-1) == labels.view(-1)).float().mean()
                
                total_accuracy += accuracy.item() * input_ids.shape[0]
                total_samples += input_ids.shape[0]
                
                hall_pred = (hall_scores > 0.5).float()
                hall_accuracy += (hall_pred == labels.unsqueeze(-1)).float().mean().item()
        
        return total_accuracy / total_samples, hall_accuracy / len(eval_loader)


class HALLOPTInference:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
    
    def forward_with_pruning(self, text, latency_budget=50, target_retention=0.45):
        tokens = self.tokenizer(text, max_length=512)
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        if input_ids.shape[1] < 512:
            padding = torch.zeros(1, 512 - input_ids.shape[1], dtype=torch.long, device=self.device)
            input_ids = torch.cat([input_ids, padding], dim=1)
        
        with torch.no_grad():
            logits, hall_scores = self.model(input_ids, target_retention=target_retention)
        
        return logits, hall_scores
    
    def detect_hallucinations(self, text, threshold=0.5):
        logits, hall_scores = self.forward_with_pruning(text)
        
        hallucination_flags = (hall_scores > threshold).cpu().numpy()
        predictions = logits.argmax(dim=-1).cpu().numpy()
        
        return {
            'predictions': predictions,
            'hallucination_flags': hallucination_flags,
            'hallucination_scores': hall_scores.cpu().numpy()
        }
    
    def quantize_model(self, bit_width=8):
        eol = EdgeOptimizationLayer(self.model.hidden_size)
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                quantized = eol.quantize_weights(param.data, bit_width)
                param.data = quantized


class HALLOPTEvaluator:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
    
    def compute_f1_score(self, predictions, targets):
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        return f1, precision, recall
    
    def compute_hallucination_metrics(self, pred_hall, true_hall):
        accuracy = np.mean(pred_hall == true_hall)
        
        tp = np.sum((pred_hall == 1) & (true_hall == 1))
        fp = np.sum((pred_hall == 1) & (true_hall == 0))
        fn = np.sum((pred_hall == 0) & (true_hall == 1))
        tn = np.sum((pred_hall == 0) & (true_hall == 0))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        fpr = fp / (fp + tn + 1e-10)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr
        }
    
    def measure_latency(self, input_ids, num_runs=5):
        import time
        
        self.model.eval()
        times = []
        
        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            start = time.time()
            with torch.no_grad():
                _ = self.model(input_ids.to(self.device))
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            end = time.time()
            times.append((end - start) * 1000)
        
        return np.array(times)
    
    def measure_energy(self, model_params, operations_count, power_consumption=10):
        energy = (operations_count * power_consumption) / (10**9)
        return energy
    
    def compute_model_complexity(self, input_size, hidden_size, num_layers, seq_len):
        attention_flops = num_layers * 2 * seq_len * seq_len * hidden_size
        ffn_flops = num_layers * 2 * seq_len * hidden_size * (hidden_size * 4)
        embedding_flops = seq_len * hidden_size
        
        total_flops = attention_flops + ffn_flops + embedding_flops
        
        return total_flops
    
    def compute_memory_usage(self, batch_size, seq_len, hidden_size, num_layers):
        param_memory = (hidden_size * num_layers * 4) / (1024**2)
        activation_memory = (batch_size * seq_len * hidden_size * num_layers * 4) / (1024**2)
        kv_cache_memory = (2 * batch_size * seq_len * hidden_size * num_layers * 4) / (1024**2)
        
        total_memory = param_memory + activation_memory + kv_cache_memory
        
        return total_memory


def create_synthetic_qa_dataset(num_samples=1000):
    data = []
    
    for i in range(num_samples):
        data.append({
            'question': f'What is the answer to question {i}?',
            'context': f'This is context for question {i} with some information.',
            'answer': f'Answer {i % 10}',
            'label': i % 2
        })
    
    return data


def create_synthetic_summarization_dataset(num_samples=1000):
    data = []
    
    for i in range(num_samples):
        data.append({
            'text': f'This is a news article number {i}. It contains information about an event. The event happened on a specific date. ' * 5,
            'summary': f'News article {i} summary with key information.',
            'label': i % 2
        })
    
    return data


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vocab_size = 10000
    hidden_size = 512
    num_layers = 6
    num_heads = 8
    batch_size = 32
    num_epochs = 15
    
    teacher_model = HALLOPTTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    student_model = HALLOPTTransformer(
        vocab_size=vocab_size,
        hidden_size=256,
        num_layers=3,
        num_heads=4
    )
    
    tokenizer = SimpleTokenizer(vocab_size)
    
    train_data = create_synthetic_qa_dataset(2000)
    eval_data = create_synthetic_qa_dataset(500)
    
    texts = [d.get('question', '') + ' ' + d.get('context', '') for d in train_data]
    tokenizer.build_vocab(texts, max_vocab=vocab_size)
    
    train_dataset = QADataset(train_data, tokenizer)
    eval_dataset = QADataset(eval_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    trainer = HALLOPTTrainer(teacher_model, student_model, device=device)
    
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        eval_acc, hall_acc = trainer.evaluate(eval_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Eval Acc: {eval_acc:.4f} | Hall Acc: {hall_acc:.4f}")
    
    inference = HALLOPTInference(student_model, tokenizer, device=device)
    
    test_text = "What is the structure of DNA?"
    results = inference.detect_hallucinations(test_text, threshold=0.5)
    
    print(f"\nInference Results:")
    print(f"Predictions: {results['predictions']}")
    print(f"Hallucination Flags: {results['hallucination_flags']}")
    
    evaluator = HALLOPTEvaluator(student_model, device=device)
    
    test_input = torch.randint(0, vocab_size, (1, 512), device=device)
    latencies = evaluator.measure_latency(test_input, num_runs=5)
    
    print(f"\nLatency Statistics (ms):")
    print(f"Mean: {latencies.mean():.2f} | Std: {latencies.std():.2f}")
    print(f"Min: {latencies.min():.2f} | Max: {latencies.max():.2f}")
    
    flops = evaluator.compute_model_complexity(batch_size, 512, 3, 512)
    print(f"\nModel Complexity:")
    print(f"FLOPs: {flops/1e9:.2f}G")
    
    memory = evaluator.compute_memory_usage(batch_size, 512, 256, 3)
    print(f"Memory: {memory:.2f}MB")
    
    torch.save(student_model.state_dict(), 'hallopt_student.pt')
    torch.save(teacher_model.state_dict(), 'hallopt_teacher.pt')
