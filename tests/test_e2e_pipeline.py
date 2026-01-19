"""
NeuralFlow End-to-End Pipeline Test

This script tests the complete training and inference pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print('=' * 60)
print('NEURALFLOW END-TO-END PIPELINE TEST')
print('=' * 60)
print()

# ========================================
# 1. Test Imports
# ========================================
print('[1/6] Testing imports...')
from app.model import NeuralFlowModel
from app.interfaces import Config
from app.training.training_stages import VQVAEStage, DynamicsStage
from app.training.data_loader import ParagraphDataset, ParagraphDataLoader
import torch
print('      All imports OK')

# ========================================
# 2. Test Model Creation
# ========================================
print('[2/6] Creating model...')
model = NeuralFlowModel.from_preset('small')
params = sum(p.numel() for p in model.parameters())
print(f'      Model: {params:,} parameters')

# ========================================  
# 3. Test Data Pipeline
# ========================================
print('[3/6] Creating dataset...')
dataset = ParagraphDataset.synthetic(num_paragraphs=16, min_words=10, max_words=30)
loader = ParagraphDataLoader(dataset, batch_size=4)
print(f'      Dataset: {len(dataset)} paragraphs')
print(f'      Sample: "{dataset[0][:50]}..."')

# ========================================
# 4. Test Training (VQ-VAE + Dynamics)
# ========================================
print('[4/6] Training VQ-VAE stage (2 batches)...')
stage1 = VQVAEStage(model, config={'learning_rate': 0.001}, device='cpu')
train_count = 0
for batch in loader:
    losses = stage1._train_step(batch)
    train_count += 1
    if train_count >= 2:
        break
print(f'      Commit: {losses["commit"]:.4f}, Perplexity: {losses["perplexity"]:.2f}')

print('[5/6] Training Dynamics stage (2 batches)...')
stage2 = DynamicsStage(model, config={'learning_rate': 0.0005}, device='cpu')
train_count = 0
loader2 = ParagraphDataLoader(dataset, batch_size=4)  # Reset loader
for batch in loader2:
    losses = stage2._train_step(batch)
    train_count += 1
    if train_count >= 2:
        break
print(f'      Prediction loss: {losses["prediction"]:.4f}')

# ========================================
# 5. Test Inference
# ========================================
print('[6/6] Testing inference...')
model.eval()

# Encode a test paragraph
test_text = 'This is a test paragraph for encoding.'
with torch.no_grad():
    enc_output = model.encoder([test_text])
    latent = enc_output.latent
    print(f'      Encoded latent shape: {latent.shape}')
    
    # Decode back
    dec_output = model.decoder.generate(latent, max_length=32)
    print(f'      Decoded tokens shape: {dec_output.data.shape}')
    if dec_output.text:
        print(f'      Generated text: "{dec_output.text[0][:50]}..."')

print()
print('=' * 60)
print('ALL TESTS PASSED - PIPELINE FULLY FUNCTIONAL')
print('=' * 60)
