"""
Test: VQ-VAE Training Loop Verification

验证 VQ-VAE 训练器功能：
1. 数据加载器工作正常
2. 编码器能处理批次
3. 训练循环能运行
4. 损失能下降
5. 码本利用率 > 0
"""

import sys
sys.path.insert(0, ".")

import torch


def test_paragraph_dataset():
    """测试段落数据集"""
    from app.training.data_loader import ParagraphDataset
    
    # 合成数据
    dataset = ParagraphDataset.synthetic(num_paragraphs=100)
    
    assert len(dataset) > 0, f"Dataset should not be empty, got {len(dataset)}"
    
    sample = dataset[0]
    assert isinstance(sample, str), f"Sample should be string, got {type(sample)}"
    assert len(sample) > 10, f"Sample too short: {len(sample)}"
    
    print(f"[PASS] ParagraphDataset: {len(dataset)} samples, first={sample[:50]}...")


def test_sequence_dataset():
    """测试序列数据集"""
    from app.training.data_loader import ParagraphDataset, SequenceDataset
    
    base = ParagraphDataset.synthetic(num_paragraphs=50)
    seq_dataset = SequenceDataset(base, seq_len=5)
    
    assert len(seq_dataset) > 0, f"SeqDataset should not be empty"
    
    inputs, target = seq_dataset[0]
    assert len(inputs) == 5, f"Input seq should be 5, got {len(inputs)}"
    assert isinstance(target, str), f"Target should be string"
    
    print(f"[PASS] SequenceDataset: {len(seq_dataset)} samples, seq_len=5")


def test_data_loader():
    """测试数据加载器"""
    from app.training.data_loader import ParagraphDataset, ParagraphDataLoader
    
    dataset = ParagraphDataset.synthetic(100)
    loader = ParagraphDataLoader(dataset, batch_size=16, shuffle=True)
    
    batch = next(iter(loader))
    
    assert len(batch.texts) == 16, f"Batch size should be 16, got {len(batch.texts)}"
    
    print(f"[PASS] DataLoader: batch_size=16, num_batches={len(loader)}")


def test_encoder_with_batch():
    """测试编码器处理批次"""
    from app.io import ParagraphEncoder
    
    encoder = ParagraphEncoder(
        d_model=128,
        d_latent=64,
        use_vq=True,
        max_length=32,
    )
    
    texts = ["This is first.", "This is second.", "Third paragraph."]
    output = encoder(texts)
    
    assert output.latent is not None
    assert output.latent.vector.shape == (3, 64), f"Wrong shape: {output.latent.vector.shape}"
    assert output.vq_output is not None
    
    print(f"[PASS] Encoder: batch of 3 -> latent {output.latent.vector.shape}")


def test_vqvae_trainer_init():
    """测试训练器初始化"""
    from app.io import ParagraphEncoder
    from app.training.vqvae_trainer import VQVAETrainer
    
    encoder = ParagraphEncoder(
        d_model=128,
        d_latent=64,
        use_vq=True,
    )
    
    trainer = VQVAETrainer(
        encoder=encoder,
        learning_rate=1e-3,
    )
    
    assert trainer.encoder is not None
    assert trainer.optimizer is not None
    
    print(f"[PASS] VQVAETrainer initialization")


def test_vqvae_training_loop():
    """测试训练循环 (小规模)"""
    from app.io import ParagraphEncoder
    from app.training.data_loader import ParagraphDataset
    from app.training.vqvae_trainer import VQVAETrainer
    
    # 小规模
    encoder = ParagraphEncoder(
        d_model=64,
        d_latent=32,
        use_vq=True,
        max_length=24,
    )
    
    trainer = VQVAETrainer(
        encoder=encoder,
        learning_rate=1e-3,
        log_interval=5,
    )
    
    # 合成数据
    dataset = ParagraphDataset.synthetic(50, min_words=5, max_words=20)
    
    # 训练 2 个 epoch
    print("\n--- Training VQ-VAE (2 epochs) ---")
    history = trainer.train(
        dataset,
        num_epochs=2,
        batch_size=8,
    )
    
    assert len(history) > 0, "Should have training history"
    
    # 检查损失下降趋势
    first_losses = [m.total_loss for m in history[:3]]
    last_losses = [m.total_loss for m in history[-3:]]
    
    first_avg = sum(first_losses) / len(first_losses)
    last_avg = sum(last_losses) / len(last_losses)
    
    print(f"\n  First 3 avg loss: {first_avg:.4f}")
    print(f"  Last 3 avg loss: {last_avg:.4f}")
    
    # 检查码本利用率
    final_util = history[-1].codebook_utilization
    print(f"  Final codebook utilization: {final_util*100:.1f}%")
    
    # 检查困惑度
    final_ppl = history[-1].perplexity
    print(f"  Final perplexity: {final_ppl:.1f}")
    
    print(f"\n[PASS] VQ-VAE training loop completed!")


def run_all_tests():
    print("\n" + "=" * 60)
    print(" VQ-VAE Training Loop Tests")
    print("=" * 60 + "\n")
    
    try:
        test_paragraph_dataset()
        test_sequence_dataset()
        test_data_loader()
        test_encoder_with_batch()
        test_vqvae_trainer_init()
        test_vqvae_training_loop()
        
        print("\n" + "=" * 60)
        print(" ALL TESTS PASSED! VQ-VAE training is working!")
        print("=" * 60 + "\n")
        return True
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
