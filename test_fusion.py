"""
Context Fusion 模块测试脚本

验收测试:
1. 基础功能 - 输出形状验证
2. 情感注入验证 - Neutral vs Angry 差异
3. 场景注入验证 - Chat vs Coding 差异
4. 门控参数验证 - 确认可学习性
5. 批量处理测试
"""

import torch
from app.memory import (
    ContextAwareEmbedding,
    ContextFusion,
    EmotionEmbedding,
    SceneEmbedding,
    RMSNorm,
    create_context_aware_embedding,
    EMOTION_IDS,
    SCENE_IDS,
)


def test_emotion_embedding():
    """测试情感嵌入层"""
    print("=" * 60)
    print("1. EmotionEmbedding Test")
    print("=" * 60)
    
    d_model = 256
    emotion_emb = EmotionEmbedding(d_model=d_model)
    
    # 测试单个情感
    emotion_id = torch.tensor([EMOTION_IDS["angry"]])
    vector = emotion_emb(emotion_id)
    
    print(f"  Input: emotion_id = {emotion_id.item()} (Angry)")
    print(f"  Output shape: {tuple(vector.shape)}")
    print(f"  Expected shape: (1, {d_model})")
    assert tuple(vector.shape) == (1, d_model), "Shape mismatch!"
    
    # 测试批量
    batch_ids = torch.tensor([0, 1, 2, 3])  # Neutral, Happy, Sad, Angry
    batch_vectors = emotion_emb(batch_ids)
    print(f"  Batch input: {batch_ids.tolist()}")
    print(f"  Batch output shape: {tuple(batch_vectors.shape)}")
    assert tuple(batch_vectors.shape) == (4, d_model)
    
    print("  [OK] EmotionEmbedding test passed")
    print()


def test_scene_embedding():
    """测试场景嵌入层"""
    print("=" * 60)
    print("2. SceneEmbedding Test")
    print("=" * 60)
    
    d_model = 256
    scene_emb = SceneEmbedding(d_model=d_model)
    
    # 测试单个场景
    scene_id = torch.tensor([SCENE_IDS["coding"]])
    vector = scene_emb(scene_id)
    
    print(f"  Input: scene_id = {scene_id.item()} (Coding)")
    print(f"  Output shape: {tuple(vector.shape)}")
    assert tuple(vector.shape) == (1, d_model)
    
    print("  [OK] SceneEmbedding test passed")
    print()


def test_context_fusion():
    """测试门控融合层"""
    print("=" * 60)
    print("3. ContextFusion Gating Test")
    print("=" * 60)
    
    d_model = 256
    batch_size = 2
    seq_len = 10
    
    fusion = ContextFusion(d_model=d_model, gate_init_value=0.01)
    
    # 模拟输入
    token_emb = torch.randn(batch_size, seq_len, d_model)
    emotion_vec = torch.randn(batch_size, d_model)
    scene_vec = torch.randn(batch_size, d_model)
    
    # 融合
    output = fusion(token_emb, emotion_vec, scene_vec)
    
    print(f"  Token embedding: {tuple(token_emb.shape)}")
    print(f"  Emotion vector: {tuple(emotion_vec.shape)}")
    print(f"  Scene vector: {tuple(scene_vec.shape)}")
    print(f"  Fusion output: {tuple(output.shape)}")
    assert tuple(output.shape) == (batch_size, seq_len, d_model)
    
    # 检验门控参数
    gate_e, gate_s = fusion.get_gate_values()
    print(f"  Gate values: emotion={gate_e:.4f}, scene={gate_s:.4f}")
    print(f"  Gates learnable: {fusion.gate_emotion.requires_grad}")
    assert fusion.gate_emotion.requires_grad, "Gate parameters must be learnable!"
    assert fusion.gate_scene.requires_grad, "Gate parameters must be learnable!"
    
    print("  [OK] ContextFusion test passed")
    print()


def test_emotion_injection():
    """核心测试: 验证情感是否成功注入数学层面"""
    print("=" * 60)
    print("4. Emotion Injection Verification (CORE TEST)")
    print("=" * 60)
    
    # 创建上下文感知嵌入层（使用非零门控以显示差异）
    embedding = create_context_aware_embedding(
        preset='small',
        gate_init_value=0.1,  # 使用非零值以确保能看到差异
    )
    
    # 模拟 "Hello" 的token ID
    token_ids = torch.tensor([[100, 200, 300, 400, 500]])  # batch=1, seq=5
    
    # 情感 Neutral (id=0)
    output_neutral = embedding(token_ids, emotion_id=EMOTION_IDS["neutral"])
    
    # 情感 Angry (id=3)
    output_angry = embedding(token_ids, emotion_id=EMOTION_IDS["angry"])
    
    # 计算差异
    diff = output_neutral - output_angry
    diff_sum = diff.abs().sum().item()
    diff_mean = diff.abs().mean().item()
    
    print(f"  Input Token IDs: {token_ids.tolist()}")
    print(f"  Neutral output shape: {tuple(output_neutral.shape)}")
    print(f"  Angry output shape: {tuple(output_angry.shape)}")
    print()
    print(f"  Diff absolute sum: {diff_sum:.6f}")
    print(f"  Diff absolute mean: {diff_mean:.6f}")
    print()
    
    if diff_sum > 0:
        print("  [OK] Emotion successfully injected at math level! (diff != 0)")
    else:
        print("  [FAIL] Emotion injection failed! (diff == 0)")
        raise AssertionError("Emotion injection failed")
    
    print()
    return diff_sum


def test_scene_injection():
    """验证场景是否成功注入"""
    print("=" * 60)
    print("5. Scene Injection Verification")
    print("=" * 60)
    
    embedding = create_context_aware_embedding(
        preset='small',
        gate_init_value=0.1,
    )
    
    token_ids = torch.tensor([[100, 200, 300]])
    
    # 场景 Chat (id=0)
    output_chat = embedding(token_ids, scene_id=SCENE_IDS["chat"])
    
    # 场景 Coding (id=1)
    output_coding = embedding(token_ids, scene_id=SCENE_IDS["coding"])
    
    diff = output_chat - output_coding
    diff_sum = diff.abs().sum().item()
    
    print(f"  Chat scene output shape: {tuple(output_chat.shape)}")
    print(f"  Coding scene output shape: {tuple(output_coding.shape)}")
    print(f"  Diff absolute sum: {diff_sum:.6f}")
    
    if diff_sum > 0:
        print("  [OK] Scene successfully injected at math level!")
    else:
        raise AssertionError("Scene injection failed")
    
    print()


def test_zero_initialization():
    """Verify Zero-Initialization strategy"""
    print("=" * 60)
    print("6. Zero-Initialization Verification")
    print("=" * 60)
    
    # Use gate_init_value=0.0
    embedding = create_context_aware_embedding(
        preset='small',
        gate_init_value=0.0,
    )
    
    gate_e, gate_s = embedding.get_gate_values()
    print(f"  Gate initial values: emotion={gate_e}, scene={gate_s}")
    
    assert gate_e == 0.0, "Emotion gate should be 0"
    assert gate_s == 0.0, "Scene gate should be 0"
    
    # IMPORTANT: Use eval mode to disable dropout
    # Otherwise dropout causes non-deterministic outputs
    embedding.eval()
    
    token_ids = torch.tensor([[100, 200, 300]])
    
    with torch.no_grad():
        output_neutral = embedding(token_ids, emotion_id=0)
        output_angry = embedding(token_ids, emotion_id=3)
    
    diff = (output_neutral - output_angry).abs().sum().item()
    
    print(f"  Output diff when gate=0 (eval mode): {diff:.10f}")
    
    # With gate=0 and eval mode, diff should be exactly 0
    assert diff < 1e-6, f"Zero-init failed: diff={diff}"
    print("  [OK] Zero-Initialization works (diff = 0 when gate = 0)")
    
    print()


def test_batch_processing():
    """测试批量处理"""
    print("=" * 60)
    print("7. Batch Processing Test")
    print("=" * 60)
    
    embedding = create_context_aware_embedding(preset='small')
    
    batch_size = 4
    seq_len = 8
    
    # 批量token IDs
    token_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # 批量情感/场景 IDs
    emotion_ids = torch.tensor([0, 1, 2, 3])  # 每个样本不同情感
    scene_ids = torch.tensor([0, 0, 1, 1])    # 部分样本不同场景
    
    output = embedding(token_ids, emotion_id=emotion_ids, scene_id=scene_ids)
    
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Output shape: {tuple(output.shape)}")
    print(f"  Expected shape: ({batch_size}, {seq_len}, 256)")
    
    assert tuple(output.shape) == (batch_size, seq_len, 256)
    print("  [OK] Batch processing test passed")
    print()


def test_rmsnorm():
    """测试 RMSNorm"""
    print("=" * 60)
    print("8. RMSNorm Normalization Test")
    print("=" * 60)
    
    d_model = 256
    norm = RMSNorm(d_model)
    
    # 输入
    x = torch.randn(2, 10, d_model) * 10  # 放大输入
    y = norm(x)
    
    print(f"  Input mean: {x.mean().item():.4f}")
    print(f"  Input std: {x.std().item():.4f}")
    print(f"  Output mean: {y.mean().item():.4f}")
    print(f"  Output std: {y.std().item():.4f}")
    
    # RMSNorm 后标准差应该接近1
    assert y.std().item() < 2.0, "RMSNorm failed to normalize"
    
    print("  [OK] RMSNorm test passed")
    print()


def test_gate_gradients():
    """验证门控参数的梯度计算"""
    print("=" * 60)
    print("9. Gate Gradient Verification")
    print("=" * 60)
    
    embedding = create_context_aware_embedding(preset='small', gate_init_value=0.1)
    
    token_ids = torch.tensor([[100, 200, 300]])
    
    # 前向传播
    output = embedding(token_ids, emotion_id=2, scene_id=1)
    
    # 模拟损失和反向传播
    loss = output.sum()
    loss.backward()
    
    # 检查梯度
    gate_e = embedding.fusion.gate_emotion
    gate_s = embedding.fusion.gate_scene
    
    print(f"  Emotion gate gradient: {gate_e.grad}")
    print(f"  Scene gate gradient: {gate_s.grad}")
    
    assert gate_e.grad is not None, "Emotion gate should have gradient"
    assert gate_s.grad is not None, "Scene gate should have gradient"
    
    print("  [OK] Gate gradient computation works")
    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Context Fusion Module Acceptance Tests")
    print("=" * 60 + "\n")
    
    try:
        test_emotion_embedding()
        test_scene_embedding()
        test_context_fusion()
        diff = test_emotion_injection()
        test_scene_injection()
        test_zero_initialization()
        test_batch_processing()
        test_rmsnorm()
        test_gate_gradients()
        
        print("\n" + "=" * 60)
        print("[PASS] All tests passed!")
        print("=" * 60)
        print()
        print("Acceptance Results:")
        print(f"  - Emotion injection diff: {diff:.6f} (> 0 = SUCCESS)")
        print("  - Gate parameters learnable: OK")
        print("  - Zero-Initialization: OK")
        print("  - RMSNorm normalization: OK")
        print()
        print("Context Fusion module is ready!")
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

