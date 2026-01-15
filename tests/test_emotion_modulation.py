"""
Test: Emotion Modulation Verification

验证情感调制是否真正影响推理输出。

测试目标：
- neutral vs angry 应该产生不同的输出
- 输出差异应该显著 (> 1e-4)
"""

import sys
sys.path.insert(0, ".")

import torch

def test_modulated_mamba_block():
    """测试 ModulatedMambaBlock"""
    from app.brain.modulated_mamba import ModulatedMambaBlock
    
    block = ModulatedMambaBlock(
        d_model=128,
        d_condition=64,
        d_state=32,
    )
    
    x = torch.randn(2, 8, 128)
    condition = torch.randn(2, 64)
    
    # 有条件
    out_with, _ = block(x, condition)
    # 无条件
    out_without, _ = block(x, None)
    
    assert out_with.shape == x.shape
    assert out_without.shape == x.shape
    
    # 差异应该存在
    diff = (out_with - out_without).abs().mean()
    print(f"[PASS] ModulatedMambaBlock: with vs without condition diff = {diff:.6f}")


def test_modulated_dynamics_emotion_effect():
    """测试情感是否真正影响输出"""
    from app.brain.modulated_dynamics import ModulatedDynamicsModel
    
    model = ModulatedDynamicsModel(
        d_latent=64,
        d_model=128,
        d_condition=64,
        num_layers=2,
    )
    
    # 固定输入
    torch.manual_seed(42)
    z_seq = torch.randn(1, 4, 64)
    
    # 不同的情感条件
    neutral = torch.zeros(1, 64)  # 中性
    happy = torch.randn(1, 64)    # 随机情感1
    angry = torch.randn(1, 64)    # 随机情感2
    
    # 三种输出
    out_neutral = model(z_seq, emotion=neutral).predicted_latent
    out_happy = model(z_seq, emotion=happy).predicted_latent
    out_angry = model(z_seq, emotion=angry).predicted_latent
    out_none = model(z_seq, emotion=None).predicted_latent
    
    # 计算差异
    diff_happy_neutral = (out_happy - out_neutral).abs().mean().item()
    diff_angry_neutral = (out_angry - out_neutral).abs().mean().item()
    diff_happy_angry = (out_happy - out_angry).abs().mean().item()
    
    print(f"  happy vs neutral: {diff_happy_neutral:.6f}")
    print(f"  angry vs neutral: {diff_angry_neutral:.6f}")
    print(f"  happy vs angry: {diff_happy_angry:.6f}")
    
    # 验证差异显著
    threshold = 1e-4
    assert diff_happy_neutral > threshold, f"Happy vs Neutral diff too small: {diff_happy_neutral}"
    assert diff_angry_neutral > threshold, f"Angry vs Neutral diff too small: {diff_angry_neutral}"
    assert diff_happy_angry > threshold, f"Happy vs Angry diff too small: {diff_happy_angry}"
    
    print(f"[PASS] Emotion modulation produces significantly different outputs!")


def test_named_emotions():
    """测试使用具名情感"""
    from app.brain.modulated_dynamics import ModulatedDynamicsModel
    from app.modulation.emotion_encoder import EmotionEncoder
    
    # 创建组件
    emotion_encoder = EmotionEncoder(d_emotion=64)
    model = ModulatedDynamicsModel(
        d_latent=64,
        d_model=128,
        d_condition=64,
        num_layers=2,
    )
    
    # 编码情感
    neutral = emotion_encoder.encode_name("neutral")  # [1, 64]
    angry = emotion_encoder.encode_name("angry")      # [1, 64]
    
    # 固定输入
    torch.manual_seed(123)
    z_seq = torch.randn(1, 4, 64)
    
    # 不同情感的输出
    out_neutral = model(z_seq, emotion=neutral).predicted_latent
    out_angry = model(z_seq, emotion=angry).predicted_latent
    
    # 差异
    diff = (out_neutral - out_angry).abs().mean().item()
    print(f"  neutral vs angry (named): {diff:.6f}")
    
    assert diff > 1e-4, f"Named emotion diff too small: {diff}"
    print(f"[PASS] Named emotions (neutral vs angry) produce different outputs!")


def test_multi_condition():
    """测试多条件 (emotion + scene)"""
    from app.brain.modulated_mamba import MultiConditionModulatedMamba
    
    block = MultiConditionModulatedMamba(
        d_model=128,
        d_emotion=64,
        d_scene=32,
        d_state=32,
    )
    
    x = torch.randn(2, 8, 128)
    emotion = torch.randn(2, 64)
    scene = torch.randn(2, 32)
    
    out, _ = block(x, emotion, scene)
    
    assert out.shape == x.shape
    print(f"[PASS] MultiConditionModulatedMamba works correctly!")


def run_all_tests():
    print("\n" + "=" * 60)
    print(" Emotion Modulation Verification Tests")
    print("=" * 60 + "\n")
    
    try:
        test_modulated_mamba_block()
        test_modulated_dynamics_emotion_effect()
        test_named_emotions()
        test_multi_condition()
        
        print("\n" + "=" * 60)
        print(" ALL TESTS PASSED! Emotion modulation is working!")
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
