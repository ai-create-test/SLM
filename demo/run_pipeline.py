"""
End-to-End Pipeline Demo

演示完整的推理流程：
Text → Encoder → Memory → Brain → Decoder → Text

运行:
    python -m demo.run_pipeline
"""

import sys
sys.path.insert(0, ".")

import torch
from app.interfaces.config import Config
from app.io import ParagraphEncoder, ParagraphDecoder
from app.brain import DynamicsModel, ReasoningLoop
from app.memory import LatentMemoryBank, CrossAttentionFuser
from app.modulation import EmotionEncoder


def demo_simple_generate():
    """简单生成演示"""
    print("\n" + "=" * 60)
    print(" Simple Generate Demo")
    print("=" * 60)
    
    # 创建组件
    print("\n[1] Creating components...")
    
    d_latent = 64
    d_model = 128
    
    encoder = ParagraphEncoder(d_model=d_model, d_latent=d_latent, use_vq=True, max_length=32)
    decoder = ParagraphDecoder(d_latent=d_latent, d_model=d_model, vocab_size=50000, max_length=32)
    dynamics = DynamicsModel(d_latent=d_latent, d_model=d_model, num_layers=2, brain_type="gru")
    
    print("  [OK] Encoder, Decoder, DynamicsModel created")
    
    # 编码输入
    print("\n[2] Encoding input text...")
    input_text = "Hello world this is a test."
    
    enc_output = encoder([input_text])
    z_input = enc_output.latent.vector  # [1, d_latent]
    print(f"  Input: '{input_text}'")
    print(f"  Encoded: {z_input.shape}")
    
    # 动力学预测
    print("\n[3] Brain predicting next latent...")
    z_seq = z_input.unsqueeze(0)  # [1, 1, d_latent]
    dyn_output = dynamics(z_seq)
    z_pred = dyn_output.predicted_latent  # [1, d_latent]
    print(f"  Predicted: {z_pred.shape}")
    
    # 解码
    print("\n[4] Decoding to text...")
    dec_output = decoder.generate(z_pred, max_length=20, temperature=0.8)
    output_text = dec_output.text[0] if dec_output.text else "(empty)"
    print(f"  Generated: '{output_text}'")
    
    print("\n[DONE] Simple generate completed!")


def demo_with_memory():
    """带记忆的生成演示"""
    print("\n" + "=" * 60)
    print(" Generate with Memory Demo")
    print("=" * 60)
    
    d_latent = 64
    d_model = 128
    
    # 创建组件
    print("\n[1] Creating components...")
    encoder = ParagraphEncoder(d_model=d_model, d_latent=d_latent, use_vq=True, max_length=32)
    decoder = ParagraphDecoder(d_latent=d_latent, d_model=d_model, vocab_size=50000, max_length=32)
    dynamics = DynamicsModel(d_latent=d_latent, d_model=d_model, num_layers=2, brain_type="gru")
    memory_bank = LatentMemoryBank(d_latent=d_latent, max_size=100)
    memory_fuser = CrossAttentionFuser(d_latent=d_latent)
    
    print("  [OK] All components created")
    
    # 先添加一些记忆
    print("\n[2] Adding memories...")
    memories = [
        "The weather is sunny today.",
        "I like to read books.",
        "Python is a great language.",
    ]
    
    for mem in memories:
        enc_out = encoder([mem])
        memory_bank.add(enc_out.latent.vector, content=mem)
    print(f"  Added {len(memories)} memories")
    
    # 编码新输入
    print("\n[3] Encoding new input...")
    input_text = "What is programming?"
    enc_output = encoder([input_text])
    z_input = enc_output.latent.vector
    print(f"  Input: '{input_text}'")
    
    # 检索记忆
    print("\n[4] Retrieving related memories...")
    retrieved = memory_bank.retrieve(z_input, k=2)
    print(f"  Retrieved {len(retrieved)} memories:")
    for item in retrieved:
        print(f"    - '{item.content}'")
    
    # 融合记忆
    print("\n[5] Fusing memory context...")
    if retrieved:
        # z_input is [1, d_latent], forward_from_items expects [batch, d_latent]
        z_fused = memory_fuser.forward_from_items(z_input, retrieved)
        print(f"  Fused: {z_fused.shape}")
    else:
        z_fused = z_input
    
    # 动力学预测
    print("\n[6] Brain predicting...")
    z_seq = z_fused.unsqueeze(1)  # [1, 1, d_latent]
    dyn_output = dynamics(z_seq)
    z_pred = dyn_output.predicted_latent
    
    # 解码
    print("\n[7] Decoding...")
    dec_output = decoder.generate(z_pred, max_length=20, temperature=0.7)
    output_text = dec_output.text[0] if dec_output.text else "(empty)"
    print(f"  Generated: '{output_text}'")
    
    print("\n[DONE] Memory-augmented generate completed!")


def demo_with_emotion():
    """带情感调制的生成演示"""
    print("\n" + "=" * 60)
    print(" Generate with Emotion Demo")
    print("=" * 60)
    
    d_latent = 64
    d_model = 128
    d_emotion = 32
    
    # 创建组件
    print("\n[1] Creating components...")
    encoder = ParagraphEncoder(d_model=d_model, d_latent=d_latent, use_vq=True, max_length=32)
    decoder = ParagraphDecoder(d_latent=d_latent, d_model=d_model, vocab_size=50000, max_length=32)
    emotion_encoder = EmotionEncoder(d_emotion=d_emotion)
    
    # 使用带情感的动力学模型
    from app.brain.modulated_dynamics import ModulatedDynamicsModel
    dynamics = ModulatedDynamicsModel(d_latent=d_latent, d_model=d_model, d_condition=d_emotion, num_layers=2)
    
    print("  [OK] All components created")
    
    # 编码输入
    print("\n[2] Encoding input...")
    input_text = "Tell me a story."
    enc_output = encoder([input_text])
    z_input = enc_output.latent.vector
    z_seq = z_input.unsqueeze(0).unsqueeze(0)  # [1, 1, d_latent]
    print(f"  Input: '{input_text}'")
    
    # 不同情感生成
    emotions = ["neutral", "happy", "angry"]
    
    for emo_name in emotions:
        print(f"\n[3] Generating with emotion: {emo_name}")
        
        # 编码情感
        emo_vec = emotion_encoder.encode_name(emo_name)  # [1, d_emotion]
        
        # 确保 z_seq 是 3D: [batch, seq_len, d_latent]
        if z_seq.dim() == 4:
            z_seq = z_seq.squeeze(0)
        
        # 预测
        dyn_output = dynamics(z_seq, emotion=emo_vec)
        z_pred = dyn_output.predicted_latent
        
        # 解码
        dec_output = decoder.generate(z_pred, max_length=20, temperature=0.8)
        output_text = dec_output.text[0] if dec_output.text else "(empty)"
        print(f"    → {emo_name}: '{output_text}'")
    
    print("\n[DONE] Emotion-modulated generate completed!")


def run_all_demos():
    print("\n" + "#" * 60)
    print("#  NeuralFlow End-to-End Demo")
    print("#" * 60)
    
    try:
        demo_simple_generate()
        demo_with_memory()
        demo_with_emotion()
        
        print("\n" + "#" * 60)
        print("#  ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("#" * 60 + "\n")
        return True
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_demos()
    sys.exit(0 if success else 1)
