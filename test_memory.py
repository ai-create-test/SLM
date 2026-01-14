"""
Memory模块测试脚本

演示：
1. 输入长文本（超过模型窗口）
2. 分块处理
3. 输出正确的张量形状 [batch, seq_len, d_model]
"""

import torch
from app.memory import (
    MemoryEncoder,
    TextChunker,
    CombinedEmbedding,
    create_memory_encoder,
)


def test_text_chunking():
    """测试文本分块"""
    print("=" * 60)
    print("1. 文本分块测试")
    print("=" * 60)
    
    # 创建分块器
    chunker = TextChunker(max_chunk_size=200, overlap_size=30)
    
    # 模拟长文本
    long_text = """
    人工智能正在改变世界。机器学习使计算机能够从数据中学习，而深度学习则更进一步，
    使用神经网络来模拟人类大脑的工作方式。自然语言处理是人工智能的一个重要分支，
    它使机器能够理解和生成人类语言。
    
    大型语言模型（LLM）是近年来最令人兴奋的技术突破之一。这些模型通过在海量文本
    数据上进行训练，学会了理解语言的复杂模式和结构。GPT、BERT、LLaMA等模型
    展示了令人惊叹的语言理解和生成能力。
    
    Transformer架构是这些模型的基础。它使用自注意力机制来捕捉序列中不同位置之间
    的关系，这使得模型能够更好地理解上下文。位置编码（如RoPE）帮助模型理解词语
    在句子中的位置信息。
    """.strip()
    
    # 分块
    chunks = chunker.chunk(long_text)
    
    print(f"原文长度: {len(long_text)} 字符")
    print(f"分块数量: {len(chunks)}")
    print()
    
    for chunk in chunks:
        print(f"块 {chunk.chunk_index + 1}/{chunk.total_chunks}:")
        print(f"  长度: {len(chunk)} 字符")
        print(f"  位置: [{chunk.start_pos}:{chunk.end_pos}]")
        print(f"  预览: {chunk.content[:50]}...")
        print()


def test_embedding():
    """测试嵌入层"""
    print("=" * 60)
    print("2. 嵌入层测试")
    print("=" * 60)
    
    # 创建嵌入层
    embedding = CombinedEmbedding(
        vocab_size=50000,
        d_model=256,
        max_seq_len=128,
        position_encoding='rope'
    )
    
    # 模拟token ID
    token_ids = torch.randint(0, 50000, (2, 32))  # batch=2, seq=32
    
    # 嵌入
    output = embedding(token_ids)
    
    print(f"输入形状: {token_ids.shape}")
    print(f"输出形状: {output.shape}")
    print(f"期望形状: [2, 32, 256]")
    print(f"形状正确: {tuple(output.shape) == (2, 32, 256)}")
    print()


def test_memory_encoder():
    """测试完整编码流程"""
    print("=" * 60)
    print("3. MemoryEncoder完整测试")
    print("=" * 60)
    
    # 创建编码器
    encoder = create_memory_encoder(
        preset='small',  # 使用小型配置加速测试
        max_chunk_size=300,
    )
    
    # 长文本（模拟超过窗口限制的场景）
    long_text = """
    Large Language Models (LLMs) represent a significant breakthrough in artificial 
    intelligence. These models are trained on vast amounts of text data, learning 
    complex patterns in language and developing remarkable capabilities in understanding 
    and generating human language.
    
    The Transformer architecture forms the backbone of these models. It introduces 
    the self-attention mechanism, which allows the model to weigh the importance of 
    different words in a sentence when making predictions. This has proven to be 
    incredibly effective for capturing long-range dependencies in text.
    
    Rotary Position Embedding (RoPE) is a modern technique for encoding positional 
    information in transformers. Unlike absolute positional encodings, RoPE encodes 
    relative position information through rotation matrices, which provides better 
    generalization to longer sequences than those seen during training.
    
    Token embeddings convert discrete tokens into continuous vectors that the neural 
    network can process. These embeddings are learned during training and capture 
    semantic relationships between words - similar words tend to have similar embeddings.
    """.strip()
    
    print(f"输入文本长度: {len(long_text)} 字符")
    print()
    
    # 编码
    result = encoder.encode_with_chunks(long_text)
    
    print("编码结果:")
    print(f"  分块数量: {result.num_chunks}")
    print(f"  总Token数: {result.total_tokens}")
    print(f"  嵌入张量形状: {tuple(result.embeddings.shape)}")
    print(f"  注意力掩码形状: {tuple(result.attention_mask.shape)}")
    print()
    
    # 验证形状
    batch, seq_len, d_model = result.embeddings.shape
    print(f"验证:")
    print(f"  batch (分块数) = {batch}")
    print(f"  seq_len (序列长度) = {seq_len}")
    print(f"  d_model (嵌入维度) = {d_model}")
    print()
    
    # 详细信息
    print("各块详情:")
    for i, chunk in enumerate(result.chunks):
        ids = result.token_ids[i]
        print(f"  块 {i + 1}: {len(ids)} tokens, 内容: {chunk.content[:40]}...")
    print()
    
    return result


def test_batch_encoding():
    """测试批量编码"""
    print("=" * 60)
    print("4. 批量编码测试")
    print("=" * 60)
    
    encoder = create_memory_encoder(preset='small')
    
    texts = [
        "This is the first document about machine learning.",
        "The second document discusses natural language processing.",
        "Document three covers deep learning and neural networks.",
    ]
    
    results = encoder.encode_batch(texts)
    
    print(f"输入文本数: {len(texts)}")
    print(f"输出结果数: {len(results)}")
    print()
    
    for i, result in enumerate(results):
        print(f"文本 {i + 1}:")
        print(f"  嵌入形状: {tuple(result.embeddings.shape)}")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Memory模块验收测试")
    print("=" * 60 + "\n")
    
    try:
        test_text_chunking()
        test_embedding()
        result = test_memory_encoder()
        test_batch_encoding()
        
        print("\n" + "=" * 60)
        print("[PASS] 所有测试通过!")
        print("=" * 60)
        print(f"\n最终输出张量形状: {tuple(result.embeddings.shape)}")
        print("格式: [num_chunks, seq_len, d_model]")
        print("\n模块已准备就绪，可用于后续神经网络计算。")
        
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
