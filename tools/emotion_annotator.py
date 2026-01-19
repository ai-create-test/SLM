#!/usr/bin/env python
"""
Emotion Annotator - 情感标注工具

交互式工具，用于标注自定义情感词的 VAD 值。

使用方法:
    python tools/emotion_annotator.py --interactive
    python tools/emotion_annotator.py --batch input.txt
    python tools/emotion_annotator.py --export

VAD 标注指南:
    Valence (效价):  -1.0 (极度不愉快) → 0.0 (中性) → +1.0 (极度愉快)
    Arousal (唤醒):  -1.0 (极度平静) → 0.0 (中等) → +1.0 (极度激动)
    Dominance (支配): -1.0 (完全被动) → 0.0 (中等) → +1.0 (完全主导)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class EmotionEntry:
    """情感词条"""
    text: str
    valence: float
    arousal: float
    dominance: float
    source: str = "human"
    notes: str = ""


class EmotionAnnotator:
    """
    情感标注工具
    
    支持:
    - 交互式单条标注
    - 批量文件标注
    - 导出到 custom_emotions.json
    """
    
    def __init__(self, output_path: Optional[str] = None):
        """
        Args:
            output_path: 输出文件路径 (默认 data/vad/custom_emotions.json)
        """
        if output_path:
            self.output_path = Path(output_path)
        else:
            # 相对于项目根目录
            self.output_path = Path(__file__).parent.parent / "data" / "vad" / "custom_emotions.json"
        
        self.entries: Dict[str, EmotionEntry] = {}
        self._load_existing()
    
    def _load_existing(self) -> None:
        """加载已有词条"""
        if self.output_path.exists():
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                for text, values in data.items():
                    if text.startswith("_"):
                        continue
                    self.entries[text] = EmotionEntry(
                        text=text,
                        valence=values.get("v", values.get("valence", 0.0)),
                        arousal=values.get("a", values.get("arousal", 0.0)),
                        dominance=values.get("d", values.get("dominance", 0.0)),
                        source=values.get("source", "human"),
                        notes=values.get("notes", ""),
                    )
            except Exception as e:
                print(f"Warning: Failed to load existing entries: {e}")
    
    def save(self) -> None:
        """保存到文件"""
        # 确保目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为 JSON 格式
        data = {
            "_comment": "Custom emotion annotations. Add your own entries here.",
        }
        for text, entry in self.entries.items():
            data[text] = {
                "v": entry.valence,
                "a": entry.arousal,
                "d": entry.dominance,
                "source": entry.source,
            }
            if entry.notes:
                data[text]["notes"] = entry.notes
        
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(self.entries)} entries to {self.output_path}")
    
    def add(self, entry: EmotionEntry) -> None:
        """添加词条"""
        self.entries[entry.text] = entry
    
    def annotate_single(self, text: str) -> EmotionEntry:
        """交互式标注单条"""
        print(f"\n{'='*50}")
        print(f"文本: {text}")
        print(f"{'='*50}")
        print("\n请输入 VAD 值 (范围 -1.0 到 1.0):")
        print("  Valence: -1.0=极度不愉快, 0.0=中性, 1.0=极度愉快")
        print("  Arousal: -1.0=极度平静, 0.0=中等, 1.0=极度激动")
        print("  Dominance: -1.0=完全被动, 0.0=中等, 1.0=完全主导")
        print()
        
        def get_float(prompt: str, default: float = 0.0) -> float:
            while True:
                try:
                    value = input(f"{prompt} [{default}]: ").strip()
                    if not value:
                        return default
                    val = float(value)
                    if -1.0 <= val <= 1.0:
                        return val
                    print("  错误: 值必须在 -1.0 到 1.0 之间")
                except ValueError:
                    print("  错误: 请输入有效数字")
        
        valence = get_float("Valence")
        arousal = get_float("Arousal")
        dominance = get_float("Dominance")
        notes = input("备注 (可选): ").strip()
        
        entry = EmotionEntry(
            text=text,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            source="human",
            notes=notes,
        )
        
        self.add(entry)
        print(f"\n✓ 已添加: {text} -> V={valence}, A={arousal}, D={dominance}")
        return entry
    
    def annotate_interactive(self) -> None:
        """交互式标注模式"""
        print("\n" + "="*60)
        print(" Emotion Annotator - 交互式标注模式")
        print("="*60)
        print("\n输入情感词/短语进行标注，输入 'q' 退出")
        print(f"当前已有 {len(self.entries)} 条记录\n")
        
        while True:
            try:
                text = input("\n输入文本 (q=退出, s=保存): ").strip()
                
                if text.lower() == 'q':
                    break
                elif text.lower() == 's':
                    self.save()
                    continue
                elif not text:
                    continue
                
                if text in self.entries:
                    existing = self.entries[text]
                    print(f"  已存在: V={existing.valence}, A={existing.arousal}, D={existing.dominance}")
                    if input("  覆盖? (y/n): ").lower() != 'y':
                        continue
                
                self.annotate_single(text)
                
            except KeyboardInterrupt:
                print("\n\n中断...")
                break
        
        # 退出前询问保存
        if input("\n保存更改? (y/n): ").lower() == 'y':
            self.save()
    
    def annotate_batch(self, input_path: str) -> None:
        """批量标注文件中的词条"""
        path = Path(input_path)
        if not path.exists():
            print(f"Error: File not found: {path}")
            return
        
        with open(path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"\n加载 {len(texts)} 条待标注文本")
        
        for i, text in enumerate(texts, 1):
            print(f"\n[{i}/{len(texts)}]")
            
            if text in self.entries:
                existing = self.entries[text]
                print(f"已存在: {text} -> V={existing.valence}, A={existing.arousal}, D={existing.dominance}")
                if input("跳过? (y/n): ").lower() == 'y':
                    continue
            
            self.annotate_single(text)
            
            # 每 5 条自动保存
            if i % 5 == 0:
                self.save()
        
        self.save()
    
    def export(self) -> None:
        """显示所有词条"""
        print(f"\n当前词条 ({len(self.entries)} 条):")
        print("-" * 60)
        
        for text, entry in sorted(self.entries.items()):
            print(f"  {text}: V={entry.valence:+.2f}, A={entry.arousal:+.2f}, D={entry.dominance:+.2f}")
        
        print("-" * 60)
        print(f"存储位置: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="情感标注工具 - 标注自定义情感词的 VAD 值"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="交互式标注模式"
    )
    parser.add_argument(
        "--batch", "-b",
        type=str,
        help="批量标注文件 (每行一个词)"
    )
    parser.add_argument(
        "--export", "-e",
        action="store_true",
        help="显示所有已标注词条"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出文件路径"
    )
    
    args = parser.parse_args()
    
    annotator = EmotionAnnotator(output_path=args.output)
    
    if args.export:
        annotator.export()
    elif args.batch:
        annotator.annotate_batch(args.batch)
    elif args.interactive:
        annotator.annotate_interactive()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
