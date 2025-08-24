import os
# 设置Hugging Face镜像站，必须在导入transformers之前设置

import torch
import json
from transformers import BertTokenizer
import argparse
from datetime import datetime

# 导入自定义模块
from model import BERTSentimentAnalyzer, load_model_info

class BERTInference:
    """
    BERT情感分析推理类
    """
    
    def __init__(self, model_path=None, model_name='bert-base-uncased', device=None):
        """
        初始化推理器
        
        Args:
            model_path (str): 训练好的模型路径
            model_name (str): BERT模型名称
            device (str): 设备类型
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型和tokenizer
        self.model = None
        self.tokenizer = None
        self.model_config = None
        
        # 情感标签映射
        self.label_map = {0: '负面', 1: '正面'}
        self.confidence_threshold = 0.6  # 置信度阈值
        
        print(f"推理器初始化完成，使用设备: {self.device}")
    
    def load_model(self, model_path=None):
        """
        加载训练好的模型
        
        Args:
            model_path (str): 模型文件路径
        """
        if model_path:
            self.model_path = model_path
        
        if not self.model_path or not os.path.exists(self.model_path):
            print("警告: 未找到训练好的模型，将使用预训练BERT模型")
            self._load_pretrained_model()
            return
        
        print(f"正在加载模型: {self.model_path}")
        
        try:
            # 加载模型检查点
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 获取模型配置
            self.model_config = checkpoint.get('model_config', {})
            model_name = self.model_config.get('model_name', self.model_name)
            num_classes = self.model_config.get('num_classes', 2)
            
            # 创建模型
            self.model = BERTSentimentAnalyzer(
                model_name=model_name,
                num_classes=num_classes
            )
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 初始化tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            
            # 打印模型信息
            if 'results' in checkpoint:
                results = checkpoint['results']
                print(f"模型性能:")
                print(f"  验证准确率: {results.get('best_val_accuracy', 'N/A'):.4f}")
                print(f"  测试准确率: {results.get('test_accuracy', 'N/A'):.4f}")
            
            print("模型加载成功！")
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("将使用预训练BERT模型")
            self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        """
        加载预训练BERT模型（未经微调）
        """
        print(f"正在加载预训练模型: {self.model_name}")
        
        self.model = BERTSentimentAnalyzer(model_name=self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        print("预训练模型加载完成（注意: 未经微调，性能可能较差）")
    
    def predict_single(self, text, max_length=512, return_details=False):
        """
        对单个文本进行情感预测
        
        Args:
            text (str): 输入文本
            max_length (int): 最大序列长度
            return_details (bool): 是否返回详细信息
            
        Returns:
            dict: 预测结果
        """
        if not self.model or not self.tokenizer:
            raise ValueError("模型未加载，请先调用load_model()")
        
        # 预处理文本
        text = str(text).strip()
        if not text:
            return {'error': '输入文本为空'}
        
        try:
            # 使用模型的predict方法
            prediction, probability = self.model.predict(text, max_length)
            
            # 获取情感标签
            sentiment = self.label_map.get(prediction, '未知')
            
            # 判断置信度
            confidence_level = 'high' if probability >= self.confidence_threshold else 'low'
            
            result = {
                'text': text,
                'prediction': prediction,
                'sentiment': sentiment,
                'probability': probability,
                'confidence': confidence_level
            }
            
            if return_details:
                result.update({
                    'model_name': self.model_name,
                    'device': str(self.device),
                    'max_length': max_length,
                    'timestamp': datetime.now().isoformat()
                })
            
            return result
            
        except Exception as e:
            return {'error': f'预测时出错: {str(e)}'}
    
    def predict_batch(self, texts, max_length=512, batch_size=16):
        """
        批量预测
        
        Args:
            texts (list): 文本列表
            max_length (int): 最大序列长度
            batch_size (int): 批次大小
            
        Returns:
            list: 预测结果列表
        """
        if not self.model or not self.tokenizer:
            raise ValueError("模型未加载，请先调用load_model()")
        
        results = []
        
        print(f"开始批量预测 {len(texts)} 个文本...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch_texts:
                result = self.predict_single(text, max_length)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # 显示进度
            processed = min(i + batch_size, len(texts))
            print(f"已处理: {processed}/{len(texts)}")
        
        return results
    
    def interactive_predict(self):
        """
        交互式预测模式
        """
        if not self.model or not self.tokenizer:
            print("错误: 模型未加载")
            return
        
        print("\n=" * 50)
        print("BERT情感分析 - 交互式预测")
        print("=" * 50)
        print("输入文本进行情感分析，输入 'quit' 退出")
        print("-" * 50)
        
        while True:
            try:
                text = input("\n请输入文本: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("退出预测模式")
                    break
                
                if not text:
                    print("请输入有效文本")
                    continue
                
                # 进行预测
                result = self.predict_single(text, return_details=True)
                
                if 'error' in result:
                    print(f"错误: {result['error']}")
                    continue
                
                # 显示结果
                print(f"\n预测结果:")
                print(f"  文本: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
                print(f"  情感: {result['sentiment']}")
                print(f"  概率: {result['probability']:.4f}")
                print(f"  置信度: {result['confidence']}")
                
                # 添加解释
                if result['confidence'] == 'low':
                    print(f"  注意: 置信度较低（< {self.confidence_threshold}），结果可能不准确")
                
            except KeyboardInterrupt:
                print("\n\n退出预测模式")
                break
            except Exception as e:
                print(f"发生错误: {e}")
    
    def analyze_file(self, file_path, output_path=None, max_length=512):
        """
        分析文件中的文本
        
        Args:
            file_path (str): 输入文件路径
            output_path (str): 输出文件路径
            max_length (int): 最大序列长度
        """
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 {file_path}")
            return
        
        print(f"正在分析文件: {file_path}")
        
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 清理文本
            texts = [line.strip() for line in lines if line.strip()]
            
            if not texts:
                print("文件中没有有效文本")
                return
            
            # 批量预测
            results = self.predict_batch(texts, max_length)
            
            # 统计结果
            positive_count = sum(1 for r in results if r.get('prediction') == 1)
            negative_count = sum(1 for r in results if r.get('prediction') == 0)
            error_count = sum(1 for r in results if 'error' in r)
            
            print(f"\n分析完成:")
            print(f"  总文本数: {len(texts)}")
            print(f"  正面情感: {positive_count}")
            print(f"  负面情感: {negative_count}")
            print(f"  错误数: {error_count}")
            
            # 保存结果
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"结果已保存到: {output_path}")
            
            return results
            
        except Exception as e:
            print(f"分析文件时出错: {e}")
            return None

def find_latest_model(model_dir='saved_models'):
    """
    查找最新的模型文件
    
    Args:
        model_dir (str): 模型目录
        
    Returns:
        str: 最新模型文件路径
    """
    if not os.path.exists(model_dir):
        return None
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if not model_files:
        return None
    
    # 按修改时间排序
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    
    return os.path.join(model_dir, model_files[0])

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='BERT情感分析推理')
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--text', type=str, help='要分析的文本')
    parser.add_argument('--file', type=str, help='要分析的文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')
    parser.add_argument('--max_length', type=int, default=256, help='最大序列长度')
    
    args = parser.parse_args()
    
    # 查找模型文件
    model_path = args.model
    if not model_path:
        model_path = find_latest_model()
        if model_path:
            print(f"自动找到模型: {model_path}")
    
    # 创建推理器
    inferencer = BERTInference(model_path=model_path)
    inferencer.load_model()
    
    # 根据参数执行不同操作
    if args.interactive:
        # 交互式模式
        inferencer.interactive_predict()
    elif args.text:
        # 单文本预测
        result = inferencer.predict_single(args.text, args.max_length, return_details=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.file:
        # 文件分析
        inferencer.analyze_file(args.file, args.output, args.max_length)
    else:
        # 默认进入交互式模式
        inferencer.interactive_predict()

if __name__ == "__main__":
    # 如果没有命令行参数，直接运行交互式模式
    import sys
    if len(sys.argv) == 1:
        print("BERT情感分析推理")
        print("使用方法:")
        print("  python inference.py --interactive  # 交互式模式")
        print("  python inference.py --text '文本'   # 单文本预测")
        print("  python inference.py --file 文件路径  # 文件分析")
        print("\n正在启动交互式模式...")
        
        # 创建推理器并运行
        model_path = find_latest_model()
        inferencer = BERTInference(model_path=model_path)
        inferencer.load_model()
        inferencer.interactive_predict()
    else:
        main()