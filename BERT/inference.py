# Copyright (c) 2025 NIS Lab, Hebei Normal University
# Author: Wang Minghu <wangminghu41@163.com>
# Internal Use Only. Unauthorized distribution is strictly prohibited.

import os
import torch
import json
from transformers import BertTokenizer
from datetime import datetime

# 导入自定义模块
from model import BERTSentimentAnalyzer

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
    

    
    def interactive_predict(self):
        """
        交互式预测模式
        """
        if not self.model or not self.tokenizer:
            print("错误: 模型未加载")
            return
        
        print("BERT情感分析 - 交互式预测")
        print("输入文本进行情感分析，输入 'quit' 退出")
        
        while True:
            try:
                text = input("请输入文本: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("退出预测模式")
                    break
                
                if not text:
                    print("请输入有效文本")
                    continue
                
                # 进行预测
                result = self.predict_single(text)
                
                if 'error' in result:
                    print(f"错误: {result['error']}")
                    continue
                
                # 显示结果
                print(f"情感: {result['sentiment']}")
                print(f"概率: {result['probability']:.4f}")
                
            except KeyboardInterrupt:
                print("退出预测模式")
                break
            except Exception as e:
                print(f"发生错误: {e}")
    


def find_latest_model(model_dir=None):
    """
    查找最新的模型文件
    
    Args:
        model_dir (str): 模型目录
        
    Returns:
        str: 最新模型文件路径
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'bert')
    
    if not os.path.exists(model_dir):
        return None
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if not model_files:
        return None
    
    # 按修改时间排序
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    
    return os.path.join(model_dir, model_files[0])

if __name__ == "__main__":
    # 直接运行交互式模式
    print("BERT情感分析推理")
    
    # 创建推理器并运行
    model_path = find_latest_model()
    if model_path:
        print(f"找到模型: {model_path}")
    
    inferencer = BERTInference(model_path=model_path)
    inferencer.load_model()
    inferencer.interactive_predict()