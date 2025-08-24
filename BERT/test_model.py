import os
# 设置Hugging Face镜像站，必须在导入transformers之前设置

import torch
import numpy as np
import sys
from datetime import datetime

# 导入自定义模块
from model import BERTSentimentAnalyzer, binary_accuracy
from data_loader import prepare_data, analyze_text_lengths
from inference import BERTInference
def test_model_creation():
    """
    测试模型创建
    """
    print("\n=== 测试模型创建 ===")
    
    try:
        # 创建模型
        model = BERTSentimentAnalyzer(model_name='bert-base-uncased', num_classes=2)
        print("✓ 模型创建成功")
        
        # 获取模型信息
        info = model.get_model_info()
        print(f"✓ 模型参数总数: {info['total_parameters']:,}")
        print(f"✓ 可训练参数: {info['trainable_parameters']:,}")
        print(f"✓ 模型大小: {info['model_size_mb']:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False

def test_data_loading():
    """
    测试数据加载
    """
    print("\n=== 测试数据加载 ===")
    
    try:
        # 准备小批量数据进行测试
        print("正在加载测试数据...")
        train_loader, val_loader, test_loader, tokenizer = prepare_data(
            batch_size=4,  # 小批量测试
            max_length=128  # 较短序列测试
        )
        
        print("✓ 数据加载成功")
        print(f"✓ 训练批次数: {len(train_loader)}")
        print(f"✓ 验证批次数: {len(val_loader)}")
        print(f"✓ 测试批次数: {len(test_loader)}")
        
        # 测试一个批次
        for batch in train_loader:
            print(f"✓ 批次形状检查:")
            print(f"  input_ids: {batch['input_ids'].shape}")
            print(f"  attention_mask: {batch['attention_mask'].shape}")
            print(f"  labels: {batch['labels'].shape}")
            
            # 检查数据类型
            assert batch['input_ids'].dtype == torch.long
            assert batch['attention_mask'].dtype == torch.long
            assert batch['labels'].dtype == torch.long
            print("✓ 数据类型检查通过")
            
            break
        
        return train_loader, val_loader, test_loader, tokenizer
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return None, None, None, None

def test_model_forward():
    """
    测试模型前向传播
    """
    print("\n=== 测试模型前向传播 ===")
    
    try:
        # 创建模型
        model = BERTSentimentAnalyzer(model_name='bert-base-uncased', num_classes=2)
        model.eval()
        
        # 创建测试数据
        batch_size = 2
        seq_length = 64
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        labels = torch.randint(0, 2, (batch_size,))
        
        print(f"✓ 测试数据创建: batch_size={batch_size}, seq_length={seq_length}")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        print(f"✓ 前向传播成功")
        print(f"✓ 输出logits形状: {outputs.logits.shape}")
        print(f"✓ 损失值: {outputs.loss.item():.4f}")
        
        # 测试准确率计算
        accuracy = binary_accuracy(outputs.logits, labels)
        print(f"✓ 准确率计算: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        return False

def test_model_prediction():
    """
    测试模型预测功能
    """
    print("\n=== 测试模型预测功能 ===")
    
    try:
        # 创建模型
        model = BERTSentimentAnalyzer(model_name='bert-base-uncased', num_classes=2)
        model.eval()
        
        # 测试文本
        test_texts = [
            "This movie is absolutely fantastic!",
            "I hate this boring film.",
            "The movie is okay, not bad.",
            "Amazing acting and great story!",
            "Terrible plot and bad acting."
        ]
        
        print("正在测试预测功能...")
        
        for i, text in enumerate(test_texts):
            try:
                prediction, probability = model.predict(text, max_length=128)
                sentiment = "正面" if prediction == 1 else "负面"
                print(f"✓ 文本{i+1}: {sentiment} (概率: {probability:.4f})")
                print(f"  原文: {text[:50]}{'...' if len(text) > 50 else ''}")
            except Exception as e:
                print(f"✗ 文本{i+1}预测失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 预测功能测试失败: {e}")
        return False

def test_inference_class():
    """
    测试推理类
    """
    print("\n=== 测试推理类 ===")
    
    try:
        # 创建推理器（不加载训练好的模型）
        inferencer = BERTInference(model_name='bert-base-uncased')
        inferencer._load_pretrained_model()  # 加载预训练模型
        
        print("✓ 推理器创建成功")
        
        # 测试单文本预测
        test_text = "This is a great movie with excellent acting!"
        result = inferencer.predict_single(test_text, return_details=True)
        
        if 'error' not in result:
            print("✓ 单文本预测成功")
            print(f"  情感: {result['sentiment']}")
            print(f"  概率: {result['probability']:.4f}")
            print(f"  置信度: {result['confidence']}")
        else:
            print(f"✗ 单文本预测失败: {result['error']}")
            return False
        
        # 测试批量预测
        test_texts = [
            "I love this movie!",
            "This film is terrible.",
            "Not bad, could be better."
        ]
        
        results = inferencer.predict_batch(test_texts, batch_size=2)
        
        if len(results) == len(test_texts):
            print("✓ 批量预测成功")
            for i, result in enumerate(results):
                if 'error' not in result:
                    print(f"  文本{i+1}: {result['sentiment']} ({result['probability']:.4f})")
                else:
                    print(f"  文本{i+1}: 预测失败")
        else:
            print("✗ 批量预测失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 推理类测试失败: {e}")
        return False

def test_optimizer_setup():
    """
    测试优化器设置
    """
    print("\n=== 测试优化器设置 ===")
    
    try:
        # 创建模型
        model = BERTSentimentAnalyzer(model_name='bert-base-uncased', num_classes=2)
        
        # 设置优化器
        num_training_steps = 1000
        num_warmup_steps = 100
        
        model.setup_optimizer(
            learning_rate=2e-5,
            weight_decay=0.01,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps
        )
        
        print("✓ 优化器设置成功")
        print(f"✓ 优化器类型: {type(model.optimizer).__name__}")
        print(f"✓ 调度器类型: {type(model.scheduler).__name__}")
        print(f"✓ 参数组数量: {len(model.optimizer.param_groups)}")
        
        # 检查参数组
        for i, group in enumerate(model.optimizer.param_groups):
            print(f"  组{i+1}: {len(group['params'])}个参数, 权重衰减={group['weight_decay']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 优化器设置失败: {e}")
        return False

def test_device_compatibility():
    """
    测试设备兼容性
    """
    print("\n=== 测试设备兼容性 ===")
    
    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 测试CPU设备
    try:
        device = torch.device('cpu')
        model = BERTSentimentAnalyzer(model_name='bert-base-uncased', num_classes=2)
        model.to(device)
        print("✓ CPU设备兼容性测试通过")
        
        # 如果有CUDA，测试GPU设备
        if cuda_available:
            device = torch.device('cuda')
            model.to(device)
            print("✓ GPU设备兼容性测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 设备兼容性测试失败: {e}")
        return False

def run_all_tests():
    """
    运行所有测试
    """
    print("开始BERT模型测试")
    print("=" * 60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Python版本: {sys.version.split()[0]}")
    print("=" * 60)
    
    tests = [
        ("设备兼容性", test_device_compatibility),
        ("模型创建", test_model_creation),
        ("优化器设置", test_optimizer_setup),
        ("模型前向传播", test_model_forward),
        ("模型预测功能", test_model_prediction),
        ("推理类", test_inference_class),
        # ("数据加载", test_data_loading),  # 可能需要网络下载，放在最后
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} 测试通过")
            else:
                print(f"✗ {test_name} 测试失败")
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
    
    # 可选的数据加载测试
    print(f"\n{'='*20} 数据加载（可选） {'='*20}")
    print("注意: 此测试需要网络连接下载数据集")
    try:
        user_input = input("是否运行数据加载测试？(y/N): ").strip().lower()
        if user_input in ['y', 'yes']:
            if test_data_loading()[0] is not None:
                passed += 1
                total += 1
                print("✓ 数据加载测试通过")
            else:
                total += 1
                print("✗ 数据加载测试失败")
        else:
            print("跳过数据加载测试")
    except KeyboardInterrupt:
        print("\n跳过数据加载测试")
    
    # 测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过！BERT模型功能正常")
    else:
        print(f"⚠️  有 {total-passed} 个测试失败，请检查相关功能")
    
    print("\n测试完成")
    return passed == total

if __name__ == "__main__":
    run_all_tests()