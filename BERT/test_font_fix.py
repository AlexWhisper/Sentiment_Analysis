#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试matplotlib中文字体修复
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置matplotlib支持中文
# 使用支持中文的字体，优先使用系统可用的字体
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 检测系统是否支持中文字体
def check_chinese_font_support():
    """
    检测系统是否支持中文字体显示
    """
    try:
        import matplotlib.font_manager as fm
        # 获取系统中所有可用的字体
        font_list = [f.name for f in fm.fontManager.ttflist]
        
        # 检查是否有支持中文的字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 
                        'Noto Sans CJK', 'Source Han Sans', 'PingFang SC', 'Hiragino Sans GB']
        
        for font in chinese_fonts:
            if font in font_list:
                print(f"Found Chinese font: {font}")
                return True
        
        # 如果没有找到已知的中文字体，尝试创建一个简单的测试
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试', fontsize=12)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error checking Chinese font support: {e}")
        return False

USE_CHINESE = check_chinese_font_support()

# 根据字体支持情况配置matplotlib
if not USE_CHINESE:
    # 如果不支持中文字体，使用默认的英文字体
    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    print("Warning: Chinese fonts not available, using English labels for plots.")
else:
    print("Chinese font support detected, using Chinese labels for plots.")

def test_plot():
    """
    测试绘图功能
    """
    # 根据字体可用性选择标签语言
    if USE_CHINESE:
        labels = {
            'title': '测试图表',
            'xlabel': '时间',
            'ylabel': '数值',
            'legend1': '训练数据',
            'legend2': '验证数据'
        }
    else:
        labels = {
            'title': 'Test Chart',
            'xlabel': 'Time',
            'ylabel': 'Value',
            'legend1': 'Training Data',
            'legend2': 'Validation Data'
        }
    
    # 创建测试数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, 'b-', label=labels['legend1'], linewidth=2)
    plt.plot(x, y2, 'r-', label=labels['legend2'], linewidth=2)
    plt.title(labels['title'], fontsize=14, fontweight='bold')
    plt.xlabel(labels['xlabel'], fontsize=12)
    plt.ylabel(labels['ylabel'], fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    save_path = 'font_test_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if USE_CHINESE:
        print(f"测试图表已保存到: {save_path}")
    else:
        print(f"Test plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Testing matplotlib font configuration...")
    test_plot()
    print("Test completed!")