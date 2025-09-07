#!/usr/bin/env python3
"""
粉红噪声频谱分析与音箱EQ调节建议工具 - 完全修复版
功能：音频分析、智能EQ调节、实时预览、频率极限分析、5频段音箱问题诊断
版本：3.0 - 完全重构EQ算法，修复所有已知问题
"""

import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from scipy import signal
from werkzeug.utils import secure_filename
import tempfile
import shutil
import sys
import json
import webbrowser
import threading
import time
import math

# 获取脚本所在目录，确保相对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)  # 允许跨域请求

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class AdvancedPinkNoiseAnalyzer:
    """
    高级粉红噪声分析器 - 完全修复版
    特点：正确的EQ算法、频率下潜优化、平滑度计算修复、参数传递完善
    """
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        
        # 标准15段EQ频率点 (Hz) - 精确Q值
        self.eq_15_freqs = [25, 40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 16000]
        self.eq_15_q_factor = 2.15
        
        # 标准31段EQ频率点 (Hz) - 精确Q值
        self.eq_31_freqs = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 
                           630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 
                           8000, 10000, 12500, 16000, 20000]
        self.eq_31_q_factor = 4.31
        
        # 5频段分析定义
        self.frequency_bands = {
            'sub_bass': {'range': (20, 60), 'name': '超低频', 'critical_freqs': [25, 31.5, 40, 50]},
            'bass': {'range': (60, 250), 'name': '低频', 'critical_freqs': [63, 80, 100, 125, 160, 200]},
            'midrange': {'range': (250, 2000), 'name': '中频', 'critical_freqs': [250, 315, 400, 500, 630, 800, 1000, 1250, 1600]},
            'treble': {'range': (2000, 8000), 'name': '高频', 'critical_freqs': [2000, 2500, 3150, 4000, 5000, 6300]},
            'air': {'range': (8000, 20000), 'name': '超高频', 'critical_freqs': [8000, 10000, 12500, 16000, 20000]}
        }
    
    def compute_spectrum(self, audio_data, nperseg=4096):
        """计算音频频谱 - 改进版"""
        try:
            # 使用Welch方法计算功率谱密度
            freqs, psd = signal.welch(
                audio_data, 
                self.sample_rate, 
                nperseg=nperseg, 
                noverlap=nperseg//2,  
                window='hann',  
                scaling='density',
                detrend='constant'
            )
            
            # 转换为dB，处理有效频率范围 (20Hz - 20kHz)
            valid_mask = (freqs >= 20) & (freqs <= 20000)
            freqs = freqs[valid_mask]
            psd = psd[valid_mask]
            
            # 避免log10(0)的问题，设置最小值
            psd_safe = np.where(psd > 1e-20, psd, 1e-20)
            psd_db = 10 * np.log10(psd_safe)
            
            return freqs, psd_db
        except Exception as e:
            print(f"计算频谱时出错: {str(e)}")
            raise
    
    def get_theoretical_pink_noise_spectrum(self, freqs):
        """获取理论粉红噪声频谱 - 优化版"""
        try:
            ref_freq = 1000.0
            ref_level = 0.0  # 1kHz处设为0dB参考
            
            # 避免频率为0的情况
            valid_freqs = np.where(freqs > 1, freqs, 1)
            
            # 粉红噪声理论公式：-3dB/octave
            pink_spectrum = ref_level - 3.0 * np.log2(valid_freqs / ref_freq)
            
            # 处理边界情况
            if len(freqs) > 1:
                pink_spectrum[0] = pink_spectrum[1] if freqs[0] == 0 else pink_spectrum[0]
            
            # 对极端频率进行合理限制
            low_freq_limit = 25   # 改为25Hz，更合理的低频下限
            high_freq_limit = 16000  # 保持16kHz高频限制
            
            low_mask = freqs < low_freq_limit
            high_mask = freqs > high_freq_limit
            
            if np.any(low_mask):
                low_ref_idx = np.argmin(np.abs(freqs - low_freq_limit))
                pink_spectrum[low_mask] = pink_spectrum[low_ref_idx]
                
            if np.any(high_mask):
                high_ref_idx = np.argmin(np.abs(freqs - high_freq_limit))
                # 高频自然衰减
                freq_ratio = freqs[high_mask] / high_freq_limit
                additional_rolloff = -6 * np.log2(freq_ratio)  # 额外6dB/octave衰减
                pink_spectrum[high_mask] = pink_spectrum[high_ref_idx] + additional_rolloff
            
            return pink_spectrum
        except Exception as e:
            print(f"生成理论频谱时出错: {str(e)}")
            raise
    
    def adaptive_smooth_spectrum(self, freqs, spectrum, base_octave_fraction=4):
        """自适应频谱平滑 - 根据频段特性调整平滑强度"""
        try:
            smoothed = np.zeros_like(spectrum)
            
            for i, freq in enumerate(freqs):
                if freq <= 20:
                    smoothed[i] = spectrum[i]
                    continue
                
                # 根据频段调整平滑参数
                if freq < 100:
                    # 低频：更细致的平滑
                    octave_fraction = base_octave_fraction * 1.2
                elif freq < 1000:
                    # 中低频：标准平滑
                    octave_fraction = base_octave_fraction
                elif freq < 4000:
                    # 中高频：保持细节
                    octave_fraction = base_octave_fraction * 0.8
                else:
                    # 高频：更强平滑
                    octave_fraction = base_octave_fraction * 0.6
                
                f_low = freq / (2**(1/(2*octave_fraction)))
                f_high = freq * (2**(1/(2*octave_fraction)))
                
                mask = (freqs >= f_low) & (freqs <= f_high)
                if np.sum(mask) > 0:
                    # 加权平均，中心频率权重更高
                    weights = np.exp(-((freqs[mask] - freq) / freq)**2 * 2)
                    smoothed[i] = np.average(spectrum[mask], weights=weights)
                else:
                    smoothed[i] = spectrum[i]
            
            return smoothed
        except Exception as e:
            print(f"平滑频谱时出错: {str(e)}")
            return spectrum
    
    def normalize_spectrum(self, freqs, spectrum, reference_freq=1000):
        """标准化频谱 - 在指定频率处对齐到0dB"""
        try:
            ref_idx = np.argmin(np.abs(freqs - reference_freq))
            offset = -spectrum[ref_idx]
            normalized_spectrum = spectrum + offset
            
            print(f"频谱标准化: 在{reference_freq}Hz处偏移{offset:.1f}dB")
            return normalized_spectrum
        except Exception as e:
            print(f"标准化频谱时出错: {str(e)}")
            return spectrum
    
    def calculate_biquad_peaking_response(self, frequency, center_freq, Q, gain_db):
        """
        精确计算双二阶峰值滤波器的频率响应
        使用Web Audio API标准实现，与前端完全一致
        """
        try:
            # 基础验证
            if not (math.isfinite(frequency) and math.isfinite(center_freq) and 
                    math.isfinite(Q) and math.isfinite(gain_db)):
                return 0.0
            
            if frequency <= 0 or center_freq <= 0 or Q <= 0.01:
                return 0.0
            
            # 限制增益范围
            gain_db = max(-60, min(60, gain_db))
            
            # 如果增益很小，直接返回0
            if abs(gain_db) < 0.01:
                return 0.0
            
            # Web Audio API标准实现
            A = 10 ** (gain_db / 40)  # 幅度增益因子
            
            # 采样率44.1kHz（与前端一致）
            sample_rate = self.sample_rate 
            w0 = 2 * math.pi * center_freq / sample_rate
            cos_w0 = math.cos(w0)
            sin_w0 = math.sin(w0)
            alpha = sin_w0 / (2 * Q)
            
            # 验证中间参数
            if not (math.isfinite(A) and math.isfinite(w0) and math.isfinite(alpha)):
                return 0.0
            
            # 峰值滤波器系数
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A
            
            # 验证系数
            if not (math.isfinite(b0) and math.isfinite(b1) and math.isfinite(b2) and 
                    math.isfinite(a0) and math.isfinite(a1) and math.isfinite(a2)):
                return 0.0
            
            if abs(a0) < 1e-10:
                return 0.0
            
            # 归一化系数
            norm_b0 = b0 / a0
            norm_b1 = b1 / a0
            norm_b2 = b2 / a0
            norm_a1 = a1 / a0
            norm_a2 = a2 / a0
            
            # 计算目标频率的频率响应
            w = 2 * math.pi * frequency / sample_rate
            cos_w = math.cos(w)
            sin_w = math.sin(w)
            
            # 验证目标频率参数
            if not (math.isfinite(w) and math.isfinite(cos_w) and math.isfinite(sin_w)):
                return 0.0
            
            # 复数形式计算频率响应
            numerator_real = norm_b0 + norm_b1 * cos_w + norm_b2 * math.cos(2*w)
            numerator_imag = -norm_b1 * sin_w - norm_b2 * math.sin(2*w)
            
            denominator_real = 1 + norm_a1 * cos_w + norm_a2 * math.cos(2*w)
            denominator_imag = -norm_a1 * sin_w - norm_a2 * math.sin(2*w)
            
            # 验证复数部分
            if not (math.isfinite(numerator_real) and math.isfinite(numerator_imag) and 
                    math.isfinite(denominator_real) and math.isfinite(denominator_imag)):
                return 0.0
            
            # 计算幅度
            numerator_magnitude = math.sqrt(numerator_real**2 + numerator_imag**2)
            denominator_magnitude = math.sqrt(denominator_real**2 + denominator_imag**2)
            
            if not (math.isfinite(numerator_magnitude) and math.isfinite(denominator_magnitude)):
                return 0.0
            
            if denominator_magnitude <= 0 or numerator_magnitude <= 0:
                return 0.0
            
            magnitude = numerator_magnitude / denominator_magnitude
            
            if not math.isfinite(magnitude) or magnitude <= 0:
                return 0.0
            
            # 转换为dB
            response_db = 20 * math.log10(magnitude)
            
            if not math.isfinite(response_db):
                return 0.0
            
            return max(-60, min(60, response_db))
            
        except Exception as e:
            print(f'滤波器响应计算错误: {str(e)}')
            return 0.0
    
    def apply_eq_precise(self, freqs, spectrum, eq_freqs, eq_gains, is_31_band=False):
        """
        精确的EQ应用函数 - 使用双二阶滤波器响应
        与前端JavaScript版本完全一致
        """
        try:
            if not len(freqs) == len(spectrum):
                raise ValueError("频率数组和频谱数组长度不匹配")
            
            q_factor = self.eq_31_q_factor if is_31_band else self.eq_15_q_factor
            eq_applied_spectrum = spectrum.copy()
            
            for i, freq in enumerate(freqs):
                if freq <= 0 or not math.isfinite(spectrum[i]):
                    continue
                
                total_response_db = 0.0
                
                # 计算所有EQ频段的累积效果
                for eq_freq, gain in zip(eq_freqs, eq_gains):
                    if abs(gain) >= 0.1:  # 只计算有显著增益的滤波器
                        response_db = self.calculate_biquad_peaking_response(
                            freq, eq_freq, q_factor, gain
                        )
                        if math.isfinite(response_db):
                            total_response_db += response_db
                
                # 应用总的EQ响应
                if math.isfinite(total_response_db):
                    eq_applied_spectrum[i] = spectrum[i] + total_response_db
            
            return eq_applied_spectrum, np.zeros_like(freqs)
            
        except Exception as e:
            print(f'精确EQ应用时出错: {str(e)}')
            return spectrum, np.zeros_like(freqs)
    
    def calculate_frequency_limits(self, freqs, spectrum, threshold_db=-6):
        """计算频率极限（与前端JavaScript算法保持一致）"""
        try:
            # 1. 找到1kHz参考点
            ref_frequency = 1000
            ref_idx = np.argmin(np.abs(freqs - ref_frequency))
            ref_level = spectrum[ref_idx]
            threshold = ref_level + threshold_db  # -6dB阈值
            
            # 2. 计算低频极限
            low_limit = None
            # 遍历所有低于1kHz的频率点（从低到高检查）
            for i in range(len(freqs)):
                current_freq = freqs[i]
                current_level = spectrum[i]
                
                # 只处理1kHz以下的频率
                if current_freq >= ref_frequency:
                    break
                
                # 条件1：电平高于阈值
                if current_level < threshold:
                    continue
                
                # 条件2：左侧所有点的电平均低于当前点
                is_left_all_lower = True
                for j in range(i):
                    if spectrum[j] >= current_level:
                        is_left_all_lower = False
                        break
                
                if is_left_all_lower:
                    low_limit = current_freq
                    break  # 取第一个符合条件的最低频率点
            
            # 3. 计算高频极限
            high_limit = None
            # 从1kHz参考点开始向右检查
            for i in range(ref_idx, len(freqs)):
                current_freq = freqs[i]
                current_level = spectrum[i]
                
                if current_level < threshold:
                    # 取前一个点作为高频极限
                    high_limit = freqs[i-1] if i > 0 else None
                    break
            
            # 处理所有高频点都高于阈值的情况
            if high_limit is None and len(freqs) > 0:
                max_freq = freqs[-1]
                high_limit = 20000 if max_freq >= 18000 else max_freq
            
            return low_limit, high_limit
        except Exception as e:
            print(f"计算频率极限时出错: {str(e)}")
            return None, None

    
    def analyze_speaker_issues_5_bands(self, freqs, spectrum_diff):
        """
        5频段音箱问题分析 - 优化版
        分析超低频、低频、中频、高频、超高频的具体问题
        """
        issues = []
        
        try:
            # 计算各频段的平均偏差和变化特征
            band_analysis = {}
            
            for band_key, band_info in self.frequency_bands.items():
                freq_min, freq_max = band_info['range']
                band_mask = (freqs >= freq_min) & (freqs <= freq_max)
                
                if np.any(band_mask):
                    band_spectrum = spectrum_diff[band_mask]
                    band_freqs = freqs[band_mask]
                    
                    # 基础统计
                    avg_level = np.mean(band_spectrum)
                    std_level = np.std(band_spectrum)
                    min_level = np.min(band_spectrum)
                    max_level = np.max(band_spectrum)
                    
                    # 计算频段内的斜率（趋势）
                    if len(band_freqs) > 3:
                        # 线性拟合计算斜率
                        log_freqs = np.log10(band_freqs)
                        slope, _ = np.polyfit(log_freqs, band_spectrum, 1)
                    else:
                        slope = 0
                    
                    band_analysis[band_key] = {
                        'name': band_info['name'],
                        'avg_level': avg_level,
                        'std_level': std_level,
                        'min_level': min_level,
                        'max_level': max_level,
                        'slope': slope,
                        'range_span': max_level - min_level
                    }
                else:
                    # 频段无数据
                    band_analysis[band_key] = {
                        'name': band_info['name'],
                        'avg_level': 0,
                        'std_level': 0,
                        'min_level': 0,
                        'max_level': 0,
                        'slope': 0,
                        'range_span': 0
                    }
            
            # 分析各频段的具体问题
            self._analyze_sub_bass_issues(band_analysis['sub_bass'], issues)
            self._analyze_bass_issues(band_analysis['bass'], issues)
            self._analyze_midrange_issues(band_analysis['midrange'], issues)
            self._analyze_treble_issues(band_analysis['treble'], issues)
            self._analyze_air_issues(band_analysis['air'], issues)
            
            # 分析整体平衡性
            self._analyze_overall_balance(band_analysis, issues)
            
            # 分析驻波和共振问题
            overall_variation = np.std(spectrum_diff)
            if overall_variation > 5:
                issues.append({
                    'severity': 'high',
                    'band': '全频段',
                    'issue': '严重频响波动',
                    'description': f'整体频响波动过大（标准差: {overall_variation:.1f}dB），存在明显驻波或房间声学问题',
                    'suggestion': '检查音箱摆位，调整听音位置，考虑添加低频陷阱和扩散板',
                    'technical_details': f'频响标准差超过5dB阈值，建议声学处理'
                })
            elif overall_variation > 3:
                issues.append({
                    'severity': 'medium',
                    'band': '全频段',
                    'issue': '中等频响不均',
                    'description': f'频响曲线有一定波动（标准差: {overall_variation:.1f}dB）',
                    'suggestion': '尝试调整音箱摆位或听音位置，可考虑轻微声学处理',
                    'technical_details': f'频响标准差{overall_variation:.1f}dB，在可接受范围内但可优化'
                })
            
            # 如果没有发现明显问题
            if not issues:
                issues.append({
                    'severity': 'good',
                    'band': '全频段',
                    'issue': '音箱表现良好',
                    'description': '各频段表现相对均衡，无明显频响缺陷',
                    'suggestion': '音箱频响特性良好，可根据个人喜好进行微调',
                    'technical_details': '所有频段均在正常范围内'
                })
                
        except Exception as e:
            print(f"分析音箱问题时出错: {str(e)}")
            issues.append({
                'severity': 'error',
                'band': '系统',
                'issue': '分析失败',
                'description': '无法完成5频段音箱问题分析',
                'suggestion': '请检查音频文件质量或重新录制',
                'technical_details': f'错误信息: {str(e)}'
            })
        
        return issues
    
    def _analyze_sub_bass_issues(self, band_data, issues):
        """分析超低频问题 (20-60Hz)"""
        avg = band_data['avg_level']
        std = band_data['std_level']
        
        if avg < -8:
            issues.append({
                'severity': 'high',
                'band': '超低频',
                'issue': '超低频严重缺失',
                'description': f'超低频段（20-60Hz）平均衰减 {abs(avg):.1f}dB，低频延伸不足',
                'suggestion': '考虑添加低音炮，或检查音箱是否支持超低频输出',
                'technical_details': f'平均电平: {avg:.1f}dB'
            })
        elif avg < -4:
            issues.append({
                'severity': 'medium',
                'band': '超低频',
                'issue': '超低频不足',
                'description': f'超低频段（20-60Hz）平均衰减 {abs(avg):.1f}dB',
                'suggestion': '适当提升低频EQ，或调整音箱与墙面距离',
                'technical_details': f'平均电平: {avg:.1f}dB，可通过EQ适度补偿'
            })
        
        if std > 4:
            issues.append({
                'severity': 'medium',
                'band': '超低频',
                'issue': '超低频不均匀',
                'description': f'超低频段波动较大（标准差: {std:.1f}dB），可能存在房间模态',
                'suggestion': '调整音箱摆位，避开房间驻波点',
                'technical_details': f'频响标准差: {std:.1f}dB，建议位置调整'
            })
    
    def _analyze_bass_issues(self, band_data, issues):
        """分析低频问题 (60-250Hz)"""
        avg = band_data['avg_level']
        std = band_data['std_level']
        range_span = band_data['range_span']
        
        if avg < -6:
            issues.append({
                'severity': 'high',
                'band': '低频',
                'issue': '低频严重不足',
                'description': f'低频段（60-250Hz）平均衰减 {abs(avg):.1f}dB，影响音乐厚度',
                'suggestion': '检查低频扬声器状态，提升低频EQ或增强低频驱动',
                'technical_details': f'平均电平: {avg:.1f}dB，需要显著补偿'
            })
        elif avg < -3:
            issues.append({
                'severity': 'medium',
                'band': '低频',
                'issue': '低频偏弱',
                'description': f'低频段（60-250Hz）平均衰减 {abs(avg):.1f}dB',
                'suggestion': '适当提升低频EQ，增强音乐的力度感',
                'technical_details': f'平均电平: {avg:.1f}dB，建议+2到+4dB补偿'
            })
        elif avg > 4:
            issues.append({
                'severity': 'medium',
                'band': '低频',
                'issue': '低频过量',
                'description': f'低频段（60-250Hz）平均提升 {avg:.1f}dB，声音可能浑浊',
                'suggestion': '适当衰减低频EQ，检查是否靠墙太近造成低频增强',
                'technical_details': f'平均电平: {avg:.1f}dB，建议-2到-3dB衰减'
            })
        
        if range_span > 8:
            issues.append({
                'severity': 'medium',
                'band': '低频',
                'issue': '低频起伏过大',
                'description': f'低频段内变化幅度达 {range_span:.1f}dB，频响不够平直',
                'suggestion': '使用多点EQ调节平滑低频响应，或改善房间声学',
                'technical_details': f'频段内最大变化: {range_span:.1f}dB'
            })
    
    def _analyze_midrange_issues(self, band_data, issues):
        """分析中频问题 (250-2000Hz) - 最关键频段"""
        avg = band_data['avg_level']
        std = band_data['std_level']
        
        if avg < -4:
            issues.append({
                'severity': 'high',
                'band': '中频',
                'issue': '中频明显凹陷',
                'description': f'中频段（250-2000Hz）平均衰减 {abs(avg):.1f}dB，严重影响人声清晰度',
                'suggestion': '检查中频扬声器状态，重点提升中频EQ',
                'technical_details': f'平均电平: {avg:.1f}dB，中频是最关键的听音频段'
            })
        elif avg < -2:
            issues.append({
                'severity': 'medium',
                'band': '中频',
                'issue': '中频偏弱',
                'description': f'中频段（250-2000Hz）平均衰减 {abs(avg):.1f}dB',
                'suggestion': '适当提升中频EQ，改善人声表现',
                'technical_details': f'平均电平: {avg:.1f}dB，建议+1到+3dB补偿'
            })
        elif avg > 4:
            issues.append({
                'severity': 'medium',
                'band': '中频',
                'issue': '中频过于突出',
                'description': f'中频段（250-2000Hz）平均提升 {avg:.1f}dB，可能导致听觉疲劳',
                'suggestion': '适当衰减中频EQ，避免声音过于尖锐',
                'technical_details': f'平均电平: {avg:.1f}dB，建议-2到-3dB衰减'
            })
        
        if std > 3:
            issues.append({
                'severity': 'medium',
                'band': '中频',
                'issue': '中频不够平滑',
                'description': f'中频段波动较大（标准差: {std:.1f}dB），影响音色一致性',
                'suggestion': '使用多段EQ精细调节中频响应曲线',
                'technical_details': f'频响标准差: {std:.1f}dB，中频需要更平滑的响应'
            })
    
    def _analyze_treble_issues(self, band_data, issues):
        """分析高频问题 (2000-8000Hz)"""
        avg = band_data['avg_level']
        std = band_data['std_level']
        slope = band_data['slope']
        
        if avg < -6:
            issues.append({
                'severity': 'high',
                'band': '高频',
                'issue': '高频严重衰减',
                'description': f'高频段（2-8kHz）平均衰减 {abs(avg):.1f}dB，声音缺乏明亮度',
                'suggestion': '检查高音扬声器状态，可能存在老化或损坏',
                'technical_details': f'平均电平: {avg:.1f}dB，高频衰减过多'
            })
        elif avg < -3:
            issues.append({
                'severity': 'medium',
                'band': '高频',
                'issue': '高频偏暗',
                'description': f'高频段（2-8kHz）平均衰减 {abs(avg):.1f}dB',
                'suggestion': '适当提升高频EQ，增加声音明亮度和细节',
                'technical_details': f'平均电平: {avg:.1f}dB，建议+2到+4dB补偿'
            })
        elif avg > 4:
            issues.append({
                'severity': 'medium',
                'band': '高频',
                'issue': '高频过亮',
                'description': f'高频段（2-8kHz）平均提升 {avg:.1f}dB，可能导致刺耳感',
                'suggestion': '适当衰减高频EQ，避免听觉疲劳',
                'technical_details': f'平均电平: {avg:.1f}dB，建议-2到-3dB衰减'
            })
        
        # 分析高频斜率
        if slope < -15:
            issues.append({
                'severity': 'medium',
                'band': '高频',
                'issue': '高频过度下降',
                'description': f'高频段下降过于陡峭（{slope:.1f}dB/decade）',
                'suggestion': '检查高音扬声器频响特性，考虑使用高频提升EQ',
                'technical_details': f'频率斜率: {slope:.1f}dB/decade'
            })
    
    def _analyze_air_issues(self, band_data, issues):
        """分析超高频问题 (8000-20000Hz)"""
        avg = band_data['avg_level']
        
        if avg < -10:
            issues.append({
                'severity': 'low',
                'band': '超高频',
                'issue': '超高频大幅衰减',
                'description': f'超高频段（8-20kHz）平均衰减 {abs(avg):.1f}dB',
                'suggestion': '超高频自然衰减是正常现象，可适度提升增加空气感',
                'technical_details': f'平均电平: {avg:.1f}dB，大部分音箱在此频段都会衰减'
            })
        elif avg > 2:
            issues.append({
                'severity': 'low',
                'band': '超高频',
                'issue': '超高频偏亮',
                'description': f'超高频段（8-20kHz）平均提升 {avg:.1f}dB',
                'suggestion': '超高频过多可能产生噪音感，可适度衰减',
                'technical_details': f'平均电平: {avg:.1f}dB，建议保持自然衰减'
            })
    
    def _analyze_overall_balance(self, band_analysis, issues):
        """分析整体频段平衡性"""
        # 计算频段间的平衡性
        bass_level = band_analysis['bass']['avg_level']
        mid_level = band_analysis['midrange']['avg_level']
        treble_level = band_analysis['treble']['avg_level']
        
        # 低频-中频平衡
        bass_mid_diff = bass_level - mid_level
        if bass_mid_diff > 4:
            issues.append({
                'severity': 'medium',
                'band': '整体平衡',
                'issue': '低频过于突出',
                'description': f'低频相对中频高出 {bass_mid_diff:.1f}dB，声音偏厚重',
                'suggestion': '适当衰减低频或提升中频，改善整体平衡',
                'technical_details': f'低频-中频差值: {bass_mid_diff:.1f}dB'
            })
        elif bass_mid_diff < -4:
            issues.append({
                'severity': 'medium',
                'band': '整体平衡',
                'issue': '低频相对不足',
                'description': f'低频相对中频低了 {abs(bass_mid_diff):.1f}dB，声音偏薄',
                'suggestion': '适当提升低频或衰减中频，增加声音厚度',
                'technical_details': f'低频-中频差值: {bass_mid_diff:.1f}dB'
            })
        
        # 中频-高频平衡
        mid_treble_diff = mid_level - treble_level
        if mid_treble_diff > 4:
            issues.append({
                'severity': 'medium',
                'band': '整体平衡',
                'issue': '声音偏暗',
                'description': f'中频相对高频高出 {mid_treble_diff:.1f}dB，缺乏明亮度',
                'suggestion': '适当提升高频或衰减中频，增加声音明亮度',
                'technical_details': f'中频-高频差值: {mid_treble_diff:.1f}dB'
            })
        elif mid_treble_diff < -4:
            issues.append({
                'severity': 'medium',
                'band': '整体平衡',
                'issue': '声音偏亮',
                'description': f'高频相对中频高出 {abs(mid_treble_diff):.1f}dB，可能过于刺激',
                'suggestion': '适当衰减高频或提升中频，改善听感平衡',
                'technical_details': f'中频-高频差值: {mid_treble_diff:.1f}dB'
            })
    
    # ========== 全新的EQ算法部分 - 完全重构 ==========
    
    def find_target_eq_bands(self, eq_freqs, target_freq, tolerance_ratio=0.8):
        """
        找到最接近目标频率的EQ频段
        tolerance_ratio: 容忍度比例，0.8表示目标频率±20%范围内的频段
        """
        target_bands = []
        for i, eq_freq in enumerate(eq_freqs):
            freq_ratio = max(eq_freq / target_freq, target_freq / eq_freq)
            if freq_ratio <= (1 + tolerance_ratio):
                distance = abs(eq_freq - target_freq)
                target_bands.append((i, eq_freq, distance))
        
        # 按距离排序，最接近的在前
        target_bands.sort(key=lambda x: x[2])
        return target_bands
    
    def calculate_extension_strategy(self, target_freq, eq_freqs, measured_spectrum, target_spectrum, freqs):
        """
        计算频率延伸策略
        """
        target_bands = self.find_target_eq_bands(eq_freqs, target_freq)
        
        strategy = {
            'primary_bands': [],      # 主要增强频段
            'support_bands': [],      # 辅助增强频段
            'control_bands': [],      # 需要控制的频段
            'target_freq': target_freq
        }
        
        if not target_bands:
            return strategy
        
        # 计算目标频率处的实际需求
        target_idx = np.argmin(np.abs(freqs - target_freq))
        target_correction = target_spectrum[target_idx] - measured_spectrum[target_idx]
        
        for i, (band_idx, eq_freq, distance) in enumerate(target_bands[:3]):  # 只考虑最接近的3个频段
            if distance < target_freq * 0.3:  # 30%范围内为主要频段
                strategy['primary_bands'].append({
                    'index': band_idx,
                    'freq': eq_freq,
                    'base_gain': target_correction * (1.5 if i == 0 else 1.2),  # 最接近的给最大增益
                    'distance': distance
                })
            else:
                strategy['support_bands'].append({
                    'index': band_idx,
                    'freq': eq_freq,
                    'base_gain': target_correction * 0.8,
                    'distance': distance
                })
        
        # 找到需要控制的频段（远离目标频率且可能干扰的）
        for i, eq_freq in enumerate(eq_freqs):
            if eq_freq < target_freq * 0.5:  # 远低于目标频率
                strategy['control_bands'].append({
                    'index': i,
                    'freq': eq_freq,
                    'action': 'attenuate',  # 衰减
                    'strength': -2 if eq_freq < target_freq * 0.3 else -1
                })
        
        return strategy
    
    def apply_extension_strategy(self, eq_values, strategy):
        """
        应用频率延伸策略到EQ设置
        """
        modified_eq = eq_values.copy()
        
        # 应用主要增强频段
        for band in strategy['primary_bands']:
            idx = band['index']
            gain = np.clip(band['base_gain'], -12, 12)
            modified_eq[idx] = gain
            print(f"  主要增强: {band['freq']:.1f}Hz = {gain:.1f}dB")
        
        # 应用辅助增强频段
        for band in strategy['support_bands']:
            idx = band['index']
            gain = np.clip(band['base_gain'], -12, 12)
            modified_eq[idx] = gain
            print(f"  辅助增强: {band['freq']:.1f}Hz = {gain:.1f}dB")
        
        # 应用控制频段
        for band in strategy['control_bands']:
            idx = band['index']
            if band['action'] == 'attenuate':
                modified_eq[idx] = np.clip(modified_eq[idx] + band['strength'], -12, 12)
                print(f"  频率控制: {band['freq']:.1f}Hz = {modified_eq[idx]:.1f}dB")
        
        return modified_eq
    
    def evaluate_extension_performance(self, freqs, corrected_spectrum, target_freq):
        """
        评估频率延伸性能
        """
        try:
            # 找到1kHz和目标频率的索引
            ref_1k_idx = np.argmin(np.abs(freqs - 1000))
            target_idx = np.argmin(np.abs(freqs - target_freq))
            
            # 计算相对衰减
            ref_level = corrected_spectrum[ref_1k_idx]
            target_level = corrected_spectrum[target_idx]
            actual_attenuation = ref_level - target_level
            
            # 理想的粉红噪声衰减
            ideal_attenuation = 3 * np.log2(1000 / max(target_freq, 20))
            
            # 延伸性能评分
            extension_error = abs(actual_attenuation - ideal_attenuation)
            
            # 如果实际衰减比理想衰减大太多，说明延伸不足
            if actual_attenuation > ideal_attenuation + 6:
                extension_error *= 2  # 加重惩罚
            
            return {
                'extension_error': extension_error,
                'actual_attenuation': actual_attenuation,
                'ideal_attenuation': ideal_attenuation,
                'target_level': target_level,
                'ref_level': ref_level
            }
            
        except Exception as e:
            print(f"延伸性能评估失败: {str(e)}")
            return {
                'extension_error': 100,
                'actual_attenuation': 0,
                'ideal_attenuation': 0,
                'target_level': 0,
                'ref_level': 0
            }
    
    def calculate_smart_eq_suggestions(self, freqs, measured_spectrum, reference_spectrum, 
                                    target_low_freq=20, target_high_freq=20000, 
                                    enable_midrange_smoothing=True):
        """
        智能EQ建议计算 - 集成中高频平滑优化
        """
        try:
            print(f"开始智能EQ校正: 目标范围{target_low_freq}Hz - {target_high_freq}Hz")
            if enable_midrange_smoothing:
                print("已启用中高频平滑优化")
            
            # 标准化测量频谱
            normalized_measured = self.normalize_spectrum(freqs, measured_spectrum, 1000)
            current_spectrum = normalized_measured.copy()
            
            # 初始化EQ设置
            eq_15_final = np.zeros(len(self.eq_15_freqs))
            eq_31_final = np.zeros(len(self.eq_31_freqs))
            
            # 迭代校正
            iterations = 3
            for iteration in range(iterations):
                print(f"\n=== 第{iteration+1}次迭代校正 ===")
                
                correction_needed = reference_spectrum - current_spectrum
                strength_factor = 0.3 ** iteration
                
                # 15段EQ处理
                eq_15_iteration = np.zeros(len(self.eq_15_freqs))
                for i, eq_freq in enumerate(self.eq_15_freqs):
                    correction_value = self._calculate_conservative_eq_value(
                        eq_freq, freqs, correction_needed, target_low_freq, target_high_freq,
                        current_spectrum, reference_spectrum
                    )
                    eq_15_iteration[i] = correction_value * strength_factor
                    eq_15_final[i] += eq_15_iteration[i]
                
                # 31段EQ处理
                eq_31_iteration = np.zeros(len(self.eq_31_freqs))
                for i, eq_freq in enumerate(self.eq_31_freqs):
                    correction_value = self._calculate_conservative_eq_value(
                        eq_freq, freqs, correction_needed, target_low_freq, target_high_freq,
                        current_spectrum, reference_spectrum
                    )
                    
                    # 低频提升限制
                    if eq_freq <= 100 and correction_value > 0:
                        if eq_31_final[i] > 8:
                            correction_value = min(correction_value, 2.0)
                        elif eq_31_final[i] > 4:
                            correction_value = min(correction_value, 4.0)
                    
                    eq_31_iteration[i] = correction_value * strength_factor
                    eq_31_final[i] += eq_31_iteration[i]
                
                # 应用本次EQ调整
                current_spectrum, _ = self.apply_eq_precise(
                    freqs, current_spectrum, self.eq_15_freqs, eq_15_iteration, is_31_band=False
                )
                
                remaining_error = np.mean(np.abs(reference_spectrum - current_spectrum))
                print(f"第{iteration+1}次迭代后平均误差: {remaining_error:.2f}dB")
                
                if remaining_error < 0.5:
                    print(f"误差已降至{remaining_error:.2f}dB，提前结束迭代")
                    break
            
            # 应用中高频平滑优化（如果启用）
            if enable_midrange_smoothing:
                print(f"\n=== 应用中高频平滑优化 ===")
                
                # 15段EQ平滑处理
                print("处理15段EQ...")
                eq_15_step1 = self.detect_and_smooth_peaks(self.eq_15_freqs, eq_15_final, 
                                                        peak_threshold=3.5, min_freq=400)
                eq_15_step2 = self.apply_midrange_smoothing_advanced(self.eq_15_freqs, eq_15_step1, 400)
                eq_15_final = self.apply_neighbor_relative_limit(self.eq_15_freqs, eq_15_step2, 400)
                
                # 31段EQ平滑处理
                print("处理31段EQ...")
                eq_31_step1 = self.detect_and_smooth_peaks(self.eq_31_freqs, eq_31_final, 
                                                        peak_threshold=3.0, min_freq=400)
                eq_31_step2 = self.apply_midrange_smoothing_advanced(self.eq_31_freqs, eq_31_step1, 400)
                eq_31_final = self.apply_neighbor_relative_limit(self.eq_31_freqs, eq_31_step2, 400)
                
                print("中高频平滑优化完成")
            
            # 最终限制EQ范围
            eq_15_final = np.clip(eq_15_final, -12, 12)
            eq_31_final = np.clip(eq_31_final, -12, 12)
            
            # 统计优化效果
            if enable_midrange_smoothing:
                eq_15_midrange_count = sum(1 for i, freq in enumerate(self.eq_15_freqs) 
                                        if freq >= 400 and abs(eq_15_final[i]) > 1.0)
                eq_31_midrange_count = sum(1 for i, freq in enumerate(self.eq_31_freqs) 
                                        if freq >= 400 and abs(eq_31_final[i]) > 1.0)
                print(f"中高频(≥400Hz)显著调整: 15段={eq_15_midrange_count}个, 31段={eq_31_midrange_count}个")
            
            print(f"迭代校正完成，总计{iteration+1}次迭代")
            
            return eq_15_final, eq_31_final, normalized_measured, [], []
            
        except Exception as e:
            print(f"智能EQ校正失败: {str(e)}")
            return np.zeros(15), np.zeros(31), measured_spectrum, [], []

    def _calculate_conservative_eq_value(self, eq_freq, freqs, correction_needed, target_low_freq, target_high_freq, 
                                    normalized_measured=None, reference_spectrum=None):
        """
        计算保守的EQ值 - 增加详细日志输出
        显示：原始频响 -> EQ校正 -> 预期校正后频响
        """
        # 严格的频率范围控制
        if eq_freq < target_low_freq:
            print(f"  范围外衰减: {eq_freq:.1f}Hz = -12.0dB (低于{target_low_freq}Hz)")
            return -12.0
        elif eq_freq > target_high_freq:
            print(f"  范围外衰减: {eq_freq:.1f}Hz = -12.0dB (高于{target_high_freq}Hz)")
            return -12.0
        
        # 找到对应频率的校正需求
        freq_idx = np.argmin(np.abs(freqs - eq_freq))
        basic_correction = correction_needed[freq_idx]
        
        # 简单的1:1校正
        conservative_correction = np.clip(basic_correction, -12, 12)
        
        # 忽略微小调整
        if abs(conservative_correction) < 0.1:
            conservative_correction = 0.0
            return conservative_correction
        
        # 详细日志输出
        if normalized_measured is not None and reference_spectrum is not None:
            original_level = normalized_measured[freq_idx]          # 原始频响
            target_level = reference_spectrum[freq_idx]             # 目标频响（粉红噪声）
            predicted_after_eq = original_level + conservative_correction  # 预期校正后
            
            print(f"  详细校正 {eq_freq:.1f}Hz: "
                f"原始={original_level:.1f}dB → "
                f"EQ{conservative_correction:+.1f}dB → "
                f"预期={predicted_after_eq:.1f}dB "
                f"(目标={target_level:.1f}dB, 偏差={predicted_after_eq-target_level:.1f}dB)")
        else:
            # 简化日志
            if abs(conservative_correction) > 6:
                print(f"  大幅校正: {eq_freq:.1f}Hz = {conservative_correction:.1f}dB")
            elif abs(conservative_correction) > 2:
                print(f"  中等校正: {eq_freq:.1f}Hz = {conservative_correction:.1f}dB")
            else:
                print(f"  微调: {eq_freq:.1f}Hz = {conservative_correction:.1f}dB")
        
        return conservative_correction

    def _verify_frequency_limits(self, eq_values, eq_freqs, target_low_freq, target_high_freq, eq_type):
        """
        验证频率范围限制是否正确应用
        """
        errors = []
        
        for i, (eq_freq, eq_val) in enumerate(zip(eq_freqs, eq_values)):
            if eq_freq < target_low_freq and eq_val != -12.0:
                errors.append(f"{eq_freq:.1f}Hz应为-12dB，实际{eq_val:.1f}dB")
            elif eq_freq > target_high_freq and eq_val != -12.0:
                errors.append(f"{eq_freq:.1f}Hz应为-12dB，实际{eq_val:.1f}dB")
        
        if errors:
            print(f"  {eq_type}频率范围错误: {', '.join(errors)}")
            # 强制修正
            for i, eq_freq in enumerate(eq_freqs):
                if eq_freq < target_low_freq or eq_freq > target_high_freq:
                    eq_values[i] = -12.0
        else:
            out_of_range_count = sum(1 for freq in eq_freqs if freq < target_low_freq or freq > target_high_freq)
            print(f"  {eq_type}频率范围控制正确: {out_of_range_count}个频段设为-12dB")

    
    def calculate_eq_with_frequency_limits(self, freqs, measured_spectrum, target_spectrum, 
                                        target_low_freq, target_high_freq):
        """
        根据目标频率范围计算EQ设置
        """
        try:
            print(f"开始频率范围EQ计算: {target_low_freq}Hz - {target_high_freq}Hz")
            
            # 使用激进算法
            eq_15_final, eq_31_final, normalized_measured, _, _ = self.calculate_smart_eq_suggestions(
                freqs, measured_spectrum, target_spectrum, target_low_freq, target_high_freq
            )
            
            # 后验证：确保范围外频率为-12dB
            for i, eq_freq in enumerate(self.eq_15_freqs):
                if eq_freq < target_low_freq or eq_freq > target_high_freq:
                    if eq_15_final[i] != -12.0:
                        print(f"强制修正15段{eq_freq:.1f}Hz: {eq_15_final[i]:.1f}dB -> -12.0dB")
                        eq_15_final[i] = -12.0
                        
            for i, eq_freq in enumerate(self.eq_31_freqs):
                if eq_freq < target_low_freq or eq_freq > target_high_freq:
                    if eq_31_final[i] != -12.0:
                        print(f"强制修正31段{eq_freq:.1f}Hz: {eq_31_final[i]:.1f}dB -> -12.0dB")
                        eq_31_final[i] = -12.0
            
            # 统计有效调节
            eq_15_active = sum(1 for i, freq in enumerate(self.eq_15_freqs) 
                            if target_low_freq <= freq <= target_high_freq and abs(eq_15_final[i]) > 0.5)
            eq_31_active = sum(1 for i, freq in enumerate(self.eq_31_freqs) 
                            if target_low_freq <= freq <= target_high_freq and abs(eq_31_final[i]) > 0.5)
            
            print(f"校正完成: 15段={eq_15_active}个有效调节, 31段={eq_31_active}个有效调节")
            
            return eq_15_final, eq_31_final
            
        except Exception as e:
            print(f"频率范围EQ计算失败: {str(e)}")
            raise
    
    # ========== 主处理函数 ==========
    
    def process_audio_files(self, file_paths):
        """处理音频文件并返回完整分析结果 - 使用新算法"""
        try:
            results = {
                'eq_15_suggestions': None,
                'eq_31_suggestions': None,
                'eq_15_clipped': None,
                'eq_31_clipped': None,
                'eq_15_freqs': self.eq_15_freqs,
                'eq_31_freqs': self.eq_31_freqs,
                'eq_15_q_factor': self.eq_15_q_factor,
                'eq_31_q_factor': self.eq_31_q_factor,
                'frequency_data': None,
                'measured_spectrum_diff': None,
                'eq_15_applied_diff': None,
                'eq_31_applied_diff': None,
                'frequency_limits': None,
                'speaker_issues': None,
                'original_measured_spectrum': None,
                'files_processed': 0
            }
            
            measured_spectra = []
            
            print(f"开始处理 {len(file_paths)} 个音频文件")
            
            # 处理每个音频文件
            for file_path in file_paths:
                try:
                    print(f"正在处理文件: {file_path}")
                    
                    if not os.path.exists(file_path):
                        print(f"文件不存在: {file_path}")
                        continue
                    
                    audio_data, sr = librosa.load(file_path, sr=None, mono=True)
                    
                    if sr != self.sample_rate:
                        print(f"重采样从 {sr}Hz 到 {self.sample_rate}Hz")
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
                    
                    if len(audio_data) == 0:
                        print(f"文件为空: {file_path}")
                        continue
                    
                    # 音频预处理
                    if np.max(np.abs(audio_data)) > 0:
                        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.5
                    
                    freqs, spectrum = self.compute_spectrum(audio_data)
                    smoothed_spectrum = self.adaptive_smooth_spectrum(freqs, spectrum)
                    measured_spectra.append(smoothed_spectrum)
                    
                    print(f"成功处理文件: {file_path}")
                        
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")
                    continue
            
            if not measured_spectra:
                raise ValueError("没有成功处理任何音频文件")
            
            results['files_processed'] = len(measured_spectra)
            print(f"成功处理了 {len(measured_spectra)} 个音频文件，开始计算平均频谱")
            
            # 计算加权平均测量频谱
            average_measured_spectrum = np.mean(measured_spectra, axis=0)
            
            # 生成参考粉红噪声频谱
            reference_spectrum = self.get_theoretical_pink_noise_spectrum(freqs)
            
            # 使用新的智能EQ算法计算建议
            eq_15, eq_31, normalized_measured, eq_15_clipped, eq_31_clipped = self.calculate_smart_eq_suggestions(
                freqs, 
                average_measured_spectrum, 
                reference_spectrum
            )
            
            # 应用精确EQ调节
            eq_15_applied, _ = self.apply_eq_precise(freqs, normalized_measured, self.eq_15_freqs, eq_15, is_31_band=False)
            eq_31_applied, _ = self.apply_eq_precise(freqs, normalized_measured, self.eq_31_freqs, eq_31, is_31_band=True)
            
            # 准备可视化数据
            viz_freqs = np.logspace(np.log10(20), np.log10(20000), 200)
            
            # 插值到可视化频率点
            viz_measured = np.interp(viz_freqs, freqs, normalized_measured)
            viz_reference = np.interp(viz_freqs, freqs, reference_spectrum)
            viz_eq_15_applied = np.interp(viz_freqs, freqs, eq_15_applied)
            viz_eq_31_applied = np.interp(viz_freqs, freqs, eq_31_applied)
            
            # 计算与参考的差值
            viz_measured_diff = viz_measured - viz_reference
            viz_eq_15_applied_diff = viz_eq_15_applied - viz_reference
            viz_eq_31_applied_diff = viz_eq_31_applied - viz_reference
            
            # 使用5频段分析诊断音箱问题
            speaker_issues = self.analyze_speaker_issues_5_bands(viz_freqs, viz_measured_diff)
            
            # 计算频率极限
            eq_15_low_limit, eq_15_high_limit = self.calculate_frequency_limits(viz_freqs, viz_eq_15_applied_diff, -6)
            eq_31_low_limit, eq_31_high_limit = self.calculate_frequency_limits(viz_freqs, viz_eq_31_applied_diff, -6)
            
            # 组装结果
            results['eq_15_suggestions'] = eq_15
            results['eq_31_suggestions'] = eq_31
            results['eq_15_clipped'] = eq_15_clipped
            results['eq_31_clipped'] = eq_31_clipped
            results['frequency_data'] = viz_freqs.tolist()
            results['measured_spectrum_diff'] = viz_measured_diff.tolist()
            results['eq_15_applied_diff'] = viz_eq_15_applied_diff.tolist()
            results['eq_31_applied_diff'] = viz_eq_31_applied_diff.tolist()
            results['frequency_limits'] = {
                'eq_15': {'low': eq_15_low_limit, 'high': eq_15_high_limit},
                'eq_31': {'low': eq_31_low_limit, 'high': eq_31_high_limit}
            }
            results['speaker_issues'] = speaker_issues
            results['original_measured_spectrum'] = viz_measured.tolist()
            
            print(f"音频处理完成，生成了{len(speaker_issues)}个诊断结果")
            
            return results
            
        except Exception as e:
            print(f"处理音频文件时出错: {str(e)}")
            raise
        
    def apply_neighbor_relative_limit(self, eq_freqs, eq_gains, base_threshold_freq=400):
        """
        对中高频区域应用相对于邻居的偏差限制
        防止单个频段相对于周围环境过于突出
        """
        limited_gains = eq_gains.copy()
        
        print(f"\n=== 邻居相对偏差限制（≥{base_threshold_freq}Hz） ===")
        
        for i, freq in enumerate(eq_freqs):
            if freq >= base_threshold_freq:
                current_gain = eq_gains[i]
                
                # 收集邻居频段的增益值
                neighbors = []
                neighbor_freqs = []
                
                # 向左查找邻居（最多2个）
                for j in range(max(0, i-2), i):
                    neighbors.append(eq_gains[j])
                    neighbor_freqs.append(eq_freqs[j])
                
                # 向右查找邻居（最多2个）
                for j in range(i+1, min(len(eq_gains), i+3)):
                    neighbors.append(eq_gains[j])
                    neighbor_freqs.append(eq_freqs[j])
                
                if len(neighbors) >= 1:
                    # 计算邻居的加权平均（距离越近权重越高）
                    weights = []
                    for nf in neighbor_freqs:
                        distance_ratio = abs(freq - nf) / freq
                        weight = 1.0 / (1.0 + distance_ratio * 2)
                        weights.append(weight)
                    
                    neighbor_avg = np.average(neighbors, weights=weights)
                    
                    # 根据频率确定相对于邻居的最大允许偏差
                    if 400 <= freq <= 1000:
                        max_deviation = 8.0  # 中频相对邻居最大±8dB
                    elif 1000 <= freq <= 2000:
                        max_deviation = 6.0  # 中高频1相对邻居最大±6dB
                    elif 2000 <= freq <= 4000:
                        max_deviation = 4.0  # 中高频2相对邻居最大±4dB
                    elif 4000 <= freq <= 8000:
                        max_deviation = 3.0  # 高频相对邻居最大±3dB
                    else:
                        max_deviation = 2.0  # 超高频相对邻居最大±2dB
                    
                    # 计算当前增益相对于邻居的偏差
                    current_deviation = current_gain - neighbor_avg
                    
                    # 如果偏差超过限制，进行调整
                    if abs(current_deviation) > max_deviation:
                        # 限制偏差到最大允许值
                        limited_deviation = np.sign(current_deviation) * max_deviation
                        new_gain = neighbor_avg + limited_deviation
                        limited_gains[i] = new_gain
                        
                        print(f"  限制 {freq}Hz: {current_gain:.1f}dB -> {new_gain:.1f}dB "
                            f"(邻居均值={neighbor_avg:.1f}dB, 偏差限制±{max_deviation:.1f}dB)")
        
        return limited_gains

    def apply_midrange_smoothing_advanced(self, eq_freqs, eq_gains, smooth_threshold_freq=400):
        """
        对400Hz以上的中高频区域进行高级EQ平滑处理
        """
        smoothed_gains = eq_gains.copy()
        
        # 找到需要平滑处理的频段索引
        smooth_indices = [i for i, freq in enumerate(eq_freqs) if freq >= smooth_threshold_freq]
        
        if len(smooth_indices) < 3:
            return smoothed_gains
        
        print(f"\n=== 中高频高级平滑处理（≥{smooth_threshold_freq}Hz） ===")
        
        # 多次平滑迭代
        for iteration in range(2):
            temp_gains = smoothed_gains.copy()
            
            for i in smooth_indices:
                freq = eq_freqs[i]
                current_gain = smoothed_gains[i]
                
                # 收集相邻频段信息
                neighbors = []
                neighbor_freqs = []
                
                # 向左查找邻居（最多2个）
                left_count = 0
                for j in range(i-1, -1, -1):
                    if eq_freqs[j] >= smooth_threshold_freq and left_count < 2:
                        neighbors.append(smoothed_gains[j])
                        neighbor_freqs.append(eq_freqs[j])
                        left_count += 1
                    elif eq_freqs[j] < smooth_threshold_freq:
                        break
                
                # 向右查找邻居（最多2个）
                right_count = 0
                for j in range(i+1, len(eq_freqs)):
                    if eq_freqs[j] >= smooth_threshold_freq and right_count < 2:
                        neighbors.append(smoothed_gains[j])
                        neighbor_freqs.append(eq_freqs[j])
                        right_count += 1
                    else:
                        break
                
                if len(neighbors) >= 2:
                    # 计算邻居的加权平均
                    weights = []
                    for nf in neighbor_freqs:
                        distance = abs(freq - nf) / freq
                        weight = 1.0 / (1.0 + distance * 2)
                        weights.append(weight)
                    
                    weighted_neighbor_avg = np.average(neighbors, weights=weights)
                    
                    # 根据频段特性确定平滑强度
                    if 400 <= freq <= 1000:
                        smooth_factor = 0.4
                    elif 1000 <= freq <= 4000:
                        smooth_factor = 0.6
                    elif 4000 <= freq <= 8000:
                        smooth_factor = 0.7
                    else:
                        smooth_factor = 0.8
                    
                    # 只对大幅偏离的点进行平滑
                    deviation = abs(current_gain - weighted_neighbor_avg)
                    if deviation > 2.0:
                        adaptive_smooth = min(smooth_factor * (deviation / 4.0), 0.8)
                        new_gain = current_gain * (1 - adaptive_smooth) + weighted_neighbor_avg * adaptive_smooth
                        temp_gains[i] = new_gain
                        
                        print(f"  平滑 {freq}Hz: {current_gain:.1f}dB -> {new_gain:.1f}dB "
                            f"(偏离={deviation:.1f}dB, 强度={adaptive_smooth:.1f})")
            
            smoothed_gains = temp_gains
        
        return smoothed_gains

    def detect_and_smooth_peaks(self, eq_freqs, eq_gains, peak_threshold=4.0, min_freq=400):
        """
        检测并平滑中高频的尖峰调整
        """
        smoothed_gains = eq_gains.copy()
        
        print(f"\n=== 尖峰检测与平滑（≥{min_freq}Hz） ===")
        
        for i, freq in enumerate(eq_freqs):
            if freq < min_freq:
                continue
                
            current_gain = eq_gains[i]
            
            # 收集左右邻居
            left_gains = []
            right_gains = []
            
            # 左邻居（最多2个）
            for j in range(max(0, i-2), i):
                if eq_freqs[j] >= min_freq:
                    left_gains.append(eq_gains[j])
            
            # 右邻居（最多2个）
            for j in range(i+1, min(len(eq_gains), i+3)):
                if eq_freqs[j] >= min_freq:
                    right_gains.append(eq_gains[j])
            
            # 判断是否为峰值
            if len(left_gains) >= 1 and len(right_gains) >= 1:
                left_avg = np.mean(left_gains)
                right_avg = np.mean(right_gains)
                neighbor_avg = (left_avg + right_avg) / 2
                
                is_peak = False
                peak_type = ""
                
                # 检测正向峰值（突起）
                if (current_gain > neighbor_avg + peak_threshold and 
                    current_gain > max(left_gains + right_gains)):
                    is_peak = True
                    peak_type = "正峰"
                
                # 检测负向峰值（凹陷）
                elif (current_gain < neighbor_avg - peak_threshold and 
                    current_gain < min(left_gains + right_gains)):
                    is_peak = True
                    peak_type = "负峰"
                
                # 对检测到的峰值进行平滑
                if is_peak:
                    peak_magnitude = abs(current_gain - neighbor_avg)
                    smooth_strength = min(peak_magnitude / 8.0, 0.7)
                    
                    smoothed_gain = current_gain * (1 - smooth_strength) + neighbor_avg * smooth_strength
                    smoothed_gains[i] = smoothed_gain
                    
                    print(f"  {peak_type}平滑 {freq}Hz: {current_gain:.1f}dB -> {smoothed_gain:.1f}dB "
                        f"(峰值={peak_magnitude:.1f}dB)")
        
        return smoothed_gains
    
    def calculate_eq_with_frequency_limits_fixed(self, freqs, measured_spectrum, target_spectrum, 
                                           target_low_freq, target_high_freq, enable_smoothing=True):
        """
        根据目标频率范围计算EQ设置 - 修复版，支持算法选择
        """
        try:
            print(f"开始频率范围EQ计算: {target_low_freq}Hz - {target_high_freq}Hz")
            print(f"算法模式: {'平滑优化' if enable_smoothing else '精确校正'}")
            
            # ✅ 关键修复：传递算法参数
            eq_15_final, eq_31_final, normalized_measured, _, _ = self.calculate_smart_eq_suggestions(
                freqs, 
                measured_spectrum, 
                target_spectrum, 
                target_low_freq, 
                target_high_freq,
                enable_midrange_smoothing=enable_smoothing  # 传递算法参数
            )
            
            # 后验证：确保范围外频率为-12dB
            for i, eq_freq in enumerate(self.eq_15_freqs):
                if eq_freq < target_low_freq or eq_freq > target_high_freq:
                    if eq_15_final[i] != -12.0:
                        print(f"强制修正15段{eq_freq:.1f}Hz: {eq_15_final[i]:.1f}dB -> -12.0dB")
                        eq_15_final[i] = -12.0
                        
            for i, eq_freq in enumerate(self.eq_31_freqs):
                if eq_freq < target_low_freq or eq_freq > target_high_freq:
                    if eq_31_final[i] != -12.0:
                        print(f"强制修正31段{eq_freq:.1f}Hz: {eq_31_final[i]:.1f}dB -> -12.0dB")
                        eq_31_final[i] = -12.0
            
            # 统计有效调节
            eq_15_active = sum(1 for i, freq in enumerate(self.eq_15_freqs) 
                            if target_low_freq <= freq <= target_high_freq and abs(eq_15_final[i]) > 0.5)
            eq_31_active = sum(1 for i, freq in enumerate(self.eq_31_freqs) 
                            if target_low_freq <= freq <= target_high_freq and abs(eq_31_final[i]) > 0.5)
            
            # 计算平滑处理的频段数（如果启用了平滑算法）
            if enable_smoothing:
                eq_15_smoothed = sum(1 for i, freq in enumerate(self.eq_15_freqs) 
                                if freq >= 400 and abs(eq_15_final[i]) > 0.5)
                eq_31_smoothed = sum(1 for i, freq in enumerate(self.eq_31_freqs) 
                                if freq >= 400 and abs(eq_31_final[i]) > 0.5)
                print(f"平滑算法处理: 15段中高频={eq_15_smoothed}个, 31段中高频={eq_31_smoothed}个")
            
            print(f"{'平滑优化' if enable_smoothing else '精确校正'}算法完成: 15段={eq_15_active}个有效调节, 31段={eq_31_active}个有效调节")
            
            return eq_15_final, eq_31_final
            
        except Exception as e:
            print(f"频率范围EQ计算失败: {str(e)}")
            raise




# Flask路由定义
def open_browser():
    """延迟打开浏览器"""
    time.sleep(1.5)
    try:
        webbrowser.open('http://127.0.0.1:8088')
    except Exception as e:
        print(f"无法自动打开浏览器: {str(e)}")

@app.route('/')
def serve_frontend():
    """提供前端页面"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """提供静态文件"""
    return send_from_directory('.', filename)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """音频分析API - 支持中高频平滑优化"""
    temp_dir = None
    try:
        if 'audio_files' not in request.files:
            return jsonify({'success': False, 'error': '没有上传文件'})
        
        files = request.files.getlist('audio_files')
        if not files or files[0].filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        # 获取平滑算法开关参数
        enable_smoothing = request.form.get('enable_smoothing', 'true').lower() == 'true'
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        file_paths = []
        
        # 保存上传的文件
        for i, file in enumerate(files[:10]):
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, f"audio_{i}_{filename}")
                file.save(file_path)
                file_paths.append(file_path)
                print(f"保存文件: {file_path}")
        
        if len(file_paths) == 0:
            raise ValueError("没有有效的音频文件")
        
        # 使用分析器（带平滑优化参数）
        analyzer = AdvancedPinkNoiseAnalyzer()
        
        # 处理音频文件
        measured_spectra = []
        for file_path in file_paths:
            try:
                audio_data, sr = librosa.load(file_path, sr=None, mono=True)
                if sr != analyzer.sample_rate:
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=analyzer.sample_rate)
                
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.5
                
                freqs, spectrum = analyzer.compute_spectrum(audio_data)
                smoothed_spectrum = analyzer.adaptive_smooth_spectrum(freqs, spectrum)
                measured_spectra.append(smoothed_spectrum)
                
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
                continue
        
        if not measured_spectra:
            raise ValueError("没有成功处理任何音频文件")
        
        # 计算平均频谱
        average_measured_spectrum = np.mean(measured_spectra, axis=0)
        reference_spectrum = analyzer.get_theoretical_pink_noise_spectrum(freqs)
        
        # 使用修改后的EQ算法（包含平滑优化选项）
        eq_15, eq_31, normalized_measured, eq_15_clipped, eq_31_clipped = analyzer.calculate_smart_eq_suggestions(
            freqs, 
            average_measured_spectrum, 
            reference_spectrum,
            enable_midrange_smoothing=enable_smoothing  # 传递平滑算法开关
        )
        
        # 其余处理逻辑保持不变...
        eq_15_applied, _ = analyzer.apply_eq_precise(freqs, normalized_measured, analyzer.eq_15_freqs, eq_15, is_31_band=False)
        eq_31_applied, _ = analyzer.apply_eq_precise(freqs, normalized_measured, analyzer.eq_31_freqs, eq_31, is_31_band=True)
        
        # 准备可视化数据
        viz_freqs = np.logspace(np.log10(20), np.log10(20000), 200)
        viz_measured = np.interp(viz_freqs, freqs, normalized_measured)
        viz_reference = np.interp(viz_freqs, freqs, reference_spectrum)
        viz_eq_15_applied = np.interp(viz_freqs, freqs, eq_15_applied)
        viz_eq_31_applied = np.interp(viz_freqs, freqs, eq_31_applied)
        
        viz_measured_diff = viz_measured - viz_reference
        viz_eq_15_applied_diff = viz_eq_15_applied - viz_reference
        viz_eq_31_applied_diff = viz_eq_31_applied - viz_reference
        
        speaker_issues = analyzer.analyze_speaker_issues_5_bands(viz_freqs, viz_measured_diff)
        
        eq_15_low_limit, eq_15_high_limit = analyzer.calculate_frequency_limits(viz_freqs, viz_eq_15_applied_diff, -6)
        eq_31_low_limit, eq_31_high_limit = analyzer.calculate_frequency_limits(viz_freqs, viz_eq_31_applied_diff, -6)
        
        # 生成EQ建议文本
        eq_suggestions = {
            'eq_15': [
                {'freq': f"{int(freq) if freq == int(freq) else freq}Hz", 'adjustment': f"{adj:+.1f}dB"}
                for freq, adj in zip(analyzer.eq_15_freqs, eq_15)
            ],
            'eq_31': [
                {'freq': f"{int(freq) if freq == int(freq) else freq}Hz", 'adjustment': f"{adj:+.1f}dB"}
                for freq, adj in zip(analyzer.eq_31_freqs, eq_31)
            ]
        }
        
        # 准备图表数据
        chart_data = {
            'frequencies': viz_freqs.tolist(),
            'measured_spectrum_diff': viz_measured_diff.tolist(),
            'eq_15_applied_diff': viz_eq_15_applied_diff.tolist(),
            'eq_31_applied_diff': viz_eq_31_applied_diff.tolist(),
            'original_measured_spectrum': viz_measured.tolist(),
            'reference_spectrum': viz_reference.tolist()
        }
        
        print(f"返回数据: 频率点{len(chart_data['frequencies'])}, 平滑算法={'启用' if enable_smoothing else '禁用'}")
        
        return jsonify({
            'success': True,
            'eq_suggestions': eq_suggestions,
            'eq_15_clipped': [bool(x) for x in eq_15_clipped],
            'eq_31_clipped': [bool(x) for x in eq_31_clipped],
            'eq_15_q_factor': analyzer.eq_15_q_factor,
            'eq_31_q_factor': analyzer.eq_31_q_factor,
            'files_processed': len(measured_spectra),
            'chart_data': chart_data,
            'frequency_limits': {
                'eq_15': {'low': eq_15_low_limit, 'high': eq_15_high_limit},
                'eq_31': {'low': eq_31_low_limit, 'high': eq_31_high_limit}
            },
            'speaker_issues': speaker_issues,
            'smoothing_enabled': enable_smoothing  # 返回平滑算法状态
        })
        
    except Exception as e:
        error_msg = f'分析失败: {str(e)}'
        print(f"分析出错: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_msg})
    finally:
        # 清理临时文件
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"清理临时目录: {temp_dir}")
            except Exception as e:
                print(f"清理临时目录失败: {str(e)}")

@app.route('/api/reoptimize', methods=['POST'])
def reoptimize_eq():
    """重新优化EQ设置API - 修复算法选择问题"""
    try:
        data = request.get_json()
        
        # 验证输入数据
        if not data or 'frequencies' not in data or 'measured_spectrum_diff' not in data:
            return jsonify({'success': False, 'error': '缺少必要的频响数据'})
        
        # 获取数据
        frequencies = np.array(data['frequencies'])
        measured_spectrum_diff = np.array(data['measured_spectrum_diff'])
        target_low_freq = data.get('target_low_freq', 20)
        target_high_freq = data.get('target_high_freq', 20000)
        
        # ✅ 关键修复：正确接收平滑算法参数
        enable_smoothing = data.get('enable_smoothing', True)
        
        print(f"重新优化EQ请求: 目标范围 {target_low_freq}Hz - {target_high_freq}Hz")
        print(f"平滑算法状态: {'启用' if enable_smoothing else '禁用'}")
        print(f"输入数据: {len(frequencies)}个频率点")
        
        # 验证数据长度
        if len(frequencies) != len(measured_spectrum_diff):
            return jsonify({'success': False, 'error': '频率数据和频谱数据长度不匹配'})
        
        # 创建优化分析器实例
        analyzer = AdvancedPinkNoiseAnalyzer()
        
        # 生成理论粉红噪声频谱作为目标
        reference_spectrum = analyzer.get_theoretical_pink_noise_spectrum(frequencies)
        
        # 重构测量频谱（从差值恢复到绝对值）
        reconstructed_measured_spectrum = reference_spectrum + measured_spectrum_diff
        
        # ✅ 关键修复：使用修改后的函数，传递平滑算法参数
        optimized_eq_15, optimized_eq_31 = analyzer.calculate_eq_with_frequency_limits_fixed(
            frequencies, 
            reconstructed_measured_spectrum, 
            reference_spectrum,
            target_low_freq, 
            target_high_freq,
            enable_smoothing  # 传递平滑算法参数
        )
        
        # 计算优化后的预测频响
        eq_15_applied, _ = analyzer.apply_eq_precise(
            frequencies, reconstructed_measured_spectrum, 
            analyzer.eq_15_freqs, optimized_eq_15, is_31_band=False
        )
        
        eq_31_applied, _ = analyzer.apply_eq_precise(
            frequencies, reconstructed_measured_spectrum, 
            analyzer.eq_31_freqs, optimized_eq_31, is_31_band=True
        )
        
        # 转换为相对于粉红噪声的差值
        eq_15_applied_diff = eq_15_applied - reference_spectrum
        eq_31_applied_diff = eq_31_applied - reference_spectrum
        
        # 计算新的频率极限
        eq_15_low_limit, eq_15_high_limit = analyzer.calculate_frequency_limits(frequencies, eq_15_applied_diff, -6)
        eq_31_low_limit, eq_31_high_limit = analyzer.calculate_frequency_limits(frequencies, eq_31_applied_diff, -6)
        
        # 分析优化效果
        eq_15_active_bands = np.sum(np.abs(optimized_eq_15) > 0.1)
        eq_31_active_bands = np.sum(np.abs(optimized_eq_31) > 0.1)
        eq_15_max_correction = np.max(np.abs(optimized_eq_15))
        eq_31_max_correction = np.max(np.abs(optimized_eq_31))
        
        print(f"{'平滑' if enable_smoothing else '精确'}算法重新优化完成:")
        print(f"  15段EQ: {eq_15_active_bands}个有效频段, 最大校正{eq_15_max_correction:.1f}dB")
        print(f"  31段EQ: {eq_31_active_bands}个有效频段, 最大校正{eq_31_max_correction:.1f}dB")
        print(f"  15段下潜: {eq_15_low_limit:.0f}Hz, 31段下潜: {eq_31_low_limit:.0f}Hz")
        
        return jsonify({
            'success': True,
            'eq_15_settings': optimized_eq_15.tolist(),
            'eq_31_settings': optimized_eq_31.tolist(),
            'eq_15_applied_diff': eq_15_applied_diff.tolist(),
            'eq_31_applied_diff': eq_31_applied_diff.tolist(),
            'frequency_limits': {
                'eq_15': {'low': eq_15_low_limit, 'high': eq_15_high_limit},
                'eq_31': {'low': eq_31_low_limit, 'high': eq_31_high_limit}
            },
            'target_frequency_range': {
                'low': target_low_freq,
                'high': target_high_freq
            },
            'optimization_stats': {
                'eq_15_active_bands': int(eq_15_active_bands),
                'eq_31_active_bands': int(eq_31_active_bands),
                'eq_15_max_correction': float(eq_15_max_correction),
                'eq_31_max_correction': float(eq_31_max_correction)
            },
            'algorithm_used': 'smoothed' if enable_smoothing else 'precise'  # 返回使用的算法
        })
        
    except Exception as e:
        error_msg = f'重新优化失败: {str(e)}'
        print(f"重新优化出错: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_msg})


if __name__ == '__main__':
    print("=== 音箱EQ校准工具 - 完全修复版 ===")
    print(f"工作目录: {BASE_DIR}")
    print(f"上传目录: {app.config['UPLOAD_FOLDER']}")
    print("主要功能:")
    print("  • 全新智能EQ算法 - 正确的频率延伸优化")
    print("  • 修复平滑度计算 - 解决异常高数值问题")
    print("  • 优化参数传递 - 消除未定义变量错误")
    print("  • 智能频段识别 - 精确定位目标频率处理")
    print("  • 5频段问题诊断 - 超低频/低频/中频/高频/超高频")
    print("  • 延伸性能评估 - 量化频率下潜效果")
    print("\nAPI接口:")
    print("  POST /api/analyze - 音频智能分析接口")
    print("  POST /api/reoptimize - EQ智能重新优化接口")
    print(f"\n访问地址: http://127.0.0.1:8088")
    print("启动成功后将自动打开浏览器...")
    
    # 启动浏览器
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        app.run(debug=False, host='0.0.0.0', port=8088)
    except Exception as e:
        print(f"启动失败: {str(e)}")
        input("按任意键退出...")