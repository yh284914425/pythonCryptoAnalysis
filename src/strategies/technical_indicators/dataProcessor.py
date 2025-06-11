import pandas as pd
import numpy as np

from typing import Dict, List, Any, Optional

# 数据预处理工具
class DataProcessor:
    """数据预处理工具 - 统一处理数据类型和格式"""
    
    @staticmethod
    def ensure_numeric(df: pd.DataFrame, required_columns: List[str] = None) -> pd.DataFrame:
        """确保数值列的类型正确，处理边界情况，避免TA-Lib输入错误"""
        if required_columns is None:
            required_columns = ['开盘价', '最高价', '最低价', '收盘价', '成交量']
        
        df_processed = df.copy()
        
        for col in required_columns:
            if col in df_processed.columns:
                # 统一转换为数值类型，错误值转为NaN
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                
                # 检查是否整列都是NaN
                if df_processed[col].isna().all():
                    # 使用合理的默认值
                    if col == '成交量':
                        df_processed[col] = 0.0
                    else:
                        # 价格列使用其他有效列的值作为参考
                        reference_value = None
                        for ref_col in required_columns:
                            if ref_col != col and ref_col in df_processed.columns:
                                valid_values = df_processed[ref_col].dropna()
                                if len(valid_values) > 0:
                                    reference_value = valid_values.iloc[-1]
                                    break
                        
                        # 如果找到参考值，使用；否则使用默认值
                        df_processed[col] = reference_value if reference_value is not None else 100.0
                else:
                    # 正常的填充逻辑
                    df_processed[col] = df_processed[col].ffill()
                    
                    # 如果还有NaN（开头），用后向填充
                    df_processed[col] = df_processed[col].bfill()
                    
                    # 最后的保护：如果仍有NaN，用中位数填充
                    if df_processed[col].isna().any():
                        median_val = df_processed[col].median()
                        if pd.notna(median_val):
                            df_processed[col] = df_processed[col].fillna(median_val)
                        else:
                            # 极端情况，使用默认值
                            default_val = 0.0 if col == '成交量' else 100.0
                            df_processed[col] = df_processed[col].fillna(default_val)
                
                # 确保最终类型为float64
                df_processed[col] = df_processed[col].astype(np.float64)
        
        return df_processed
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """验证数据质量"""
        quality_report = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        required_columns = ['开盘价', '最高价', '最低价', '收盘价', '成交量']
        
        for col in required_columns:
            if col not in df.columns:
                quality_report['valid'] = False
                quality_report['issues'].append(f"缺少列: {col}")
                continue
                
            # 检查数据类型
            if not pd.api.types.is_numeric_dtype(df[col]):
                quality_report['issues'].append(f"{col} 不是数值类型")
            
            # 检查负值
            if (df[col] < 0).any():
                quality_report['issues'].append(f"{col} 包含负值")
            
            # 检查NaN比例
            nan_ratio = df[col].isnull().sum() / len(df)
            if nan_ratio > 0.1:
                quality_report['issues'].append(f"{col} NaN比例过高: {nan_ratio:.1%}")
            
            quality_report['stats'][col] = {
                'nan_count': df[col].isnull().sum(),
                'nan_ratio': nan_ratio,
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        return quality_report
