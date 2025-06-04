import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os

class TradingViewChart:
    """
    类似TradingView的专业K线图表类
    """
    
    def __init__(self, data_dir='crypto_data'):
        """
        初始化图表
        :param data_dir: 数据目录
        """
        self.data_dir = data_dir
        self.data = None
        self.fig = None
        self.symbol = 'BTCUSDT'
        self.current_interval = '1h'
        
        # 可用的时间周期
        self.intervals = {
            '1m': '1分钟', '3m': '3分钟', '5m': '5分钟', '15m': '15分钟', '30m': '30分钟',
            '1h': '1小时', '2h': '2小时', '4h': '4小时', '6h': '6小时', '8h': '8小时', '12h': '12小时',
            '1d': '1天', '3d': '3天', '1w': '1周', '1M': '1月'
        }
        
        # 技术指标配置
        self.indicators_config = {
            'ma_periods': [5, 10, 20, 60, 120, 200],
            'ma_colors': ['#FF6B35', '#2196F3', '#9C27B0', '#F44336', '#4CAF50', '#FF9800'],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_period': 20,
            'bollinger_std': 2
        }
        
    def load_data(self, symbol, interval):
        """
        加载指定交易对和时间周期的数据
        :param symbol: 交易对
        :param interval: 时间周期
        """
        self.symbol = symbol
        self.current_interval = interval
        
        filename = f"{symbol}_{interval}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                self.data = pd.read_csv(filepath, encoding='utf-8-sig')
                beijing_tz = pytz.timezone('Asia/Shanghai')
                self.data['开盘时间'] = pd.to_datetime(self.data['开盘时间'])
                
                # 如果时间是naive，添加北京时区
                if self.data['开盘时间'].dt.tz is None:
                    self.data['开盘时间'] = self.data['开盘时间'].dt.tz_localize(beijing_tz)
                
                # 确保数据类型正确
                for col in ['开盘价', '最高价', '最低价', '收盘价', '成交量']:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                
                print(f"✅ 加载数据: {symbol} {interval} - {len(self.data)} 条记录")
                return True
            except Exception as e:
                print(f"❌ 加载数据失败: {e}")
                return False
        else:
            print(f"❌ 数据文件不存在: {filepath}")
            return False
    
    def calculate_technical_indicators(self):
        """
        计算所有技术指标
        """
        if self.data is None or len(self.data) == 0:
            return
        
        # 移动平均线
        for period in self.indicators_config['ma_periods']:
            if len(self.data) >= period:
                self.data[f'MA{period}'] = self.data['收盘价'].rolling(window=period).mean()
        
        # 布林带
        period = self.indicators_config['bollinger_period']
        std_multiplier = self.indicators_config['bollinger_std']
        if len(self.data) >= period:
            self.data[f'MA{period}'] = self.data['收盘价'].rolling(window=period).mean()
            self.data[f'MA{period}_std'] = self.data['收盘价'].rolling(window=period).std()
            self.data['BOLL_UPPER'] = self.data[f'MA{period}'] + std_multiplier * self.data[f'MA{period}_std']
            self.data['BOLL_LOWER'] = self.data[f'MA{period}'] - std_multiplier * self.data[f'MA{period}_std']
        
        # RSI
        period = self.indicators_config['rsi_period']
        if len(self.data) >= period:
            delta = self.data['收盘价'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        fast = self.indicators_config['macd_fast']
        slow = self.indicators_config['macd_slow']
        signal = self.indicators_config['macd_signal']
        
        if len(self.data) >= slow:
            exp1 = self.data['收盘价'].ewm(span=fast).mean()
            exp2 = self.data['收盘价'].ewm(span=slow).mean()
            self.data['MACD'] = exp1 - exp2
            self.data['MACD_signal'] = self.data['MACD'].ewm(span=signal).mean()
            self.data['MACD_hist'] = self.data['MACD'] - self.data['MACD_signal']
        
        # 成交量移动平均
        self.data['Volume_MA'] = self.data['成交量'].rolling(window=20).mean()
    
    def create_professional_chart(self, show_volume=True, show_indicators=True, ma_lines=[5, 10, 20], height=800):
        """
        创建专业的K线图表
        :param show_volume: 显示成交量
        :param show_indicators: 显示技术指标
        :param ma_lines: 显示的移动平均线周期
        :param height: 图表高度
        """
        if self.data is None or len(self.data) == 0:
            print("❌ 没有数据可显示")
            return None
        
        # 计算技术指标
        if show_indicators:
            self.calculate_technical_indicators()
        
        # 创建子图布局
        if show_volume and show_indicators:
            subplot_titles = [
                f'{self.symbol} - {self.intervals.get(self.current_interval, self.current_interval)}',
                '成交量',
                'RSI(14)',
                'MACD(12,26,9)'
            ]
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.02,
                row_heights=[0.55, 0.15, 0.15, 0.15],
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )
        elif show_volume:
            subplot_titles = [
                f'{self.symbol} - {self.intervals.get(self.current_interval, self.current_interval)}',
                '成交量'
            ]
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.02,
                row_heights=[0.75, 0.25]
            )
        else:
            fig = make_subplots(rows=1, cols=1)
            fig.update_layout(title=f'{self.symbol} - {self.intervals.get(self.current_interval, self.current_interval)}')
        
        # K线图
        fig.add_trace(
            go.Candlestick(
                x=self.data['开盘时间'],
                open=self.data['开盘价'],
                high=self.data['最高价'],
                low=self.data['最低价'],
                close=self.data['收盘价'],
                name='价格',
                increasing_line_color='#26a69a',  # 上涨蜡烛颜色
                decreasing_line_color='#ef5350',  # 下跌蜡烛颜色
                increasing_fillcolor='#26a69a',
                decreasing_fillcolor='#ef5350',
                line=dict(width=1),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # 移动平均线
        if show_indicators:
            for i, period in enumerate(ma_lines):
                ma_col = f'MA{period}'
                if ma_col in self.data.columns and i < len(self.indicators_config['ma_colors']):
                    fig.add_trace(
                        go.Scatter(
                            x=self.data['开盘时间'],
                            y=self.data[ma_col],
                            mode='lines',
                            name=f'MA{period}',
                            line=dict(
                                color=self.indicators_config['ma_colors'][i],
                                width=1.5
                            ),
                            opacity=0.7,
                            hovertemplate=f'MA{period}: %{{y:.2f}}<extra></extra>'
                        ),
                        row=1, col=1
                    )
            
            # 布林带
            if 'BOLL_UPPER' in self.data.columns:
                # 上轨
                fig.add_trace(
                    go.Scatter(
                        x=self.data['开盘时间'],
                        y=self.data['BOLL_UPPER'],
                        mode='lines',
                        name='布林上轨',
                        line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dot'),
                        showlegend=False,
                        hovertemplate='布林上轨: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # 下轨
                fig.add_trace(
                    go.Scatter(
                        x=self.data['开盘时间'],
                        y=self.data['BOLL_LOWER'],
                        mode='lines',
                        name='布林下轨',
                        line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dot'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.05)',
                        showlegend=False,
                        hovertemplate='布林下轨: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # 成交量
        if show_volume:
            # 计算成交量颜色
            colors = []
            for i in range(len(self.data)):
                if i == 0:
                    colors.append('#26a69a')
                else:
                    if self.data.iloc[i]['收盘价'] >= self.data.iloc[i]['开盘价']:
                        colors.append('#26a69a')
                    else:
                        colors.append('#ef5350')
            
            fig.add_trace(
                go.Bar(
                    x=self.data['开盘时间'],
                    y=self.data['成交量'],
                    name='成交量',
                    marker_color=colors,
                    opacity=0.7,
                    hovertemplate='成交量: %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 成交量移动平均线
            if 'Volume_MA' in self.data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.data['开盘时间'],
                        y=self.data['Volume_MA'],
                        mode='lines',
                        name='成交量MA',
                        line=dict(color='orange', width=1),
                        opacity=0.8
                    ),
                    row=2, col=1
                )
        
        # RSI指标
        if show_indicators and 'RSI' in self.data.columns:
            row_idx = 3 if show_volume else 2
            fig.add_trace(
                go.Scatter(
                    x=self.data['开盘时间'],
                    y=self.data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='#FF6B35', width=2),
                    hovertemplate='RSI: %{y:.2f}<extra></extra>'
                ),
                row=row_idx, col=1
            )
            
            # RSI参考线
            fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1,
                         annotation_text="超买(70)", annotation_position="bottom right",
                         row=row_idx, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1,
                         annotation_text="超卖(30)", annotation_position="top right",
                         row=row_idx, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", line_width=1,
                         row=row_idx, col=1)
        
        # MACD指标
        if show_indicators and 'MACD' in self.data.columns:
            row_idx = 4 if show_volume else 3
            
            # MACD线
            fig.add_trace(
                go.Scatter(
                    x=self.data['开盘时间'],
                    y=self.data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='#2196F3', width=2),
                    hovertemplate='MACD: %{y:.4f}<extra></extra>'
                ),
                row=row_idx, col=1
            )
            
            # 信号线
            fig.add_trace(
                go.Scatter(
                    x=self.data['开盘时间'],
                    y=self.data['MACD_signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='#FF9800', width=2),
                    hovertemplate='Signal: %{y:.4f}<extra></extra>'
                ),
                row=row_idx, col=1
            )
            
            # MACD柱状图
            colors = ['#26a69a' if hist >= 0 else '#ef5350' for hist in self.data['MACD_hist']]
            fig.add_trace(
                go.Bar(
                    x=self.data['开盘时间'],
                    y=self.data['MACD_hist'],
                    name='MACD柱',
                    marker_color=colors,
                    opacity=0.6,
                    hovertemplate='MACD柱: %{y:.4f}<extra></extra>'
                ),
                row=row_idx, col=1
            )
        
        # 专业化布局设置
        fig.update_layout(
            height=height,
            showlegend=True,
            hovermode='x unified',
            template='plotly_dark',
            
            # 专业的TradingView风格
            font=dict(
                family="Roboto, sans-serif",
                size=12,
                color='#D1D5DB'
            ),
            
            plot_bgcolor='#131722',  # TradingView深色背景
            paper_bgcolor='#131722',
            
            # 图例设置
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=10)
            ),
            
            # 边距设置
            margin=dict(l=10, r=10, t=50, b=10),
            
            # 禁用range slider
            xaxis_rangeslider_visible=False,
            
            # 十字线
            dragmode='pan'
        )
        
        # X轴设置 - 类似TradingView的时间格式
        fig.update_xaxes(
            type='date',
            tickformat='%m/%d %H:%M',
            tickangle=0,
            gridcolor='rgba(128,128,128,0.2)',
            linecolor='rgba(128,128,128,0.3)',
            mirror=True,
            showspikes=True,
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1,
            spikecolor='rgba(128,128,128,0.5)'
        )
        
        # Y轴设置
        fig.update_yaxes(
            gridcolor='rgba(128,128,128,0.2)',
            linecolor='rgba(128,128,128,0.3)',
            mirror=True,
            showspikes=True,
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1,
            spikecolor='rgba(128,128,128,0.5)',
            side='right'  # 价格轴在右侧，像TradingView
        )
        
        # 设置主图表的Y轴标题
        fig.update_yaxes(title_text="价格 (USDT)", row=1, col=1)
        
        if show_volume:
            fig.update_yaxes(title_text="成交量", row=2, col=1)
        
        if show_indicators:
            if show_volume:
                fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
                fig.update_yaxes(title_text="MACD", row=4, col=1)
            else:
                fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
                fig.update_yaxes(title_text="MACD", row=3, col=1)
        
        # 增强的缩放和交互功能
        fig.update_layout(
            # 启用所有交互功能
            dragmode='pan',
            selectdirection='h',
            
            # 工具栏配置
            modebar=dict(
                bgcolor='rgba(0,0,0,0)',
                color='rgba(255,255,255,0.7)',
                activecolor='white'
            )
        )
        
        self.fig = fig
        return fig
    
    def show(self):
        """
        显示图表
        """
        if self.fig:
            self.fig.show(config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{self.symbol}_{self.current_interval}_chart',
                    'height': 800,
                    'width': 1400,
                    'scale': 2
                },
                'scrollZoom': True  # 启用鼠标滚轮缩放
            })
        else:
            print("❌ 请先创建图表")
    
    def save_html(self, filename=None):
        """
        保存为HTML文件
        """
        if not filename:
            filename = f"{self.symbol}_{self.current_interval}_professional_chart.html"
        
        if self.fig:
            self.fig.write_html(
                filename,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'scrollZoom': True,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'{self.symbol}_{self.current_interval}_chart',
                        'height': 800,
                        'width': 1400,
                        'scale': 2
                    }
                }
            )
            print(f"✅ 专业图表已保存为: {filename}")
        else:
            print("❌ 请先创建图表")
    
    def get_price_info(self):
        """
        获取当前价格信息
        """
        if self.data is None or len(self.data) == 0:
            return None
        
        latest = self.data.iloc[-1]
        previous = self.data.iloc[-2] if len(self.data) > 1 else latest
        
        change = latest['收盘价'] - previous['收盘价']
        change_percent = (change / previous['收盘价']) * 100
        
        return {
            '交易对': self.symbol,
            '时间周期': self.intervals.get(self.current_interval, self.current_interval),
            '最新时间': latest['开盘时间'],
            '当前价格': latest['收盘价'],
            '24h变化': change,
            '24h变化率': change_percent,
            '24h最高': self.data['最高价'].tail(24).max() if len(self.data) >= 24 else latest['最高价'],
            '24h最低': self.data['最低价'].tail(24).min() if len(self.data) >= 24 else latest['最低价'],
            '24h成交量': self.data['成交量'].tail(24).sum() if len(self.data) >= 24 else latest['成交量'],
            '数据条数': len(self.data)
        }
    
    def get_available_intervals(self):
        """
        获取可用的时间周期
        """
        available = []
        for interval in self.intervals.keys():
            filename = f"{self.symbol}_{interval}.csv"
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                available.append(interval)
        return available

# 保持向后兼容
CandlestickChart = TradingViewChart 