import pandas as pd

df = pd.DataFrame({'close': [10, 12, 11, 13, 12, 14, 13]})

df['local_high'] = (df['close'] > df['close'].shift(1)) & (df['close'] > df['close'].shift(-1))
df['local_low'] = (df['close'] < df['close'].shift(1)) & (df['close'] < df['close'].shift(-1))

print(df)