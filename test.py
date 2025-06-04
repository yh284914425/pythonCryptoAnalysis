# import pandas as pd
 
# df = pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])

# print(df["a"])
# print(df[["a","b"]])


import pandas as pd

# 创建一个有name的Series
s = pd.Series([1, 2, 3, 4, 5], 
              index=['a', 'b', 'c', 'd', 'e'],
              name='my_series')

print(s)
print(f"Series的name: {s.name}")

print(s["my_series"])