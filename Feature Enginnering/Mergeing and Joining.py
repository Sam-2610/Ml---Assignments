# Merging and Joining

import pandas as pd

df_customers = pd.DataFrame({
    "customer_id" : [1,2,3,4],
    "name" : ["Adam","Bob","Charlie","Dave"]
})

df_orders = pd.DataFrame({
    "order_id" : [101,102,103,104],
    "customer_id" : [2,1,4,5],
    "amount" : [250,120,300,180]
})

pd.merge(df_customers,df_orders,on="customer_id") # Inner Join
pd.merge(df_customers,df_orders,on="customer_id",how="left") # Left Join
pd.merge(df_customers,df_orders,on="customer_id",how="right")# Right Join
pd.merge(df_customers,df_orders,on="customer_id",how="outer")# Outer Join

