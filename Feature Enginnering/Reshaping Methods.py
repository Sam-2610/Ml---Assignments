# Reshaping Methods

import pandas as pd

df = pd.DataFrame({
    "Country" : ["USA","USA","India","India"],
    "Year" : [2020,2021,2020,2021],
    "Sales" : [100,120,90,110],
    "Profit" : [20,25,18,22]
})


# Melted(Wide -> Long)
melted = df.melt(
    id_vars=["Country","Year"],
    value_vars=["Sales","Profit"],
    var_name="metric",
    value_name="value"
)

print(melted)

#Pivot(Long -> Wide)
Original = melted.pivot(
    index=["Country","Year"],
    columns="metric",
    values="value"
)

print(Original)