import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

titanic_dataset = sns.load_dataset("titanic")
print(titanic_dataset.head())