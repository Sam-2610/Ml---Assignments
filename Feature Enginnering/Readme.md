# Python Pandas Practice

Three focused practice modules covering the core pandas workflow — transform, merge, and reshape — as used in data analytics and machine learning pipelines.

---

## Files Practiced

### 1. `Transforming Data.py`
Feature engineering, conditional column creation, sorting, ranking, and aggregation with `groupby`.

| Method | Purpose |
|--------|---------|
| `apply()` | Create a `tax` column based on `income` values |
| `map()` | Standardize `gender` values |
| `assign()` | Add a calculated `new_income` column |
| `replace()` | Update country labels |
| `rename()` | Rename columns |
| `sort_values()` / `sort_index()` | Order data and reset index |
| `rank()` | Calculate rankings with different methods |
| `groupby()` + `agg()` | Group and aggregate data |

---

### 2. `Merging and Joining.py`
Combining DataFrames using all four join types on a shared key column.

| Join Type | Description |
|-----------|-------------|
| Inner join | Rows with matching keys in both tables |
| Left join | All rows from left, matched rows from right |
| Right join | All rows from right, matched rows from left |
| Outer join | All rows from both tables |

> Practiced joining customer and order data using `pd.merge()` with a common key.

---

### 3. `Reshaping Methods.py`
Converting between wide and long table formats for analysis and reporting.

- Used `melt()` to convert a wide table to long format
- Used `pivot()` to convert long format back to wide

---

## Skills Practiced

- DataFrame creation
- Data transformation and feature engineering
- Mapping and replacing values
- Sorting and ranking
- Grouping and aggregation
- Merging and joining datasets
- Reshaping data with `melt()` and `pivot()`

---

## Notes

These files cover the most common data preparation steps used before modelling — cleaning, structuring, and enriching raw data into analysis-ready form. The operations practiced here are foundational to both analytics and machine learning workflows.

---

## Author

**Satyam Sagar**  
satyamsagar@gmail.com

