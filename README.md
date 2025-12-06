# Drug-Food-Interactions
Machine Learning Project with the purpose of identifying conflicts between drug and food ingredients using both text and images.

We've started working on the dataset we generated to do EDA and feature transformation.
- The column distributions were visualized -> all of them were right skewed.
- We checked the drug count, we have 20 of each (evenly distributed, which is logical because it is AI generated).
## Feature Transformation
- We fixed the right skewing by performing **`log-1p`** on the data, fixing the distributions, except Tyramine which remained heavily right skewed (normal thing).
- We label encoded the drug column.
- We visualized the correlation matrix with a heatmap and feature engineered multiple new columns.
## Preliminary Results
- We made a quick model to test how we did, and it got an accuracy rate of 0.91.

(TBA)
