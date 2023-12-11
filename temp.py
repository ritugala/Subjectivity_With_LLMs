import pandas as pd
data = pd.read_csv("/home/ubuntu/Subjectivity_With_LLMs/predictions/Preprocessed_Features.csv")
data = data.iloc[:,1:]
# breakpoint()
# columns_to_check = data.columns[4:18]
# for col in columns_to_check:
#     data[col] = data[col].str.lower()
# data = data[data[columns_to_check].isin(['yes', 'no']).all(axis=1)]
# breakpoint()
columns = ['Filler Words','Concise','Rambling','External References','Complicated Terms','Hedging Language','Assertive Language']

# Binarizing the columns: 1 for 'yes', 0 for 'no'
for col in columns:
    data[col] = data[col].map({'yes': 1, 'no': 0})
    
data2 = pd.read_csv("/home/ubuntu/Subjectivity_With_LLMs/subjective_discourse/subjective_discourse/data/gold/CongressionalHearing/train.tsv", sep="\t")

columns_to_add = ['Filler Words','Concise','Rambling','External References','Complicated Terms','Hedging Language','Assertive Language']
breakpoint()
# Merge data2 with selected columns from data
merged_df = data2.merge(data[['Index'] + columns_to_add], left_on='qa_index_digits', right_on='Index')

# Drop the 'Index' column from merged_df as it's no longer needed
merged_df.drop('Index', axis=1, inplace=True)

merged_df.to_csv("/home/ubuntu/Subjectivity_With_LLMs/subjective_discourse/subjective_discourse/data/gold/CongressionalHearing/features_preprocessed.tsv", sep="\t", index= False)
