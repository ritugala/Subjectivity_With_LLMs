import pandas as pd

df1 = pd.read_csv("/home/ubuntu/Subjectivity_With_LLMs/predictions/mistral_llm_features_dev_v2.csv")
df2 = pd.read_csv("/home/ubuntu/Subjectivity_With_LLMs/predictions/mistral_llm_features_v2.csv")

#Explicitly checked that it works
data = pd.concat([df1, df2])
# create an array for this separated by commas
columns = ["Filler Words","Stuttering","Concise","Repetition","Rambling","External References","Complicated Terms","Sarcasm","Rhetorical Questions","Logical Coherence","Hedging Language","Assertive Language"]

# columns = ['Filler Words','Concise','Rambling','External References','Complicated Terms','Hedging Language','Assertive Language']

# Binarizing the columns: 1 for 'yes', 0 for 'no'
data.drop('Logical Contradictions', axis=1, inplace=True)
data.fillna('no', inplace=True)
for col in columns:
    data[col] = data[col].map({'yes': 1, 'no': 0})
breakpoint()
for fold in range(0,4):
    for split in ["train.tsv", "test.tsv"]:
        folder_name = f"/home/ubuntu/Subjectivity_With_LLMs/subjective_discourse/subjective_discourse/data/gold/CongressionalHearingFolds/fold{fold}/"
        data2 = pd.read_csv(folder_name+split, sep="\t")

        # columns_to_add = ['Filler Words','Concise','Rambling','External References','Complicated Terms','Hedging Language','Assertive Language']
        columns_to_add = ["Filler Words","Stuttering","Concise","Repetition","Rambling","External References","Complicated Terms","Sarcasm","Rhetorical Questions","Logical Coherence","Hedging Language","Assertive Language"]

        merged_df = data2.merge(data[['Index'] + columns_to_add], left_on='qa_index_digits', right_on='Index')

        merged_df.drop('Index', axis=1, inplace=True)
        new_split_name = "old_features_" + split 
        merged_df.to_csv(folder_name+new_split_name, sep="\t", index= False)

        print("Len: ", merged_df.shape)



