import pandas as pd
from sklearn.model_selection import train_test_split

input_file = "/slurm/home/yrd/chen3lab/zhangyu/ondemand/Project/Food/Acp_predictor_project/data/ACP240.csv"
df = pd.read_csv(input_file)


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)



train_file = "/slurm/home/yrd/chen3lab/zhangyu/ondemand/Project/Food/New_ACP/data/ACP240/train.csv"
test_file = "/slurm/home/yrd/chen3lab/zhangyu/ondemand/Project/Food/New_ACP/data/ACP240/test.csv"
train_df.to_csv(train_file, index=False)  # index=False

test_df.to_csv(test_file, index=False)

print(f"Finish ! {len(train_df)}{train_file}{len(test_df)}{test_file}")