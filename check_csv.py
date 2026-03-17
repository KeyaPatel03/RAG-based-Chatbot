import pandas as pd

df = pd.read_csv('rag_evaluation_results.csv')
print('Columns:', df.columns.tolist())
print('\nNumber of rows:', len(df))
print('\nFirst question:')
print(df['user_input'].iloc[0][:300] if len(df) > 0 else 'Empty')
print('\nMetric columns with NaN:')
for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
    if col in df.columns:
        print(f'{col}: {df[col].isna().sum()} NaN values out of {len(df)}')
        print(f'  Data type: {df[col].dtype}')
        # Check if any values are strings
        non_null = df[col].dropna()
        if len(non_null) > 0:
            print(f'  Sample values: {non_null.head(3).tolist()}')
