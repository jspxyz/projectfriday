import json
import pandas as pd
import sqlite3
import numpy as np

conn = sqlite3.connect('journal.db')
cur = conn.cursor()

# data_full = cur.execute('SELECT * FROM journal_entries;').fetchall()

# dates = cur.execute('SELECT date FROM journal_entries;').fetchall()

##### creating text_polarity dataset
# text_pol_query = pd.read_sql_query('''SELECT text_polarity_prob 
#                                         FROM journal_entries''', conn)

# # print(text_pol_query)
# text_pol_table = text_pol_query['text_polarity_prob'].map(eval)
# # print(text_pol_table)
# text_pol_forchart = text_pol_table.apply(pd.Series)

# text_pol_forchart['date'] = pd.read_sql_query('''SELECT date 
#                             FROM journal_entries''', conn)
# text_pol_forchart['date'] = pd.to_datetime(text_pol_forchart['date'], yearfirst=True, format='%Y%m%d')
# text_pol_forchart.set_index('date', inplace=True)

# print(text_pol_forchart)
##### end creating text_polarity dataset




##### start creating a date list
# date = pd.read_sql_query('''SELECT date 
#                             FROM journal_entries''', conn)

# date['date'] = pd.to_datetime(date['date'], yearfirst=True, format='%Y%m%d')
# date_list = date['date'].to_list()

# print(date_list)
##### end date list end

##### start text polarity score list start
text_pol_query = pd.read_sql_query('''SELECT text_polarity_prob 
                                        FROM journal_entries''', conn)

text_pol_table = text_pol_query['text_polarity_prob'].map(eval)
text_pol_forchart = text_pol_table.apply(pd.Series)
text_pol_forchart.columns = ['text_score', 'label']
text_pol_score = text_pol_forchart['text_score'].to_list()

print(text_pol_score)

#### end text polarity score list done



##### START audio polarity score list
audio_pol_query = pd.read_sql_query('''SELECT audio_polarity_prob 
                                        FROM journal_entries''', conn)
audio_pol = audio_pol_query['audio_polarity_prob'].map(eval)
audio_pol = audio_pol.apply(pd.Series)
print(audio_pol)

def get_pol_score(x):
    pol_index = np.argmax(x.values)
    if pol_index == 0:
        return x[pol_index] * -1
    elif pol_index == 1: 
        return 0
    else:
        return x[pol_index]

audio_pol['audio_score'] = audio_pol.apply(get_pol_score, axis=1)

audio_pol_score = audio_pol['audio_score'].to_list()

print(audio_pol_score)

# mood_data = pd.concat([text_pol_forchart, audio_pol], axis=1)
# print(mood_data)
# print(audio_pol_query['audio_polarity_prob'])

# print(audio_pol_query)

# print(audio_pol_query['audio_polarity_prob'])

# for i in audio_pol_query['audio_polarity_prob']:
#     dic = eval(i)
#     for k in audio_pol_query['audio_polarity']:
#         print(k)
#         print(dic[k])

#         print("-"*10)
#         print("this is i")
#         print(i)
#         print(type(i))
#         print("-"*10)
#         x = eval(i)
#         # x = '"{}"'.format(i)
#         print("this is x")
#         print(x)
#         print(type(x))



##### END audio polarity score list


# text_pol_forchart['date'] = pd.to_datetime(text_pol_forchart['date'], yearfirst=True, format='%Y%m%d')
# text_pol_forchart.set_index('date', inplace=True)