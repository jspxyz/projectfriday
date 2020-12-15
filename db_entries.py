'''
Code for Entries Database
Created on 20201215 0745
Initial code from Kermadec Lab 2.5a
'''

import sqlite3

# connect to database
conn = sqlite3.connect('journal.db')
# conn = sqlite3.connect('journal.db')

# create a cursor
cur = conn.cursor()

# example output
# {'text': "this is a clean version I really hope it's not index out of range of light is it it doesn't ", 'text_confidence': 0.76, 'text_wordcount': 20, 'keywords': 'clean version', 'text_polarity': 'negative', 'text_polarity_prob': {'score': -0.580131, 'label': 'negative'}, 'text_emotion': 'fear', 'text_emotion_prob': {'sadness': 0.09857, 'joy': 0.101424, 'fear': 0.167227, 'disgust': 0.04349, 'anger': 0.015309}, 'audio_polarity': 'neutral', 'audio_polarity_prob': {'negative': 0.03578012436628342, 'neutral': 0.9490557312965393, 'positive': 0.015164160169661045}}

# Create table categories in the database using a function
def create_entries_table():
    query = """
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date integer
            text text
            text_confidence integer
            text_wordcount integer
            keywords text
            text_polarity text
            text_polarity_prob text
            text_emotion text
            text_emotion_prob text
            audio_polarity text
            audio_polarity_prob text
            audio_emotion text
            audio_emotion_prob text
            entry_filepath text
            create_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    try:
        cur.execute(query)
        conn.commit()
    except Exception as err:
        print('ERROR BY CREATE TABLE', err)
        
create_entries_table()

# creating class to add things to a database
class Category:
    def __init__(self, name, url, parent_id=None, cat_id=None): 
        self.cat_id = cat_id # these are the same categories as in SQL database made above
        self.name = name
        self.url = url
        self.parent_id = parent_id

    def __repr__(self):
        return f"ID: {self.cat_id}, Name: {self.name}, URL: {self.url}, Parent: {self.parent_id}"

    def save_into_db(self): # saving itself into a table. same as INSERT ROW OF DATA section above
        column_list = ['p_title', 
                'cat_id', 
                'seller_product_id', 
                'sku', 
                'price', 
                'p_product_id', 
                'brand', 
                'category',
                'p_url', 
                'img_url', 
                'p_original_price',
                'discount',
                'refund',
                'TIKI_now']
        
        query = """
            INSERT INTO categories (name, url, parent_id)
            VALUES (?, ?, ?);
        """
        val = (self.name, self.url, self.parent_id)
        try:
            cur.execute(query, val)
            self.cat_id = cur.lastrowid
            conn.commit()
        except Exception as err:
            print('ERROR BY INSERT:', err)

# query the database
conn.execute("SELECT * FROM entries")
# conn.fetchone()
# conn.fetchmany(3)
print(conn.fetchall())

# Datatypes:
# NULL
# INTEGER
# REAL
# TEXT
# BLOB

print('Command executed')
# commit our command
conn.commit()

# close our connection
conn. close()