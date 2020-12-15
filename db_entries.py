'''
Code for Entries Database
Created on 20201215 0745
Initial code from Kermadec Lab 2.5a
'''

import sqlite3

# connect to database
# conn = sqlite3.connect('journal.db')
# # conn = sqlite3.connect('journal.db')

# # create a cursor
# cur = conn.cursor()

# example output
# {'text': "this is a clean version I really hope it's not index out of range of light is it it doesn't ", 'text_confidence': 0.76, 'text_wordcount': 20, 'keywords': 'clean version', 'text_polarity': 'negative', 'text_polarity_prob': {'score': -0.580131, 'label': 'negative'}, 'text_emotion': 'fear', 'text_emotion_prob': {'sadness': 0.09857, 'joy': 0.101424, 'fear': 0.167227, 'disgust': 0.04349, 'anger': 0.015309}, 'audio_polarity': 'neutral', 'audio_polarity_prob': {'negative': 0.03578012436628342, 'neutral': 0.9490557312965393, 'positive': 0.015164160169661045}}

# Create table categories in the database using a function
def create_journal_entries_table(cur):
    query = """
        CREATE TABLE IF NOT EXISTS journal_entries (
            je_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date integer,
            text_content text,
            text_confidence integer,
            text_wordcount integer,
            keywords text,
            text_polarity text,
            text_polarity_prob text,
            text_emotion text,
            text_emotion_prob text,
            audio_polarity text,
            audio_polarity_prob text,
            audio_emotion text,
            audio_emotion_prob text,
            entry_filepath text,
            create_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    try:
        cur.execute(query)
        # conn.commit()
    except Exception as err:
        print('ERROR BY CREATE TABLE', err)

# creating class to add things to a database
class Journal_Entry:
    def __init__(self, results_dict, cur, je_id=None, create_at=None): # date, text_content, text_confidence, text_wordcount, keywords, text_polarity, text_polarity_prob, text_emotion, text_emotion_prob, audio_polarity, audio_polarity_prob, audio_emotion, audio_emotion_prob, entry_filepath, create_at, 
        self.je_id = je_id # these are the same categories as in SQL database made above
        self.date = results_dict['date']
        self.text_content = results_dict['text_content']
        self.text_confidence = results_dict['text_confidence']
        self.text_wordcount = results_dict['text_wordcount']
        self.keywords = results_dict['keywords']
        self.text_polarity = results_dict['text_polarity']
        self.text_polarity_prob = str(results_dict['text_polarity_prob'])
        self.text_emotion = results_dict['text_emotion']
        self.text_emotion_prob = str(results_dict['text_emotion_prob'])
        self.audio_polarity = results_dict['audio_polarity']
        self.audio_polarity_prob = str(results_dict['audio_polarity_prob'])
        self.audio_emotion = results_dict['audio_emotion']
        self.audio_emotion_prob = str(results_dict['audio_emotion_prob'])
        self.entry_filepath = results_dict['entry_filepath']
        self.create_at = create_at
        self.cur = cur

    def __repr__(self):
        return f"JE_ID: {self.je_id}, Date: {self.date}, text_content: {self.text_content}, text_confidence: {self.text_confidence}, text_wordcount: {self.text_wordcount}, keywords: {self.keywords}, text_polarity: {self.text_polarity}, text_polarity_prob: {self.text_polarity_prob}, text_emotion: {self.text_emotion}, text_emotion_prob: {self.text_emotion_prob}, audio_polarity: {self.audio_polarity}, audio_polarity_prob: {self.audio_polarity_prob}, audio_emotion: {self.audio_emotion}, audio_emotion_prob: {self.audio_emotion_prob}, entry_filepath: {self.entry_filepath}, create_at: {self.create_at}"

    def save_into_db(self): # saving itself into a table. same as INSERT ROW OF DATA section above
        column_list = ['date', 
                'text_content', 
                'text_confidence', 
                'text_wordcount', 
                'keywords', 
                'text_polarity', 
                'text_polarity_prob', 
                'text_emotion',
                'text_emotion_prob', 
                'audio_polarity', 
                'audio_polarity_prob',
                'audio_emotion',
                'audio_emotion_prob',
                'entry_filepath',]
        
        query = f"""
            INSERT INTO journal_entries ({', '.join(column_list)})
            VALUES ({', '.join(['?' for _ in range(len(column_list))])});
        """
        val = (self.date, self.text_content, self.text_confidence, self.text_wordcount, self.keywords, self.text_polarity, self.text_polarity_prob, self.text_emotion, self.text_emotion_prob, self.audio_polarity, self.audio_polarity_prob, self.audio_emotion, self.audio_emotion_prob, self.entry_filepath)
        print(query)
        # conn = sqlite3.connect('journal.db')

        # create a cursor
        # cur = self.conn.cursor()

        try:
            self.cur.execute(query, val)
            self.je_id = self.cur.lastrowid
            # conn.commit()
        except Exception as err:
            print('ERROR BY INSERT:', err)

# entry = Journal_Entry(results_dict)
# entry.save_into_db()