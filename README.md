# Improve Your Mind Like an Athlete

# Description
A journaling application using machine learning to guide your self improvement.

# Problem to Solve
Mental Health is a rising issue, even before COVID.

In the United States between 2017-2018, 19% of adults experienced a mental illness, an increase of 1.5 million people over last year’s dataset.

There's been an increase of self-improvement books, methods, writers, apps, etc. all availabe on the internet.

Yet when it comes to someone's own mental improvement, how does one actually know they are improving besides just "feeling"?

If an athelete is trying to improve their game, do they just "feel" like they are getting better?

Whether you're at work, or in the gym, there's some baseline starting point to improve on.

Measured as a KPI, max weight, repititions, distance, or if you can fit into your university clothes, we all need some measurement.

But when it comes to mental health, where's the starting point?

# Solution
Introducing Project Friday (insert real app name). The Journal designed around the user's mental health.

This Journal analyzes the User's audio entries to classify sentiment (currently just for polarity) based on the major Keys of your day. Whether it's negative or positive, the Journal will show you how each Key impacts your overall day. From there, the User can start to become self aware on how to brigthen their tomorrow, whether by removing negative Keys or by doing positive Keys.

The Journal can also give slight "nudges" to help the User focus on positive Keys to improve their overall day.

# Challenges
There are a few challenges I'm running across.

1. When applying sentiment, most datasets and papers I find are for either text analysis or audio analysis. I'd like the Journal to be able to analyze on both.

2. Unclear how to build the model to apply both text and audio sentiment analysis.

3. Finding a labeled dataset that has both text and audio sentiment analysis.

# Datasets

## Keyword Extraction Dataset
Need to find

## Audio Sentiment Analysis Datasets

**Four major datasets**
- Surrey Audio-Visual Expressed Emotion
- Ryerson Audio-Visual Database of Emotional Speech and Song
- Toronto emotional speech set
- Crowd Sourced Emotional Multimodal Actors Dataset

### Dataset Details

#### 1. SAVEE dataset
  - Surrey Audio-Visual Expressed Emotion
    - 480 files - 107 MB
    - Male only
    - [Kaggle Link](https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee)
  - The SAVEE database was recorded from four native English male speakers (identified as DC, JE, JK, KL), postgraduate students and researchers at the University of Surrey aged from 27 to 31 years.
  - Emotion has been described psychologically in discrete categories: anger, disgust, fear, happiness, sadness and surprise. A neutral category is also added to provide recordings of 7 emotion categories.
  - The text material consisted of 15 TIMIT sentences per emotion: 3 common, 2 emotion-specific and 10 generic sentences that were different for each emotion and phonetically-balanced.
  - The 3 common and 2 × 6 = 12 emotion-specific sentences were recorded as neutral to give 30 neutral sentences. This resulted in a total of 120 utterances per speaker


#### 2. RAVDESS dataset
  - Ryerson Audio-Visual Database of Emotional Speech and Song
    - 1440 files - 429 MB
    - Male & Female
    - [Kaggle Link](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)
  - The dataset was created by 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent.
  - Speech emotions includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.
  - Each actor has 60 recordings.


#### 3. TESS dataset
  - Toronto emotional speech set
    - 2800 files - 428 MB
    - Female only
    - [Kaggle Link](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)
  - There are a set of 200 target words were spoken in the carrier phrase "Say the word _' by two actresses (aged 26 and 64 years).
  - Recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral).


#### 4. CREMA-D dataset
  - Crowd Sourced Emotional Multimodal Actors Dataset
    - 7442 files - 451 MB
    - Male & Female
    - [Kaggle Link](https://www.kaggle.com/ejlok1/cremad)
  - These clips were from 48 male and 43 female actors (91 total) between the ages of 20 and 74 coming from a variety of races and ethnicities (African America, Asian, Caucasian, Hispanic, and Unspecified).
  - Actors spoke from a selection of 12 sentences.
  - The sentences were presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad) and four different emotion levels (Low, Medium, High, and Unspecified).

# Modeling

- Input: Journal Entry as audio file
- Output1: Polarity Sentiment for the Entry
- Output2: Polarity Sentiment impact by Keys 

## Model 1

**Model 1.A - Audio Sentiment Analysis Model**
- input audio
- generate MFCC
- use CNN2D

**Model 1.B - Text Sentiment Analysis Model**
- input audio
- extract text
- LSTM with attention (attention to pull Keys)

**Combine Model 1.A & 1.B**
- combine model feature vectors with Dense Layer
- Softmax 3

**Potential Issues**
Have no idea if this will work. Overall response from Teachers is not great.
Need to create this quickly and train on a small sample set to see results.

## Model 2
Similar to Model 1, but instead of combining with Dense Layer, use something called **ensemble learning**.
What is this? Good question.

Ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone

Potential Issues: No idea what that is. I just heard of this on Wednesday the 25th.

## Model 3
Same as Model 1, except keep output of Model 1.A and 1.B separate.

Show predictions separately, or could take some sort of average. However, leaning away from averaging.

## Model 4
Transformer Model??? Similar issues with Model 2, as in, not really sure how to implement

# Phases
My approach for the next 3 weeks.

## Phase 1
- Complete Model 1.A (Audio)
    - Model 1.A is currently 75% complete
- Train on Model 1.A
- Find dataset for text sentiment analysis
- Complete Model 1.B (Text)
- research Ensemble Learning & Transformer Model

## Phase 2
- add recording capability
- build Model 2 - ensemble learning, if research is promising
    - if not promisinig, skip to Model 4 Transformer
- start working on Flask/Bootstrap
- What does Model 3, keeping predictions seperate, look like?
- start building Model 4

## Phase 3
- finalize which Model
- Train, train, train, train
- continue Flask/Bootstrap
- start presentation
- prep Demo
- clean up README

## Phase 4 (last days before I.DD)
- practice Demo
- build Gifs for contingency plan if Demo fails
- contingency plan if model fails
- finish presentation

