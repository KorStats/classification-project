from nltk import tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.probability import FreqDist
import xlrd
import pymysql

host = "localhost"
db_id = "root"
db_pw = "12345"
db_name = "Sentiment"

db =  pymysql.connect(host, db_id, db_pw, db_name, use_unicode=True, charset="utf8")
cursor = db.cursor()

# vader_lexicon.txt파일에서 Subject와 Object에 해당되는 단어들에 대해 분류
# C:\Users\Won\Anaconda3\Lib\site-packages\vaderSentiment
n_instances = 5000
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]

# 80번까지의 단어 모음은 학습용, 81~100번까지의 단어 모음은 테스트용
# 학습 데이터셋은 5000개 까지 지정되어 있음
# C:\Users\Won\AppData\Roaming\nltk_data\corpora\subjectivity의 plot.tok.gt9.5000(Object 학습용), quote.tok.gt9.5000(Subject 학습용)
# 출처 : http://www.cs.cornell.edu/people/pabo/movie-review-data
# 학습 데이터셋의 개수가 증가할 수록 정확도 및 기타 지수들이 증가하는 추세임을 확인

train_subj_docs = subj_docs[:4001]
test_subj_docs = subj_docs[4001:5001]
train_obj_docs = obj_docs[:4001]
test_obj_docs = obj_docs[4001:5001]

training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs

# 부정과 구두점 사이의 범위에 나타나는 단어에 '_NEG' 접미사 추가
# 예) 'never', 'becomes_NEG', 'the_NEG', 'clever_NEG', 'crime_NEG', 'comedy_NEG', 'it_NEG', 'thinks_NEG', 'it_NEG', 'is_NEG', '.'
# 위의 문장에서 'never'와 '.' 사이에 _NEG가 추가됨
# 아마도.. 문장의 전체적인 흐름이 부정인데 긍정의 의미를 가진 단어로 인해 잘못 평가되는 오류를 막기 위한 것으로 추측됨(ex. 위의 문장에서 comedy, clever 등..)
sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])

# 아래의 경로에 부여된 점수에 의해 감성점수가 계산됨
# C:\Users\Won\AppData\Roaming\nltk_data\sentiment/vader_lexicon.zip
# lexicon_file="sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt"

unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

# 감정 분석을 위한 특성 추출
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)

cursor.execute("DELETE FROM sent140")
db.commit()

# 테스트 데이터셋에 대해 감정 평가 점수 매기기
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
     print('{0}: {1}'.format(key, value))

# 엑셀파일에서 트윗 읽어오기
workbook = xlrd.open_workbook('sentiment1402.xlsx')
worksheet = workbook.sheet_by_index(0)
tweets = []
date = []
user = []

for i in range(worksheet.nrows):
    date.append(worksheet.cell_value(i,0))
    user.append(worksheet.cell_value(i,1))
    tweets.append(worksheet.cell_value(i,2))
    # tweets.append(worksheet.cell_value(i, 2))
sid = SentimentIntensityAnalyzer()

i = 0

for sentence in tweets:
    sentence = sentence.replace('"', ' ')
    print(sentence)
    ss = sid.polarity_scores(sentence)

    tokens = nltk.word_tokenize(sentence)
    print(tokens)
    tagged_tokens=nltk.pos_tag(tokens)
    print(tagged_tokens)

    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()
    cursor.execute("INSERT INTO sent140 (date, user, tweet, negative, neutral, positive, compound) VALUES (%s, %s, %s, %s, %s, %s, %s)",(date[i], user[i], sentence, ss['neg'], ss['neu'], ss['pos'], ss['compound']))
    # cursor.execute("INSERT INTO sent140 (tweet, negative, neutral, positive, compound) VALUES (%s, %s, %s, %s, %s)",(sentence, ss['neg'], ss['neu'], ss['pos'], ss['compound']))
    db.commit()
    i += 1
'''
------------- 참고자료 ----------------
http://www.nltk.org/howto/sentiment.html
http://blog.naver.com/PostView.nhn?blogId=ossiriand&logNo=220597867112&parentCategoryNo=&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView
https://pypi.python.org/pypi/vaderSentiment

======== 아래는 Compound 점수 매기는 방법! =======
출처 : http://stackoverflow.com/questions/40325980/how-is-the-vader-compound-polarity-score-calculated-in-python-nltk
compound = normalize(sum_s)

def normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score/math.sqrt((score*score) + alpha)
    return norm_score

'''