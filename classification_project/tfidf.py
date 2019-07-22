import nltk
import numpy
import math
import csv
import string
import time
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem import WordNetLemmatizer


#from nltk import word_tokenize,sent_tokenize

class Document():
    def __init__(self):
        self.i = 0
        self.user = ''
        self.text = ''
        self.hashtags = ''
        self.classification = -1
        self.assignment = -1
        self.tokens = []
        self.tf = {}
        self.idf ={}
        self.tfidf = {}
        self.words = {}
        self.mean_fake = 0
        self.mean_real = 0

class Tup():
    def __init__(self):
        self.doc = 0
        self.score = 0

closed_class_stop_words = ["a","the","an","and","or","but","about","above","after","along","amid","among",\
                           "as","at","by","for","from","in","into","like","minus","near","of","off","on",\
                           "onto","out","over","past","per","plus","since","till","to","under","until","up",\
                           "via","vs","with","that","can","cannot","could","may","might","must",\
                           "need","ought","shall","should","will","would","have","had","has","having","be",\
                           "is","am","are","was","were","being","been","get","gets","got","gotten",\
                           "getting","seem","seeming","seems","seemed",\
                           "enough", "both", "all", "your" "those", "this", "these", \
                           "their", "the", "that", "some", "our", "no", "neither", "my",\
                           "its", "his" "her", "every", "either", "each", "any", "another",\
                           "an", "a", "just", "mere", "such", "merely" "right", "no", "not",\
                           "only", "sheer", "even", "especially", "namely", "as", "more",\
                           "most", "less" "least", "so", "enough", "too", "pretty", "quite",\
                           "rather", "somewhat", "sufficiently" "same", "different", "such",\
                           "when", "why", "where", "how", "what", "who", "whom", "which",\
                           "whether", "why", "whose", "if", "anybody", "anyone", "anyplace", \
                           "anything", "anytime" "anywhere", "everybody", "everyday",\
                           "everyone", "everyplace", "everything" "everywhere", "whatever",\
                           "whenever", "whereever", "whichever", "whoever", "whomever" "he",\
                           "him", "his", "her", "she", "it", "they", "them", "its", "their","theirs",\
                           "you","your","yours","me","my","mine","I","we","us","much","and/or",\
                           "http","i","rt"
                           ]

punctuation_arr = [']','[','_','$','#','@','\n','(', ')', '.', ',' , '!', '?', ':', ';','/','"','\\','-','´','“'];
#emojis get messed up from removing punctuation
def cosine_sim(dic1,dic2):
    num = 0
    dena = 0
    for key1,val1 in dic1.items():
        num += val1*dic2.get(key1,0.0)
        dena += val1*val1
    denb = 0
    for val2 in dic2.values():
        denb += val2*val2
    if (dena == 0 or denb == 0):
        return 0
    else:
        return num/math.sqrt(dena*denb)

def cosine_similarity(v1,v2):
    xx,yy,xy = 0,0,0
    i = 0
    for word in v1:
        x = v1[word]
        if word in v2:
            y = v2[word]
        else:
            y = 0
        xx += x*x
        yy += y*y
        xy += x*y
        i += 1
    if xx == 0 or yy == 0:
        return 0
    else:
        return xy/math.sqrt(xx*yy)

def tf_idf(num,halftest):
    start = time.time()

    fake_with_word = {}
    real_with_word = {}
    fake_tweets = []
    real_tweets = []
    unidentified_with_word = {}
    unidentified_tweets = []

    lemmatizer = WordNetLemmatizer()


    file = open("fake_tweets.csv", "r")
    file2 = open("real_tweets.csv", "r")
    file3 = open("trump_hillary.csv", "r")
    lines = csv.reader(file,delimiter=',', quotechar='"') #lines is a list
    lines2 = csv.reader(file2,delimiter=';', quotechar='"')
    lines3 = csv.reader(file3, delimiter=',', quotechar='"')
    #strip punctuation especially , and .
    x = 0
    for line in lines:
        if x == num:
            break
        if x != 0:
            doc = Document()
            doc.i = x
            doc.text = line[7]
            doc.hashtags = line[10]
            doc.user = line[1]
            fake_tweets.append(doc)
        x += 1

    x = 0
    for line in lines2:
        if x == num:
            break
        if x != 0:
            doc = Document()
            doc.i = x
            doc.text = line[3]
            doc.user = line[1]
            real_tweets.append(doc)
        x += 1

    x = 0

    for line in lines:
        if x == num+halftest:
            break
        if x >= num:
            doc = Document()
            doc.i = x
            doc.text = line[7]
            doc.user = line[1]
            doc.classification = 0
            unidentified_tweets.append(doc)
        x += 1

    x = 0
    for line in lines2:
        if x == num+halftest:
            break
        if x >= num:
            doc = Document()
            doc.i = x
            doc.text = line[3]
            doc.user = line[1]
            doc.classification = 1
            unidentified_tweets.append(doc)
        x += 1

    file.close()
    file2.close()

    for x in range(0,len(fake_tweets)):
        fake_tweets[x].classification = 0
        #fake_tweets[x].tokens = nltk.word_tokenize(fake_tweets[x].hashtags)
        #print(fake_tweets[x].tokens)
        fake_tweets[x].tokens += TweetTokenizer().tokenize(fake_tweets[x].text)
        for y in range(0, len(fake_tweets[x].tokens)):
            fake_tweets[x].tokens[y] = fake_tweets[x].tokens[y].lower()
            fake_tweets[x].tokens[y] = fake_tweets[x].tokens[y].translate(str.maketrans('','', string.punctuation))
            fake_tweets[x].tokens[y] = fake_tweets[x].tokens[y].translate(str.maketrans('','',  string.digits))
            fake_tweets[x].tokens[y] = lemmatizer.lemmatize(fake_tweets[x].tokens[y])
            if fake_tweets[x].tokens[y] in closed_class_stop_words or fake_tweets[x].tokens[y] in punctuation_arr or fake_tweets[x].tokens[y][0:4] == 'http':
                fake_tweets[x].tokens[y] = ''
        fake_tweets[x].tokens = [i for i in fake_tweets[x].tokens if i != '']
        #print(fake_tweets[x].tokens)

    for x in range(0,len(fake_tweets)):
        for y in range(0, len(fake_tweets[x].tokens)):
            if fake_tweets[x].tokens[y] in fake_tweets[x].words:
                fake_tweets[x].words[fake_tweets[x].tokens[y]]  += 1
            else:
                fake_tweets[x].words[fake_tweets[x].tokens[y]] = 1

    for x in range(0,len(fake_tweets)):
        for y in range(0, len(fake_tweets[x].tokens)):
            if fake_tweets[x].tokens[y] not in fake_tweets[x].tf:
                fake_tweets[x].tf[fake_tweets[x].tokens[y]] = fake_tweets[x].words[fake_tweets[x].tokens[y]] / len(fake_tweets[x].tokens)

    for x in range(0,len(fake_tweets)):
        for word in fake_tweets[x].words:
            if word in fake_with_word:
                fake_with_word[word] += 1
            else:
                fake_with_word[word] = 1

    for x in range(0,len(fake_tweets)):
        for word in fake_tweets[x].words:
            if word not in fake_tweets[x].idf:
                fake_tweets[x].idf[word] = math.log(len(fake_tweets)/fake_with_word[word])
                fake_tweets[x].tfidf[word] = fake_tweets[x].tf[word] * fake_tweets[x].idf[word]


    #print(fake_tweets[0].tfidf)


    for x in range(0,len(real_tweets)):
        real_tweets[x].tokens = TweetTokenizer().tokenize(real_tweets[x].text)
        real_tweets[x].classification = 1
        for y in range(0, len(real_tweets[x].tokens)):
            real_tweets[x].tokens[y] = real_tweets[x].tokens[y].lower()
            real_tweets[x].tokens[y] = real_tweets[x].tokens[y].translate(str.maketrans('','', string.punctuation))
            real_tweets[x].tokens[y] = real_tweets[x].tokens[y].translate(str.maketrans('','',  string.digits))
            real_tweets[x].tokens[y] = lemmatizer.lemmatize(real_tweets[x].tokens[y])
            if real_tweets[x].tokens[y] in closed_class_stop_words or real_tweets[x].tokens[y] in punctuation_arr or real_tweets[x].tokens[y][0:4] == 'http':
                real_tweets[x].tokens[y] = ''
        real_tweets[x].tokens = [i for i in real_tweets[x].tokens if i != '']
        #print(real_tweets[x].tokens)

    for x in range(0,len(real_tweets)):
        for y in range(0, len(real_tweets[x].tokens)):
            if real_tweets[x].tokens[y] in real_tweets[x].words:
                real_tweets[x].words[real_tweets[x].tokens[y]]  += 1
            else:
                real_tweets[x].words[real_tweets[x].tokens[y]] = 1

    for x in range(0,len(real_tweets)):
        for y in range(0, len(real_tweets[x].tokens)):
            if real_tweets[x].tokens[y] not in real_tweets[x].tf:
                real_tweets[x].tf[real_tweets[x].tokens[y]] = real_tweets[x].words[real_tweets[x].tokens[y]] / len(real_tweets[x].tokens)

    for x in range(0,len(real_tweets)):
        for word in real_tweets[x].words:
            if word in real_with_word:
                real_with_word[word] += 1
            else:
                real_with_word[word] = 1

    for x in range(0,len(real_tweets)):
        for word in real_tweets[x].words:
            if word not in real_tweets[x].idf:
                real_tweets[x].idf[word] = math.log(len(real_tweets)/real_with_word[word])
                real_tweets[x].tfidf[word] = real_tweets[x].tf[word] * real_tweets[x].idf[word]


    for x in range(0,len(unidentified_tweets)):
        unidentified_tweets[x].tokens = TweetTokenizer().tokenize(unidentified_tweets[x].text)
        for y in range(0, len(unidentified_tweets[x].tokens)):
            unidentified_tweets[x].tokens[y] = unidentified_tweets[x].tokens[y].lower()
            unidentified_tweets[x].tokens[y] = unidentified_tweets[x].tokens[y].translate(str.maketrans('','', string.punctuation))
            unidentified_tweets[x].tokens[y] = unidentified_tweets[x].tokens[y].translate(str.maketrans('','',  string.digits))
            unidentified_tweets[x].tokens[y] = lemmatizer.lemmatize(unidentified_tweets[x].tokens[y])
            if unidentified_tweets[x].tokens[y] in closed_class_stop_words or unidentified_tweets[x].tokens[y] in punctuation_arr or unidentified_tweets[x].tokens[y][0:4] == 'http':
                unidentified_tweets[x].tokens[y] = ''
        unidentified_tweets[x].tokens = [i for i in unidentified_tweets[x].tokens if i != '']
        #print(unidentified_tweets[x].tokens)

    for x in range(0,len(unidentified_tweets)):
        for y in range(0, len(unidentified_tweets[x].tokens)):
            if unidentified_tweets[x].tokens[y] in unidentified_tweets[x].words:
                unidentified_tweets[x].words[unidentified_tweets[x].tokens[y]]  += 1
            else:
                unidentified_tweets[x].words[unidentified_tweets[x].tokens[y]] = 1

    for x in range(0,len(unidentified_tweets)):
        for y in range(0, len(unidentified_tweets[x].tokens)):
            if unidentified_tweets[x].tokens[y] not in unidentified_tweets[x].tf:
                unidentified_tweets[x].tf[unidentified_tweets[x].tokens[y]] = unidentified_tweets[x].words[unidentified_tweets[x].tokens[y]] / len(unidentified_tweets[x].tokens)

    for x in range(0,len(unidentified_tweets)):
        for word in unidentified_tweets[x].words:
            if word in unidentified_with_word:
                unidentified_with_word[word] += 1
            else:
                unidentified_with_word[word] = 1

    for x in range(0,len(unidentified_tweets)):
        for word in unidentified_tweets[x].words:
            if word not in unidentified_tweets[x].idf:
                unidentified_tweets[x].idf[word] = math.log(len(unidentified_tweets)/unidentified_with_word[word])
                unidentified_tweets[x].tfidf[word] = unidentified_tweets[x].tf[word] * unidentified_tweets[x].idf[word]



    centroid = {}
    centroiddiv = {}
    centroid2 = {}
    centroid2div = {}

    for x in range(0, len(fake_tweets)):
        for word in fake_tweets[x].tfidf:
            if word in centroid:
                centroid[word] += fake_tweets[x].tfidf[word]
                #centroiddiv[word] += 1
            else:
                centroid[word] = fake_tweets[x].tfidf[word]
                #centroiddiv[word] = 1

        for word in real_tweets[x].tfidf:
            if word in centroid2:
                centroid2[word] += real_tweets[x].tfidf[word]
                #centroid2div[word] += 1
            else:
                centroid2[word] = real_tweets[x].tfidf[word]
                #centroid2div[word] = 1

    #logreg = LogisticRegression()
    #dv = DictVectorizer(sparse=True)
    #fake = dv.fit_transform(centroid)
    #real = dv.fit_transform(centroid2)
    #print(fake)
    #print('success')
    #print(fake.getnnz())
    #c1 = [[]]
    #c2 = [[]]
    #c1key = [[]]#[[0 for y in range(2)]for x in range(len(centroid))]
    #c1val = [[]]
    #c2key = [[]]
    #c2val = [[]]
    #l1key = []
    #l2key = []
    #c1[i][0]
    #c1[i][1]
    #for word in centroid:
        #pair = []
        #pair.append(word)
        #pair.append(centroid[word])
        #c1.append(pair)
        #c1key.append(word)
        #c1val.append(centroid[word])
        #l1key.append(0)

    #print(c1)
    #for word in centroid2:
        #pair = []
        #pair.append(word)
        #pair.append(centroid2[word])
        #c2.append(pair)
        #c2key.append(word)
        #c2val.append(centroid2[word])
        #l2key.append(1)

    #logreg.fit(c1key,)
    #logreg.fit(c2key,c2m)

    #vectorize each word in fake tweets then

        #for word in centroid:
            #centroid[word] /= centroiddiv[word]
        #for word in centroid2:
            #centroid2[word] /= centroid2div[word]


    true_positive = 0 #fakefake predicted its fake and it is fake 11
    true_negative = 0 #realreal predicted its real and it is real
    false_positive = 0 #fakereal predicted its fake but its real
    false_negative = 0 #realfake predicted its real but its fake
    incorrect = 0
    correct = 0
    for x in range(0,len(unidentified_tweets)):
        average_fake_score = cosine_sim(unidentified_tweets[x].tfidf,centroid)
        average_real_score = cosine_sim(unidentified_tweets[x].tfidf,centroid2)
        if average_fake_score > average_real_score:
            unidentified_tweets[x].assignment = 0
        else:
            unidentified_tweets[x].assignment = 1


        if unidentified_tweets[x].assignment != unidentified_tweets[x].classification:
            incorrect += 1
        else:
            correct += 1
        if unidentified_tweets[x].classification == 0: #is fake
            if unidentified_tweets[x].assignment == 0:
                true_positive += 1
            else:
                #print(average_fake_score)
                false_negative += 1 #biggest problem
        else:
            if unidentified_tweets[x].assignment == 0:
                #print(average_real_score)
                false_positive += 1
            else:
                true_negative += 1
    print(incorrect/(incorrect+correct),true_positive, false_positive,false_negative,true_negative)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    print(num,halftest,incorrect/(incorrect+correct),true_positive,false_negative,false_positive,true_negative)
    print('F1 =', 2*(precision*recall)/(precision+recall))

    """
            real    fake
    real realreal  realfake

    fake fakereal  fakefake

    centroid_fake = Document()
    centroid_real = Document()
    for x in range(0,len(fake_tweets)):
        for word in fake_tweets[x].tfidf:
            if word in centroid_fake.tfidf:
                centroid_fake.tfidf[word] += fake_tweets[x].tfidf[word]
            else:
                centroid_fake.tfidf[word] = fake_tweets[x].tfidf[word]
    for x in range(0,len(real_tweets)):
        for word in real_tweets[x].tfidf:
            if word in centroid_real.tfidf:
                centroid_real.tfidf[word] += real_tweets[x].tfidf[word]
            else:
                centroid_real.tfidf[word] = real_tweets[x].tfidf[word]
    #for word in centroid_fake.tfidf:
        #centroid_fake.tfidf[word] /= len(fake_tweets)
    #for word in centroid_real.tfidf:
        #centroid_real.tfidf[word] /= len(real_tweets)
    #print(centroid_fake.tfidf)
    #print(centroid_real.tfidf)
    true_positive = 0 #fakefake predicted its fake and it is fake 11
    true_negative = 0 #realreal predicted its real and it is real
    false_positive = 0 #fakereal predicted its fake but its real
    false_negative = 0 #realfake predicted its real but its fake
    incorrect = 0
    correct = 0
    for x in range(0,len(unidentified_tweets)):
        average_fake_score = cosine_similarity(unidentified_tweets[x].tfidf,centroid_fake.tfidf)
        average_real_score = cosine_similarity(unidentified_tweets[x].tfidf,centroid_real.tfidf)
        unidentified_tweets[x].mean_fake = average_fake_score
        unidentified_tweets[x].mean_real = average_real_score

        if unidentified_tweets[x].mean_fake > unidentified_tweets[x].mean_real:
            unidentified_tweets[x].assignment = 0
            print(0)
        else:
            unidentified_tweets[x].assignment = 1
            print(1)
        if unidentified_tweets[x].assignment != unidentified_tweets[x].classification:
            incorrect += 1
        else:
            correct += 1
        if unidentified_tweets[x].classification == 0: #is fake
            if unidentified_tweets[x].assignment == 0:
                true_positive += 1
            else:
                false_negative += 1 #biggest problem
        else:
            if unidentified_tweets[x].assignment == 0:
                false_positive += 1
            else:
                true_negative += 1
"""
    """
    true_positive = 0 #fakefake predicted its fake and it is fake 11
    true_negative = 0 #realreal predicted its real and it is real
    false_positive = 0 #fakereal predicted its fake but its real
    false_negative = 0 #realfake predicted its real but its fake
    incorrect = 0
    correct = 0
    for x in range(0,len(unidentified_tweets)):
        average_fake_score = 0
        average_real_score = 0
        for y in range(0, len(fake_tweets)):
            average_fake_score += cosine_sim(unidentified_tweets[x].tfidf,fake_tweets[y].tfidf)
            average_real_score += cosine_sim(unidentified_tweets[x].tfidf,real_tweets[y].tfidf)
        unidentified_tweets[x].mean_fake = average_fake_score / len(fake_tweets)
        unidentified_tweets[x].mean_real = average_real_score / len(real_tweets)
        if unidentified_tweets[x].mean_fake > unidentified_tweets[x].mean_real:
            unidentified_tweets[x].assignment = 0
        else:
            unidentified_tweets[x].assignment = 1
        if unidentified_tweets[x].assignment != unidentified_tweets[x].classification:
            incorrect += 1
        else:
            correct += 1
        if unidentified_tweets[x].classification == 0: #is fake
            if unidentified_tweets[x].assignment == 0:
                true_positive += 1
            else:
                false_negative += 1 #biggest problem
        else:
            if unidentified_tweets[x].assignment == 0:
                false_positive += 1
            else:
                true_negative += 1
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    print(num,halftest,incorrect/(incorrect+correct),true_positive,false_negative,false_positive,true_negative)
    print('F1 =', 2*(precision*recall)/(precision+recall))

    #output = open("output5000.csv", "w")

    #for x in range(0,len(real_tweets)):
        #if real_tweets[x].mean_fake > real_tweets[x].mean_real:
            #incorrect += 1
            #output.write(str(x) + ',' + str(real_tweets[x].mean_fake) + ',' + str(real_tweets[x].mean_real) + ',0,' + str(real_tweets[x].classification) + '\n')
        #else:
            #correct += 1
            #output.write(str(x) + ',' + str(real_tweets[x].mean_fake) + ',' + str(real_tweets[x].mean_real) + ',1,' + str(real_tweets[x].classification) + '\n')
        #print(x,real_tweets[x].mean_fake,real_tweets[x].mean_real)

    #output.close()
"""
    end = time.time()
    print('time:',end-start)

#tf_idf(501,50)
#tf_idf(1001,50)
#tf_idf(2001,50)
tf_idf(8001,1000) #good
#tf_idf(128001,50)

#tf_idf(8001,100)
#tf_idf(16001,50)
