import base64

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pickle
from datetime import datetime
import json
import streamlit as st
import pandas as pd
import os
import pyunpack
from outlook_msg import Message
from english_words import english_words_set
import nltk
from tqdm import tqdm

nltk.download('stopwords')

def get_params():
    st.sidebar.write(os.listdir())
    st.sidebar.header('Upload Your Data')
    file = st.sidebar.file_uploader(label='.zip containing folders as folder_name = labels')
    st.sidebar.header('Train Test Split')
    train_param = st.sidebar.slider(label='% for training', min_value=50, max_value=80, step=5, value=70)
    st.sidebar.header('Model Selection')
    default = st.sidebar.radio(label='', options=['Default', 'Train your own'])
    st.sidebar.header('Vectorization')
    #col1, col2 = st.sidebar.beta_columns(2)
    if default == 'Default':
        vectorization = st.sidebar.selectbox(label='', options=['CountVector', 'Tf-Idf'], index=1)
        st.sidebar.header('Voting type if ensemble')
        voting = st.sidebar.selectbox(label='', options=['soft', 'hard'], index=0)
        min_ngram = st.sidebar.number_input(label='min_ngrams',min_value=1, max_value=5, value=1)
        max_ngram = st.sidebar.number_input(label='max_ngrams', min_value=1, max_value=5, value=2)
    else:
        vectorization = st.sidebar.selectbox(label='', options=['CountVector', 'Tf-Idf'])
        st.sidebar.header('Voting type if ensemble')
        voting = st.sidebar.selectbox(label='', options=['soft', 'hard'])
        min_ngram = st.sidebar.number_input(label='min_ngrams',min_value=1, max_value=5)
        max_ngram = st.sidebar.number_input(label='max_ngrams', min_value=1, max_value=5)
    
    svm, nb, lr = False, False, False
    model_params = dict()
    if not default == 'Default':
        svm = st.sidebar.checkbox(label='SVM')
        nb = st.sidebar.checkbox(label='MultiNomial nb')
        lr = st.sidebar.checkbox(label='Logistic Regression')
        st.sidebar.header('Hyperparameter Tuning')
    else:
        #'''st.sidebar.header('Hyperparameter Tuning')
        #svm_kernel = st.sidebar.selectbox(label='SVM Kernel', options=['rbf', 'linear', 'poly', 'sigmoid'])
        #svm_c = st.sidebar.slider(label='SVM C', min_value=1.0, max_value=10.0)
        #svm_param = (svm_kernel, svm_c)
        #model_params['svm'] = svm_param
        #lr_c = st.sidebar.slider(label='LR C', min_value=1.0, max_value=10.0)
        #lr_solver = st.sidebar.selectbox(label='LR Solver', options=['newton-cg', 'sag', 'saga', 'lbfgs'])
        #lr_param = (lr_solver, lr_c)
        #model_params['lr'] = lr_param
        #nb_alpha = st.sidebar.slider(label='NB Alpha', min_value=1.0, max_value=10.0)
        #model_params['nb'] = nb_alpha'''
        pass


    if svm:
        svm_kernel = st.sidebar.selectbox(label='SVM Kernel', options=['rbf', 'linear', 'poly', 'sigmoid'])
        svm_c = st.sidebar.slider(label='SVM C', min_value=1.0, max_value=10.0)
        svm_param = (svm_kernel, svm_c)
        model_params['svm'] = svm_param
    if lr:
        lr_c = st.sidebar.slider(label='LR C', min_value=1.0, max_value=10.0)
        lr_solver = st.sidebar.selectbox(label='LR Solver', options=['newton-cg', 'sag', 'saga', 'lbfgs'])
        lr_param = (lr_solver, lr_c)
        model_params['lr'] = lr_param
    if nb:
        nb_alpha = st.sidebar.slider(label='NB Alpha', min_value=1.0, max_value=10.0)
        model_params['nb'] = nb_alpha

    if svm or lr or nb:
        params = {'train_param': train_param, 'vectorization':vectorization, 'model_params': model_params, 'min_ngram':min_ngram, 'max_ngram':max_ngram, 'voting':voting}
    else:
        params = {'train_param': train_param, 'vectorization':vectorization, 'min_ngram':min_ngram, 'max_ngram':max_ngram, 'voting':voting}

    return file, params


def display_model_summary(params):
    st.header('Training Parameters:')
    train_param = params['train_param']
    vec = params['vectorization']
    model = params.get('model_params', 'Default')
    attr, val = st.beta_columns(2)
    attr.text('Split %: ')
    attr.text('Vectorization: ')
    attr.text('Model: ')
    val.text('Train: '+str(train_param)+' Test: '+str(100-train_param))
    val.text(vec)
    if model == 'Default':
        val.text(model)
    else:
        for k, v in model.items():
            m = '**' + k + '**'
            val.markdown(m)
            val.text(v)

def create_clf(params):
    model = params.get('model_params', 'Default')
    if model == 'Default':
        #with open('./default/model.pkl', 'br') as f:
        #    clf = pickle.load(f)
        #f.close()
        clf = pickle.load(open('./default/model.pkl', 'rb'))
    else:
        if len(model)>1:
            estimator = []
            for k, v in model.items():
                if k == 'svm':
                    estimator.append(('SVC', SVC(kernel=v[0],
                                                  C=v[1],
                                                  probability = True)))
                elif k == 'lr':
                    estimator.append(('LR',
                                      LogisticRegression(solver=v[0],
                                                         C=v[1],
                                                         multi_class='multinomial')))
                else:
                    estimator.append(('nb', MultinomialNB(alpha=v)))
            clf = VotingClassifier(estimators=estimator, voting=params['voting'])

        else:
            st.text('Single')
            for k, v in model.items():
                if k == 'svm':
                    clf = SVC(kernel=v[0],C=v[1])
                elif k == 'lr':
                    clf=LogisticRegression(solver=v[0],C=v[1],multi_class='multinomial')
                else:
                    clf=MultinomialNB(alpha=v)
    return clf

def preprocess(X, min_ngram=1, max_ngram=2, vec='Tf-Idf', test=False, vocab=None):
    ps = PorterStemmer()
    Stopwords = set(stopwords.words('english'))
    X = X.apply(lambda x: re.sub(r'From:\s\S+@\S+', '', x))
    X = X.apply(lambda x: re.sub(r'To:\s\S+@\S+', '', x))
    X = X.apply(lambda x: re.sub(r'Cc:\s\S+@\S+', '', x))
    X = X.apply(lambda x: re.sub(r'Sent:\s.*', '', x))
    X = X.apply(lambda x: re.sub(r'Subject:\s\S+', '', x))
    X = X.apply(lambda x: re.sub('\S+@\S+', ' ', str(x)))
    X = X.apply(lambda x: re.sub('[^a-zA-Z]', ' ', str(x)))
    X = X.apply(lambda x: x.lower().split())
    X = X.apply(lambda x: ' '.join([ps.stem(word) for word in x if word not in Stopwords and word in english_words_set]))
    #X = X.apply(lambda x: ' '.join([word for word in x if word not in Stopwords and word in english_words_set]))
    #st.write(english_words_set)
    X = X.apply(lambda x: x.replace('original message', ''))

    if test:
        cv = TfidfVectorizer(ngram_range=(min_ngram, max_ngram), max_features=1000, vocabulary=vocab)
        X = cv.fit_transform(X).toarray()
        X = pd.DataFrame(X, columns=cv.get_feature_names())
        return X

    st.write(X)
    st.write(vec)
    if vec == 'Tf-Idf':
        cv = TfidfVectorizer(ngram_range=(min_ngram, max_ngram), max_features=1000)
    elif vec == 'CountVector':
        cv = CountVectorizer(ngram_range=(min_ngram, max_ngram), max_features=1000)

    X = cv.fit_transform(X).toarray()
    #pickle.dump(cv.vocabulary_, open("feature.pkl", "wb"))
    X = pd.DataFrame(X, columns=cv.get_feature_names())
    return X, cv.vocabulary_

def train(clf, x_train, x_test, y_train, y_test):
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    values = x_train.values
    #plt.scatter(x_train.shape, y)
    st.markdown("**Validation Accuracy :**")
    st.write(accuracy_score(y_test, pred))
    st.markdown("**Confusion Matrix:**")
    st.write(confusion_matrix(y_test, pred))
    st.markdown('**Classification Report**')
    st.text(classification_report(y_test, pred))
    #st.pyplot(plot_decision_regions(x_train.values, y_train.values, clf=clf))
    return clf

def load_data(params, fileName, train=True, default=True):
    train_param = params['train_param']
    vec = params['vectorization']
    min_ngram, max_ngram = params['min_ngram'], params['max_ngram']
    folder_name = fileName[:fileName.index('.')]
    cats = os.listdir('./'+folder_name)
    selectedCats = st.multiselect("Select Categories for Classification", cats)
    st.markdown(selectedCats)
    emails = []
    email_cat = []
    body = []
    for cat in selectedCats:
        for file in os.listdir('./'+folder_name+'/'+cat):
            emails.append(file)
            email_cat.append(cat)
            msg = Message('./' + folder_name + '/' + cat + '/' + file)
            content = msg.body
            #st.text(content)
            body.append(content)

    df = pd.DataFrame(data=list(zip(emails, body, email_cat)), columns=['email_name', 'body', 'category'])
    #df = sklearn.utils.shuffle(df)
    #df.reset_index(inplace=True)
    #df.drop(columns=['index'], inplace=True)
    if len(selectedCats)>1:
        st.dataframe(df.head())
        st.write(df.describe())

        X, feature_vec = preprocess(df['body'], min_ngram, max_ngram, vec)
        
        Y = df['category'].astype('category').cat.codes
        rt_cat_codes = dict(enumerate(df['category'].astype('category').cat.categories))
        cat_codes = json.dumps(rt_cat_codes)
        st.write(cat_codes)
        #st.write(json.dumps(cat_codes))
        #cat_index = pd.Dataframe(dict(enumerate(df['category'].astype('category').cat.categories)))
        #st.write(cat_index)
        #cat_json = cat_index.to_json(index=False)
        #st.write(cat_json)
        b64 = base64.b64encode(cat_codes.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/json;base64,{b64}" download="myfile.json">Download CategoiesIndex.json File</a>'
        st.markdown(href, unsafe_allow_html=True)
        #st.markdown('hello')
        return train_test_split(X, Y, test_size=(100 - train_param) / 100, random_state=42, shuffle=True), feature_vec, rt_cat_codes

#def load_new_data(fileName):


def extract_file(dataFile):
    print('inside')
    # print(os.listdir())
    file_name = dataFile.name
    print(file_name)
    f = open('./' + file_name, "wb")
    f.write(dataFile.read())
    f.close()
    print('done')
    print('./' + file_name)
    st.write(tqdm(pyunpack.Archive('./' + file_name).extractall('.')))
    st.write(file_name+' Extracted')



def make_predictions(predictFile, clf, cat_codes, feature_vec):
    extract_file(predictFile)
    folder_name = predictFile.name[:predictFile.name.index('.')]
    emails = []
    body = []
    for file in os.listdir('./'+folder_name):
        emails.append(file)
        msg = Message('./' + folder_name + '/'+ file)
        content = msg.body
        #st.text(content)
        body.append(content)

    X=preprocess(pd.Series(body), test=True, vocab=feature_vec)
    Y = list(clf.predict(X))
    try:
        Y = list(map(lambda x: cat_codes[str(x)], Y))
    except:
        Y = list(map(lambda x: cat_codes[int(x)], Y))
    df = pd.DataFrame(data=list(zip(emails, Y)), columns=['email_name', 'Predicted Category'])
    return df



def default_view(dataFile, clf):
    f = open('./default/cat_codes.json')
    cat_codes = json.load(f)
    f.close()
    feature_vec = pickle.load(open("./default/features.pkl", "rb"))
    if dataFile:
        try:
            (x_train, x_test, y_train, y_test), feature_vec, cat_codes = load_data(params, dataFile.name, train=True)
            st.text("Size of training data: "+str(len(x_train)))
            st.text("Size of Validation data: "+str(len(x_test)))


            if st.checkbox(label='Train'):
                clf = train(clf, x_train, x_test, y_train, y_test)

                output_model = pickle.dumps(clf)
                b64_model = base64.b64encode(output_model).decode()
                href_model = f'<a href="data:file/output_model;base64,{b64_model}" download="model.pkl">Download Trained Model .pkl File</a>'
                st.markdown(href_model, unsafe_allow_html=True)

                output_feature = pickle.dumps(feature_vec)
                b64_feature = base64.b64encode(output_feature).decode()
                href_feature = f'<a href="data:file/output_model;base64,{b64_feature}" download="features.pkl">Download Features .pkl File</a>'
                st.markdown(href_feature, unsafe_allow_html=True)
        except:
            pass
    predictFile = st.file_uploader(label='.zip containing folder of emails (.msg)')
    if predictFile != None:
        prediction = make_predictions(predictFile, clf, cat_codes, feature_vec)
        st.dataframe(prediction)
        output_csv = prediction.to_csv(index=False)
        b64_csv = base64.b64encode(output_csv.encode()).decode()
        href_csv = f'<a href="data:file/output_csv;base64,{b64_csv}" download="predictions.csv">Download predictions.csv File</a>'
        st.markdown(href_csv, unsafe_allow_html=True)


def own_model_view(dataFile, clf):
    if dataFile:
        try:
            (x_train, x_test, y_train, y_test), feature_vec, cat_codes = load_data(params, dataFile.name, train=True, default=False)
            st.text("Size of training data: "+str(len(x_train)))
            st.text("Size of Validation data: "+str(len(x_test)))

            if st.checkbox(label='Train'):
                clf = train(clf, x_train, x_test, y_train, y_test)

                output_model = pickle.dumps(clf)
                b64_model = base64.b64encode(output_model).decode()
                href_model = f'<a href="data:file/output_model;base64,{b64_model}" download="myfile.pkl">Download Trained Model .pkl File</a>'
                st.markdown(href_model, unsafe_allow_html=True)

                output_feature = pickle.dumps(feature_vec)
                b64_feature = base64.b64encode(output_feature).decode()
                href_feature = f'<a href="data:file/output_model;base64,{b64_feature}" download="myfile.pkl">Download Features .pkl File</a>'
                st.markdown(href_feature, unsafe_allow_html=True)

                predictFile = st.file_uploader(label='.zip containing folder of emails (.msg)')
                if predictFile != None:
                    prediction = make_predictions(predictFile, clf, cat_codes, feature_vec)
                    st.dataframe(prediction)
                    output_csv = prediction.to_csv(index=False)
                    b64_csv = base64.b64encode(output_csv.encode()).decode()
                    href_csv = f'<a href="data:file/output_csv;base64,{b64_csv}" download="predictions.csv">Download predictions.csv File</a>'
                    st.markdown(href_csv, unsafe_allow_html=True)
        
        except:
            pass



#st.multiselect("Select Categories for Classification", cats)
dataFile, params = get_params()
print(params)
print(dataFile)
try:
    extract_file(dataFile)
except Exception as e:
    print(e)
    pass

#x, y = load_data(params, dataFile.name)
#load_data(params, dataFile.name)
display_model_summary(params)
clf = create_clf(params)
print(clf)
st.text(clf)

if params.get('model_params', 'Default') == 'Default':
    default_view(dataFile, clf)
else:
    own_model_view(dataFile, clf)



# try:
#     if params.get('model_params', 'Default') == 'Default':
#         try:
            
#         except:
#             st.markdown('**Upload data if you want to train...**')
#     else:
#         (x_train, x_test, y_train, y_test), feature_vec = load_data(params, dataFile.name)
#         st.text("Size of training data: "+str(len(x_train)))
#         st.text("Size of Validation data: "+str(len(x_test)))

#     if dataFile:
#         if st.checkbox(label='Train'):
#             clf = train(clf, x_train, x_test, y_train, y_test)
#             st.markdown('**Import new Data**')

#             output_model = pickle.dumps(clf)
#             b64_model = base64.b64encode(output_model).decode()
#             href_model = f'<a href="data:file/output_model;base64,{b64_model}" download="myfile.pkl">Download Trained Model .pkl File</a>'
#             st.markdown(href_model, unsafe_allow_html=True)

#             output_feature = pickle.dumps(feature_vec)
#             b64_feature = base64.b64encode(output_feature).decode()
#             href_feature = f'<a href="data:file/output_model;base64,{b64_feature}" download="myfile.pkl">Download Features .pkl File</a>'
#             st.markdown(href_feature, unsafe_allow_html=True)



# except Exception as e:
#     print(e)
