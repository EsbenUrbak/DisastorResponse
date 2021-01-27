# Disaster Response classifier
When a real disaster happens very large numbers of messages/tweets/facebook statuses will be sent to/picked up by response teams. It is generally a huge task just to decipher which are important and which are just comments about the situation. Figure Eight has provided a data set containing more than 30000 real messages sent to disaster response teams and they have also classifed each into various categories. The goal of this gibhub depository is to build a machine learning model that can catogorize such messages in the future helping disastor response team to quicker decipher such messages.

 The project is divided into three sections: 
1. **Data Processing**: Here I build an ETL (Extract, Transform, and Load) pipeline that processes messages and category data from CSV file and save them into an SQLite database
2. **Machine Learning**: Here I tokenize the text and process it using "Count Vectorizer" and "Tfidf" after which I train a Adaboost classifier
3. **Web development**: Here i setup a webpage where I show some basic graphs about the data and model accuracy. At the same time this webpage can also be used to test the model for any sentence
 
## Libraries
The project uses Python 3 and the following libraries:
for process_data.py:
-   [Pandas](http://pandas.pydata.org/)
-   [sqlalchemy](https://www.sqlalchemy.org/)
for train_classifier:
-   [Pandas](http://pandas.pydata.org/)
-   [sqlalchemy](https://www.sqlalchemy.org/)
-   [NumPy](http://www.numpy.org/)
-   [nltk](https://www.nltk.org/)
-   [scikit-learn](http://scikit-learn.org/stable/)
-   [Pickle](https://docs.python.org/3/library/pickle.html)
for the webapp:
-   [Pandas](http://pandas.pydata.org/)
-   [sqlalchemy](https://www.sqlalchemy.org/)
-   [Flask](https://flask.palletsprojects.com/en/1.1.x/)
-   [nltk](https://www.nltk.org/)

## Data
The dataset is provided by "Figure Eight" and consists of: 
-   **disaster_categories.csv**: message categories
-   **disaster_messages.csv**: multilingual disaster response messages

## Instructions:
To run the full setup locally please do the following:
0. Clone the depository
1. "ETL pipeline": Run the "process_data.py" file: python process_data.py   -> this will generate a SQL database
2. "machine learning": Run the train_classifier.py: python train_classifier.py    (if input is "True" the model will do a gridsearch. Otherwise it will just fit the model) -> this will train/fit the model.
3. "flash webapp": run the "run.py" in the apps folder  ->  This will setup a server which can be accessed by inputting the IP in your browser (please be aware if you  have anti-virus software this might block it. So if it doesnt work please try to disable)


## Results 
**Machine learning**:
The model gains a good accuracy both by category and also as an average of all. See table below in the appendix.

**The website**:
The web firstly shows 3 graphs:
1. Distribution of message genres
2. Distribution of caegories across messages
3. The accuracy of the model by catagory
It also allows the user to test out the model by inputting sentences which it will then categorize by highlighting that category in green.

## Acknowledgements
Thanks to "Figure Eight" for the dataset.

## Appendix:
Model stats
                  category  precision    recall    fscore  acccuracy
0                  related   0.887426  0.501268  0.439102   0.774981
1                  request   0.864888  0.656434  0.699747   0.875667
2                    offer   0.497521  0.500000  0.498757   0.995042
3              aid_related   0.732860  0.648944  0.641166   0.695843
4             medical_help   0.748379  0.513996  0.508548   0.923913
5         medical_products   0.929225  0.518081  0.521928   0.949275
6        search_and_rescue   0.987393  0.531915  0.553616   0.974828
7                 security   0.491228  0.500000  0.495575   0.982456
8                 military   0.733594  0.502792  0.497327   0.967010
9              child_alone   1.000000  1.000000  1.000000   1.000000
10                   water   0.893446  0.744070  0.798657   0.961289
11                    food   0.868151  0.827829  0.846396   0.942410
12                 shelter   0.884577  0.654749  0.710956   0.932304
13                clothing   0.801482  0.566233  0.605734   0.982456
14                   money   0.917305  0.524096  0.540057   0.977307
15          missing_people   0.494470  0.500000  0.497220   0.988940
16                refugees   0.817192  0.516847  0.524262   0.967201
17                   death   0.865107  0.513471  0.514139   0.952136
18               other_aid   0.682652  0.502970  0.470601   0.864607
19  infrastructure_related   0.966241  0.501408  0.485340   0.932494
20               transport   0.800976  0.523918  0.533133   0.950801
21               buildings   0.858282  0.566263  0.601797   0.950038
22             electricity   0.866031  0.515367  0.525130   0.981884
23                   tools   0.496854  0.500000  0.498422   0.993707
24               hospitals   0.744943  0.517989  0.531261   0.989512
25                   shops   0.497807  0.500000  0.498901   0.995614
26             aid_centers   0.493134  0.499903  0.496495   0.986079
27    other_infrastructure   0.477494  0.499900  0.488440   0.954805
28         weather_related   0.863826  0.703387  0.733623   0.824561
29                  floods   0.933484  0.724704  0.789389   0.947178
30                   storm   0.865467  0.583350  0.618847   0.915713
31                    fire   0.994563  0.516949  0.530054   0.989130
32              earthquake   0.943959  0.864186  0.899194   0.969680
33                    cold   0.935547  0.573200  0.621400   0.981884
34           other_weather   0.689749  0.505368  0.498468   0.950229
35           direct_report   0.834157  0.644054  0.679312   0.852975
Average Accuracy 0.9409430883973217
