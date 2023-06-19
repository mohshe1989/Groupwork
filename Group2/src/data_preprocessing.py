import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DataAnalyzer:
    def __init__(self, file):
        self.df = pd.read_csv(file, skipinitialspace=True, sep=',')

    def data_descriptions(self):
        print("name of columns of data")
        print(self.df.columns)
        print("---------------------------")
        print("---------------------------")
        print("Header of data")
        print(self.df.head())
        print("---------------------------")
        print("---------------------------")
        print("infomrmation about data ")
        print(self.df.info())
        print("---------------------------")
        print("---------------------------")
        print("nummber of th isnull vlaues ")
        print(self.df.isnull().sum())
        print("---------------------------")
        print("---------------------------")
        print("nummber of isna values" )
        print(self.df.isna().sum())
        print("---------------------------")
        print("---------------------------")
        print("statisical description of all the data ")
        print(self.df.describe())
        print("---------------------------")
        print("---------------------------")
        # columns counter the values 
        print("sum data depende on it values")
        col_liste = list(self.df.columns)
        for i in col_liste:
            print(self.df[i].value_counts())
            print("---------------------------")
            print("---------------------------")
            print("---------------------------")
            print("---------------------------")

    def preprocessing_and_clean_data(self):
        # delete the duplicates rows
        self.df.drop_duplicates()

        """ 
            # Converting Categorical Variables Into Numeric Values Using LabelEncoder & OneHotEncoder
            Labelcodig as the follow:
            # gender
                0- Female
                1- Male   
            -----------------------    
            # Smoking encoding
                0- No Info
                1- current
                2- not current
                3- former
                4- never
                5- ever
        """
        X_cat = self.df.drop(['age', 'bmi'], axis=1)
        X_num = self.df[['age', 'bmi']]
        y = self.df['smoking_history']
        le = LabelEncoder()
        y = le.fit_transform(y)
        for col in X_cat.columns:
            X_cat[col] = le.fit_transform(X_cat[col])
        X = pd.concat([X_num, X_cat], axis=1)
        return X

    def correlation(self):
        # calculate the correlation between all columns
        fig = plt.figure(figsize=(9, 9))
        sns.heatmap(self.df.corr(), annot=True)
        plt.show()
        return self.df.corr()


file = "diabetes_prediction_dataset.csv"
analyzer = DataAnalyzer(file)

# analyzer.data_descriptions()
print(analyzer.preprocessing_and_clean_data())
# analyzer.correlation()

