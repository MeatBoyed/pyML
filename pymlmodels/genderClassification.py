import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer


class GenderClassification:
    """
     This is the Gender Classification prediction module, 
     which makes use of the created Gender Classification
     Machine Learning model.
    
     :param dataFrame: teh dataFrame which it'll extract the names from
     :param column: the name of the column from the datafram to extract the names from

     :method predict - Will use the array of extracted names, and use the Machine Learning model
     to predict their gender. Will return a numpy array of its predictions correlating to the given
     names array.
     """

    # Initialize the Machine Learning model and Count Vectorizer
    try:
        model = joblib.load("./gender.joblib")
    except FileNotFoundError:
        print("########################################################## \n# genderClassification.joblib is not in root directory! #  \n##########################################################")
        raise

    vectorizer = CountVectorizer()

    # Import data set and create a Dictionary
    try:
        dataSet1 = pd.read_csv("./names_dataset.csv", sep=",")
        dataSet2 = pd.read_csv(
            "./universalnames_dataset.csv", sep=",")
        rawData = pd.concat([dataSet1, dataSet2])
    except FileNotFoundError:
        print("####################################################################### \n# One or two of the Data Sets can not be found in /datasets directory # \n#######################################################################")
        raise FileExistsError

    data = rawData.dropna()
    vectorizer.fit_transform(data["name"])

    def __init__(self, dataFrame, column="name"):
        names = dataFrame[column]
        self.namesCV = self.vectorizer.transform(names)

    def predict(self):
        self.answer = self.model.predict(self.namesCV)

if __name__ == "__main__":
    pass
