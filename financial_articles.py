from gravityai import gravityai as grav
import pickle
import pandas as pd

model = pickle.load(open('financial_text_clasifier.pkl', 'rb'))
tfidf_vectorise = pickle.load(open('financial_text_vectoriser.pkl', 'rb'))
lael_encoder = pickle.load(open('financial_text_encoder.pkl','rb'))

#Reading of the csv data
def process(inPath,outPath):
    input_df = pd.read_csv(inPath)
    #term frequency vectoriser to vectorise the data
    feature_vector = tfidf_vectorise.transform(input_df['body'])  #uses to figure out which words weigh heavily in the csv file
    predicting_classes = model.predict(feature_vector) #predicts the classes of the csv file
    input_df['predicted_class'] = label_encoder.inverse_transform(predicting_classes) #adds the predicted class to the csv file
    #save the results to a csv file
    output_df = input_df[['id','category']]
    output_df.to_csv(outPath, index=False)

grav.wait_for_requests(process, 'input.csv', 'output.csv')
