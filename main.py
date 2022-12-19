# Importing needed libraries
import numpy as np
import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval #module that converts a string of lists to a normal list

from flask import Flask,render_template,request, url_for
app = Flask(__name__)

df = pd.read_csv('Hotel_Reviews - full.csv')
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 5

# Replacing 'united kingdom' with 'UK' for easy use
df.Hotel_Address = df.Hotel_Address.str.replace('United Kingdom','UK')
# Splitting the hotel address and picking out the last string which would be the countries
df['countries'] = df.Hotel_Address.apply(lambda x: x.split(' ')[-1])
df.countries.unique() # All the hotels are located in six(6) countries 

# Dropping unneeded columns
df.drop(['Additional_Number_of_Scoring',
       'Review_Date','Reviewer_Nationality',
       'Negative_Review', 'Review_Total_Negative_Word_Counts',
       'Total_Number_of_Reviews', 'Positive_Review',
       'Review_Total_Positive_Word_Counts',
       'Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score',
       'days_since_review', 'lat', 'lng'],1,inplace=True)

#module that converts a string of lists to a normal list
from ast import literal_eval
#function to convert array of tags to string
def impute(col):
  col = col[0]
  if (type(col) != list):
    return "".join(literal_eval(col))
  else:
    return col
#using the function
df['Tags']  = df[['Tags']].apply(impute,axis=1)

def Input_your_destination_and_description(location,description):
    # Making these columns lowercase
    df['countries']=df['countries'].str.lower()
    df['Tags']=df['Tags'].str.lower()
    
    # Dividing the texts into small tokens (sentences into words)
    description = description.lower()
    description_tokens=word_tokenize(description)  
    
    sw = stopwords.words('english') # List of predefined english  stopwords to be used for computing
    lemm = WordNetLemmatizer() 
# We now define the functions below connecting these imported packages
    filtered_sen = {w for w in description_tokens if not w in sw}
    f_set=set()
    for fs in filtered_sen:
        f_set.add(lemm.lemmatize(fs))
    
    
    # Defining a new variable that takes in the location inputted and bring out the features defined below
    country_feat = df[df['countries']==location.lower()]
    country_feat = country_feat.set_index(np.arange(country_feat.shape[0]))
    l1 =[];l2 =[];cos=[];
    for i in range(country_feat.shape[0]):
        temp_tokens=word_tokenize(country_feat['Tags'][i])
        temp1_set={w for w in temp_tokens if not w in sw}
        temp_set=set()
        for se in temp1_set:
            temp_set.add(lemm.lemmatize(se))
        rvector = temp_set.intersection(f_set)
       

        cos.append(len(rvector))
    country_feat['similarity']=cos
    country_feat=country_feat.sort_values(by='similarity',ascending=False)
    country_feat.drop_duplicates(subset='Hotel_Name',keep='first',inplace=True)
    #country_feat.sort_values('Average_Score',ascending=False,inplace=True)
    country_feat.reset_index(inplace=True)
    #return country_feat[['Hotel_Name','Average_Score','Hotel_Address']].iloc[0:10,:].to_string(header=False,index=False)
    return country_feat[['Hotel_Name','Average_Score','Hotel_Address']].iloc[0:10,:]
    #ibm["Close"].to_string(header=False)


STOPWORDS = set(stopwords.words('english'))

@app.route('/', methods=['GET','POST'])
def hello():
    data=''
    if request.method == 'POST' and 'location' in request.form and 'description' in request.form:
        loc = request.form['location']
        desc = request.form['description']
        data = Input_your_destination_and_description(loc,desc)
        return render_template("result.html", column_names=data.columns.values,row_data=list(data.values.tolist()), zip=zip)
    return render_template("desc.html")

if __name__=="__main__":
    app.run(debug=True)