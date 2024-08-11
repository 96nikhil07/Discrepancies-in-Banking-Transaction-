from flask import *
import numpy as np
import pandas as pd
from os import environ
from flask_cors import CORS
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth

app = Flask(__name__)
CORS(app)

mydf = pd.read_csv('1574188964.csv')


@app.route('/success', methods=['GET','POST'])  
def success():
    user_id=""
    Gf1=""
    if request.method == 'GET':
        user_id = request.args.get('CID')

    if request.method == 'POST':
        user_id = request.form['CID']

    print(user_id)
    if user_id!="":
        df = mydf[['Bid from', 'Bid to','Sale Price']]
        #print(df)
        df_filtered = df.loc[df['Bid from'] == int(user_id)]
        df_sorted = df_filtered.sort_values('Sale Price', ascending=False)
        top_10 = df_sorted.nlargest(10, 'Sale Price')
        #print(top_10)

        new_df = top_10[['Bid to']]
        nparr= np.unique(new_df.values)

        dataset = []
        dataset.append(nparr)

        for a in nparr:
            #print(a)
            user_id=a
            df_filtered1 = df.loc[df['Bid from'] == user_id]
            df_sorted1 = df_filtered1.sort_values('Sale Price', ascending=False)
            top_101 = df_sorted1.nlargest(10, 'Sale Price')
            new_df1 = top_101[['Bid to']]
            nparr1= np.unique(new_df1.values)
            dataset.append(nparr1)
         
        #print(dataset)

        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        #print(df)


        frequent_itemsets = fpgrowth(df, min_support=0.5, use_colnames=True)

        #print(frequent_itemsets['itemsets'])
        
        for a in range(len(frequent_itemsets['itemsets'])):
            #print(frequent_itemsets['itemsets'][a])
            f1=str(frequent_itemsets['itemsets'][a]).replace("frozenset({","")
            f1=f1.replace("})","")
            f1=f1.replace(" ","")
            Gf1=Gf1+f1+"#"
            
        print(Gf1)

    return Gf1


@app.route('/shutdown')
def shutdown():
    sys.exit()
    os.exit(0)
    return
   
if __name__ == '__main__':
   HOST = environ.get('SERVER_HOST', '0.0.0.0')
   try:
      PORT = int(environ.get('SERVER_PORT', '5555'))
   except ValueError:
      PORT = 5000
   #app.run(debug=True,HOST,PORT)
   #app.run(debug=True, host='0.0.0.0', port=5555)
   app.run(HOST,PORT)
   #app.run(debug=True)

