from flask import Flask,render_template,redirect,request
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
#__name__ == __main__
app = Flask(__name__)
M=joblib.load("twitter_model.pkl")
Vectorizer=joblib.load("trained_count_vectorizer_twitter.pkl")
@app.route('/')
def hello():
	return render_template("index.html")

@app.route('/', methods = ['POST'])
def tweet():
	if request.method == 'POST':
		  
		  tweet = [(request.form['tweet'])]
		  
		  transformed_tweet = Vectorizer.transform(tweet)
		  
		  result = M.predict(transformed_tweet)
	
	return render_template("index.html", your_sentiment = result[0])    
if __name__ == '__main__':
	app.run(debug=True)
