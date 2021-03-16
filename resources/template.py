import pandas as pd
import re
import numpy as np
from tqdm import notebook, tqdm
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBRegressor
import json


class BERTExplain:
	
	def __init__(self, args, logger, folder_path):
		self.args = args
		self.logger = logger
		self.folder_path = folder_path


	def load_stop_words(self, path="./lib/stopWords.txt"):
		"""
		Load the stop words for the explainability code alone
		Input: path to the stop words file
		Output: None
		"""
		file = open(path, 'r')
		self.stop_words = []
		for word in file.readlines():
			self.stop_words.append(word.strip('\n'))

	def find_text_cols(self, features):
		"""
		Find the name of the text column
		Input: Pandas dataframe contaiong all the featuers after feature selection
		Output: Name of column containing text
		"""
		for col in features.columns:
			if re.search(r'_emb\d{1,3}$', col):
				return re.sub(r'_emb\d{1,3}$', '', col)


	def create_augmented_df(self, text_col, features):
		"""
		Create an augmented dataframe where we combine the text column with all the text embeddings computed
		for that column.
		Input: Pandas series containg the text column from the dataframe
		Output: Pandas Dataframe with text column and embeddings combined
		"""
		
		features_to_be_dropped = []
		for col in features.columns:
			if not re.search(r'_emb\d{3}$', col):
				features_to_be_dropped.append(col)

		features = features.drop(features_to_be_dropped, axis=1)
		features['augmented_text_col'] = text_col
		return features


	def row_sampling(self, augmented_df):
		"""
		If the dataframe has too many rows, take a subset to compute explainability.
		Input: Pandas augmented dataframe with all the rows.
		Output: Pandas augmented dataframe with a subset of rows.
		"""
		if len(augmented_df) < 10000:
			return augmented_df

		non_empty_rows = augmented_df[ 
								(augmented_df['augmented_text_col'].notnull()) & \
								(df['augmented_text_col']!=u'') 
							]
		return non_empty_rows.sample(n = min(len(non_empty_rows), 10000))


	def create_document_term_matrix(self, text_sub):
		"""
		Create document-term matrix for the subset of text column from the augmented_df.
		Input: text column from augmented df
		Output: document_term_matrix for the text column
		"""
		vec = CountVectorizer(min_df=3, stop_words=self.stop_words)#, max_features=5000)
		X = vec.fit_transform(text_sub)
		df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
		return df

	def compute_important_words(self, fv):
		"""
		Compute the list of important words from the document_term matrix for the given feature vector
		Input: Feature vector
		Output: List of words which are important for that feature vector
		"""
		model = XGBRegressor()
		model.fit(self.document_term_matrix, fv)
		xgb_importance = model.feature_importances_

		imp_word_indices = xgb_importance.argsort()[-20:][::-1]
		res = []
		for ind in imp_word_indices:
			res.append(self.document_term_matrix.columns[ind])
		print("computed successfully")
		return res
		

	def explain(self, pandas_df, features, ho, data_io):
		"""
		Wrapper function to run the complete explainability pipeline
		pandas_df: Input dataframe with text columsn only
		features: Text features computed from BERT 
		"""
		self.load_stop_words()

		features = ho.h2o_to_pandas(features, self.folder_path)
		text_col_name = list(pandas_df.columns)[0]  # self.find_text_cols(features)

		augmented_df = self.create_augmented_df(pandas_df[text_col_name], features)
		augmented_df_sub = self.row_sampling(augmented_df)
		self.document_term_matrix = self.create_document_term_matrix(augmented_df_sub['augmented_text_col'])	

		output = {}
		for col in augmented_df_sub.columns:
			if re.search(r'_emb\d{3}$', col):
				output[col] = self.compute_important_words(augmented_df_sub[col])

		data_io.export_to_json(output, "BERTExplainability.json")
