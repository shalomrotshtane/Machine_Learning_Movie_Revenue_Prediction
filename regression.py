import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
import json
from collections import Counter
from sklearn.preprocessing import StandardScaler
import pickle

################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################

SEPARATOR = '*'
NUM_TOP_OBSERVED_ELEMENTS = 50


def parse_col_vals(entry):
    """
    this function receives a data-frame entry of json form and converts it into an asterisk separated string
    for instance: [{'id': 12, 'name': 'Action'}, {'id': 13, 'name': 'Horror'}] -> 'Action*Horror'
    """
    if type(entry) != str and math.isnan(entry):
        return
    ret_val = ''
    try:
        entry_objects = json.loads(entry.replace('\"', '').replace('\'', '\"'))
    except json.decoder.JSONDecodeError:
        entry_objects = eval(entry)
    entry_objects = entry_objects if type(entry_objects) == list else [entry_objects]
    for obj in entry_objects:
        for k, v in obj.items():
            if k == 'name':
                ret_val += v + SEPARATOR
                break
    return ret_val[:-1] if ret_val != '' else ret_val


def col_top_appearance_count(col: pd.Series):
    col = col.dropna()
    col_as_list = SEPARATOR.join(col.tolist()).split(SEPARATOR)
    col_element_count = {val: col_as_list.count(val) for val in set(col_as_list)}
    return dict(Counter(col_element_count).most_common(NUM_TOP_OBSERVED_ELEMENTS))


def feature_average(data: pd.DataFrame, feature: str):
    """
    this function gets feature column and replaces all the index with the 0 budget to the average of all the other
    index with budget, we doing that cause we believe that if we have a lot of movies without a certain feature it can
    cause to a lot of noise.
    """
    values = data[feature]
    temp = (values != 0)
    average = values[temp].to_numpy().mean()
    return values.replace(0, average).replace(np.nan, average)


class PredictionModel:
    def __init__(self):
        self.revenue_model = GradientBoostingRegressor(criterion='mse', max_depth=6, max_features='sqrt',
                                                       min_samples_leaf=4, min_samples_split=9, n_estimators=110, loss='huber')
        self.score_model = GradientBoostingRegressor(criterion='mse', max_depth=6, max_features='sqrt',
                                                     min_samples_leaf=4, min_samples_split=9, n_estimators=110, loss='huber')
        self.scalar = StandardScaler()
        self.train_data, self.test_set, self.text_vectorizer, self.revenue_response_vec, \
        self.vote_response_vec, self.categorical_json_cols, self.text_vector = None, None, None, None, None, None, None

    def pre_process(self, data, mode='train'):
        data.dropna(subset=['release_date'], inplace=True)
        release_date = data['release_date']

        release_month = release_date.str.split('/', expand=True)[1]
        release_year = release_date.str.split('/', expand=True)[2]

        release_year = release_year.apply(lambda x: 'NONE' if (type(x) != str) else
        ('80s' if (1980 > int(x) >= 1970) else ('90s' if (1990 > int(x) >= 1980) else
                                                ('2000s' if (2000 > int(x) >= 1990) else (
                                                    '2010s' if (2010 > int(x) >= 2000) else
                                                    ('2020s' if (2020 > int(x) >= 2010) else 'NONE'))))))

        release_month_dummies = pd.get_dummies(release_month)
        release_year_dummies = pd.get_dummies(release_year)
        if 'NONE' in release_year_dummies.columns:
            release_year_dummies.drop(['NONE'], axis='columns', inplace=True)
        data = data.join(release_year_dummies).join(release_month_dummies)

        data.drop(['id', 'original_title', 'homepage'], axis='columns', inplace=True)
        data.dropna(
            subset=['overview', 'production_companies', 'production_countries',
                    'release_date', 'runtime', 'spoken_languages', 'status', 'title', 'cast', 'crew', 'original_language', 'genres'], inplace=True)
        if 'revenue' in data.columns:
            data.dropna(subset=['revenue'], inplace=True)
        if 'vote_average' in data.columns:
            data.dropna(subset=['vote_average'], inplace=True)

        data['original_language'] = data['original_language'].apply(lambda x: 1 if (x == "en") else 0)
        data['status'] = data['status'].apply(lambda x: 1 if (x == "Released") else 0)
        for feature in ['budget', 'vote_count']:
            data[feature] = feature_average(data=data, feature=feature)

        data.drop(['release_date', 'spoken_languages', 'title', 'cast', 'crew', 'tagline'], axis='columns',
                  inplace=True)

        # -------------------------- handle genres and collection categorical variable --------------------------
        categorical_json_cols = ['genres']
        for col in categorical_json_cols:
            data[col] = data[col].apply(parse_col_vals)
            data = pd.concat([data, data[col].str.get_dummies(sep=SEPARATOR)], axis=1)
            data = data.drop(columns=[col])

        # ---------------- handle categorical variables by creating a column for Top 10 values ----------------
        if mode == 'train':
            self.categorical_json_cols = {col: {} for col in
                                          ('belongs_to_collection', 'production_companies', 'production_countries',
                                           'keywords')}
        for col in self.categorical_json_cols.keys():
            data[col] = data[col].apply(parse_col_vals)
            if mode == 'train':
                self.categorical_json_cols[col] = col_top_appearance_count(data[col])
            new_col = data[col].apply(
                lambda s: len(set(s.split(SEPARATOR)).intersection(set(self.categorical_json_cols[col])))
                if type(s) == str else 0
            )
            new_col = new_col.rename(f"Top{NUM_TOP_OBSERVED_ELEMENTS} {col}")
            data = pd.concat([data, new_col], axis=1)
            data = data.drop(columns=[col])

        return data

    def pre_process_training_data(self):
        self.train_data = self.pre_process(self.train_data)

    def test_data_pre_process(self):
        self.test_set = self.pre_process(self.test_set, mode='test')

    def measure_feature_correlation(self, threshold, response_feature):
        data = self.train_data
        for feature in self.train_data.columns:
            if feature != response_feature:
                correlation = self.train_data[feature].corr(self.train_data[response_feature])
                print(f"feature {feature} is with correlation {correlation}")
                if abs(correlation) < threshold:
                    data = data.drop(columns=[feature])

    def fit(self, path, path1=None):
        if path1:
            self.train_data = pd.concat([pd.read_csv(path), pd.read_csv(path1)])
        else:
            self.train_data = pd.read_csv(path)
        self.pre_process_training_data()
        self.revenue_response_vec = self.train_data['revenue']
        self.train_data = self.train_data.drop(columns=['revenue'])
        self.vote_response_vec = self.train_data['vote_average']
        self.train_data = self.train_data.drop(columns=['vote_average'])
        # drop features with low parson correlation
        # df = self.measure_feature_correlation(threshold=0.1, response_feature='revenue')
        self.train_data['overview'] = self.train_text_model()
        self.revenue_model.fit(self.train_data.to_numpy(), self.revenue_response_vec.to_numpy())  # train the revenue model
        self.score_model.fit(self.train_data.to_numpy(), self.vote_response_vec.to_numpy())  # train the vote average model

    def predict(self, csv_file):
        """
        This function predicts revenues and votes of movies given a csv file with movie details.
        Note: Here you should also load your model since we are not going to run the training process.
        :param csv_file: csv with movies details. Same format as the training dataset csv.
        :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
        """
        self.test_set = pd.read_csv(csv_file)
        self.test_data_pre_process()
        self.test_set['overview'] = self.train_text_model(mode='test')
        df = pd.DataFrame(0, index=np.arange(self.test_set.shape[0]), columns=self.train_data.columns)
        df.update(self.test_set)
        vote_prediction = self.score_model.predict(df).tolist()
        revenue_prediction = self.revenue_model.predict(df).tolist()
        return revenue_prediction, vote_prediction

    def train_text_model(self, mode='train'):
        """
        this function trains a model for text-overview score generation
        """
        if mode == 'train':
            tfidf = TfidfVectorizer(smooth_idf=False)
            X = tfidf.fit_transform(self.train_data['overview'])
            self.text_vectorizer = tfidf
            M = np.mean(X, axis=0)
            self.text_vector = M
            X = X @ M.T
        else:
            X = self.text_vectorizer.transform(self.test_set['overview'])
            X = X @ self.text_vector.T
        return X


class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "regression"
        return super().find_class(module, name)


def save_params(model):
    with open("tuple_model.pkl", 'wb') as file:
        pickle.dump(model, file)


def reload_params():
    with open("tuple_model.pkl", 'rb') as file:
        model = MyCustomUnpickler(file).load()
    return model


def predict(csv_file):
    model = reload_params()
    return model.predict(csv_file)
