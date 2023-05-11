# in progress: In this cell, write your BaselineChallenge flow in the baseline_challenge.py file.

from metaflow import FlowSpec, step, Flow, current, Parameter, IncludeFile, card, current
from metaflow.cards import Table, Markdown, Artifact, Image
import numpy as np 
from dataclasses import dataclass

labeling_function =  lambda x: 1 if x.rating >=4 else 0 # done: Define your labeling function here.

@dataclass
class ModelResult:
    "A custom struct for storing model evaluation results."
    name: None
    params: None
    pathspec: None
    acc: None
    rocauc: None

class BaselineChallenge(FlowSpec):

    split_size = Parameter('split-sz', default=0.2)
    _test_ratio = 0.4
    data = IncludeFile('data', default='Womens Clothing E-Commerce Reviews.csv')
    kfold = Parameter('k', default=5)
    scoring = Parameter('scoring', default='accuracy')

    @step
    def start(self):

        import pandas as pd
        import io 
        from sklearn.model_selection import train_test_split
        
        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        # done: load the data in flowspec. 
        _df_raw = pd.read_csv(self.data)


        # filter down to reviews and labels 
        _df_raw.columns = ["_".join(name.lower().strip().split()) for name in _df_raw.columns]
        _df_raw = _df_raw[~_df_raw.review_text.isna()]

        _df_raw['review'] = _df_raw['review_text'].astype('str')
        _has_review_df = _df_raw[_df_raw['review_text'] != 'nan']

        reviews = _has_review_df['review_text']
        labels = _has_review_df.apply(labeling_function, axis=1)
        _df_clean = pd.DataFrame({'label': labels, **_has_review_df})

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({'review': reviews, 'label': labels})

        # features (X), label (y)
        X = _df.iloc[:, ~_df.columns.isin(['label'])]
        y = _df[['label']]

        "Split the data into training (fit), validation (trial run), test (final exam) sets"
        SEED = 89


        def train_validation_test_split (
            X, y, train_ratio: float, validation_ratio: float, test_ratio: float
        ):
            # Split up dataset into train and test, of which we split up the test.
            X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1- train_ratio), random_state=SEED
            )

            # Split up test into two (validation and test).
            X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=(test_ratio / (test_ratio + validation_ratio)), random_state=SEED,
            )

            # Return the splits
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        # split the data 80/20, or by using the flow's split-sz CLI argument
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = train_validation_test_split(
            X, y, 1.0-split_size, (1.0-_test_ratio)*(1.0-split_size), _test_ratio*(1.0-split_size)
        )
        
        print(f'num of rows in train set: {self.X_train.shape[0]}')
        print(f'num of rows in validation set: {self.X_val.shape[0]}')

        self.next(self.baseline, self.model)

    @step
    def baseline(self):
        "Compute the baseline"

        from sklearn.metrics import accuracy_score, roc_auc_score
        import pandas as pd
        from model import NbowModel
        
        self._name = "baseline"
        params = "Always predict 1"
        pathspec = f"{current.flow_name}/{current.run_id}/{current.step_name}/{current.task_id}"
        def getMajorityClass(df, col_label):
            return df[col_label].mode()[0]

        predictions = getMajorityClass(self.X_train, 'label') # done: predict the majority class
        # print(f'the majority label is {predictions}')


        acc = accuracy_score(self.y_train["label"],predictions) # done: return the accuracy_score of these predictions    

        # ideally use the defined method with the model class for rocauc calc but Im getting an understandable error
        # "ValueError: Only one class present in y_true. ROC AUC score is not defined in that case."
        # rocauc = model_test.eval_rocauc(
        #     X=_df['review'], 
        #     labels=pd.DataFrame({'label': [predictions] * _df.shape[0]})
        #  ) # done: return the roc_auc_score of these predictions
        # acc = model_test.eval_acc(
        #     X=self.X_train['review'], 
        #     labels=pd.DataFrame({'label': [predictions] * self.X_train.shape[0]})
        #     ) 

        # People have bypassed this with below but using the y_val dataset always. 
        # Update: NOW I know why (while debugging): this model method is not defined in this scope + don't need it.
        rocauc = roc_auc_score(self.X_train["label"],pd.DataFrame({'label': [predictions] * self.X_train.shape[0]}))
        # print(f'Accuracy of majority class model is {round(acc*100, 2)}%, AUC of {rocauc}')

        self.result = ModelResult("Baseline", params, pathspec, acc, rocauc)
        self.next(self.aggregate)

    @step
    def model(self):

        # done: import your model if it is defined in another file.
        from model import NbowModel
        import pandas as pd

        self._name = "model"
        # NOTE: If you followed the link above to find a custom model implementation, 
            # you will have noticed your model's vocab_sz hyperparameter.
            # Too big of vocab_sz causes an error. Can you explain why? 
        self.hyperparam_set = [{'vocab_sz': 100}, {'vocab_sz': 300}, {'vocab_sz': 500}]  
        pathspec = f"{current.flow_name}/{current.run_id}/{current.step_name}/{current.task_id}"

        self.results = []
        for params in self.hyperparam_set:
            model = NbowModel(vocab_sz=150) # done: instantiate your custom model here!
            model.fit(X=self.X_train['review'], y=self.y_train['label'])
            try:
                acc = model.eval_acc(self.X_test['review'], self.y_test['label'])# done: evaluate your custom model in an equivalent way to accuracy_score.
                rocauc = model.eval_rocauc(self.X_test['review'], self.y_test['label']) # done: evaluate your custom model in an equivalent way to roc_auc_score.
                self.results.append(ModelResult(f"NbowModel - vocab_sz: {params['vocab_sz']}", params, pathspec, acc, rocauc))
            except KeyError:
                self.results.append(ModelResult(f"NbowModel - vocab_sz: {params['vocab_sz']}", params, pathspec, None, None))

        self.next(self.aggregate)

    @card(type='corise')
    @step
    def aggregate(self, inputs):

        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import rcParams 
        rcParams.update({'figure.autolayout': True})

        rows = []
        violin_plot_df = {'name': [], 'accuracy': []}
        for task in inputs:
            if task._name == "model": 
                for result in task.results:
                    print(result)
                    rows, violin_plot_df = self.add_one(rows, result, violin_plot_df)
            elif task._name == "baseline":
                print(task.result)
                rows, violin_plot_df = self.add_one(rows, task.result, violin_plot_df)
            else:
                raise ValueError("Unknown task._name type. Cannot parse results.")
            
        current.card.append(Markdown("# All models from this flow run"))

        
        current.card.append(
            Table(
                rows,
                headers=["Model name", "Params", "Task pathspec", "Accuracy", "ROCAUC"]
            )
        )
        
        fig, ax = plt.subplots(1,1)
        plt.xticks(rotation=40)
        sns.violinplot(data=violin_plot_df, x="name", y="accuracy", ax=ax)
        
        # TODO: Append the matplotlib fig to the card
        # Docs: https://docs.metaflow.org/metaflow/visualizing-results/easy-custom-reports-with-card-components#showing-plots
        current.card.append(Image.from_matplotlib(fig))
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    BaselineChallenge()
