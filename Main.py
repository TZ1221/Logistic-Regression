from MultivariateBernoulli import MultivariateBernoulli
from Multinomial import Multinomial
import pandas as pd
from LogisticRegression import LogisticRegression


def LoadData(filepath):
    data = pd.read_csv(filepath_or_buffer=filepath, header=None)
    return data.drop_duplicates()


if __name__ == '__main__':
    
    '''
    print('Start')
    print()
    print('Spambase Dataset - Logistic Regression')
    print()
    spambaseData = LoadData('spambase.csv')
    LogisticRegression_spambaseData = LogisticRegression(spambaseData, 0.75,0.00001)
    LogisticRegression_spambaseData.validate()
    print('======================================')
    
    print('Breast Cancer Dataset - Logistic Regression')
    print()
    breastCancerData = LoadData('breastcancer.csv')
    LogisticRegression_breastCancer = LogisticRegression( breastCancerData, 0.75, 0.00001)
    LogisticRegression_breastCancer.validate()
    print('======================================')
    
    print('Pima Indian Diabetes Dataset - Logistic Regression')
    print()
    diabetesData = LoadData( 'diabetes.csv')
    LogisticRegression_diabetes = LogisticRegression(diabetesData, 0.1,0.0000001)
    LogisticRegression_diabetes.validate()
    print('======================================')

    print('multivariateBernoulli')
    print()
    multivariateBernoulli = MultivariateBernoulli()
    multivariateBernoulli.run('train_data.csv', 'train_label.csv', 'test_data.csv','test_label.csv')
    print('======================================')
    '''
    
    print('Multinomial')
    print()
    multinomial = Multinomial()
    multinomial.run('train_data.csv', 'train_label.csv', 'test_data.csv', 'test_label.csv')
    print('END')