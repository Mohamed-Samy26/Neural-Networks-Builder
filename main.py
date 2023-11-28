from views import ClassificationApp 
from models import LayerInfo as li, MultiLayerPrecepetron as mlp
import pandas as pd
import helpers.preprocessing as pp


def test():
    df = pd.read_excel('./Dry_Bean_Dataset.xlsx')
    df = pp.label_encode(df, 'Class')
    df = pp.imputation(df)
    df = pp.normalize_columns(df)
    df = pp.standardize_columns(df)
    train_x, test_x, train_y, test_y = pp.xy_split(df, 'Class')
    
    layers = [
        li.LayerInfo(True, 8, 'sigmoid'),
        li.LayerInfo(True, 17, 'sigmoid'),
        li.LayerInfo(True, 20, 'sigmoid'),
        li.LayerInfo(True, 3, 'sigmoid'),
    ]
    
    model = mlp.MultiLayerPrecepetron(5, 3,layers)
    
    model.train(train_x, train_y, 10, 0.01)
    
    acc = model.accuracy(test_x, test_y)
    
    print(acc)
    
    

if __name__ == "__main__":    
    app = ClassificationApp()
    app.run()
    # test()

    
    