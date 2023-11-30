import numpy as np
from views import ClassificationApp
from models import LayerInfo as li, MultiLayerPrecepetron as mlp
import pandas as pd
import helpers.preprocessing as pp
import pickle as pl
import controllers.mlp_controller as mlp


if __name__ == "__main__":
    # app = ClassificationApp()
    # app.run()
    mlp.infer_mlp([li.LayerInfo(True, 5, "hidden", "sigmoid")])