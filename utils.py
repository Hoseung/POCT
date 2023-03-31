
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

import hashlib
import qrcode


lot_codes = {"AniCheck-bIgG": "BIG",
             "Other product": "new code"}

class LotCorrector():
    def __init__(self, fin, product, lotnum, degree = 7, exp_date = "231225", plot=False):
        self.fin = fin
        self.product = product
        self.lotnum = lotnum
        self.degree = degree
        self.exp_date = exp_date
        self.plot=plot
        self.fn_lot_correct = f"{product}_{lotnum}.dat"
        self.run()
        print("DONE!")

    def run(self):
        coeffs = self.calculate_coeffs()
        self.write_correction_coeff(coeffs, self.fn_lot_correct)
        self.saveQR()
    
    def saveQR(self):
        
        product = self.product 
        lotnum = self.lotnum
        exp_date = self.exp_date
        fn_lot_correct = self.fn_lot_correct
        try:
            h = hashlib.sha256(open(fn_lot_correct, 'rb').read()).hexdigest()
        except:
            print(f"Can't find the coeff file: {fn_lot_correct}")
            return 
        
        lot_code = lot_codes[product]
        
        qr_string = f"Proteomtech {product} {lot_code}{lotnum} {exp_date} {h[:10]}"
        
        
        img = qrcode.make(qr_string)
        fn_qr = self.get_fn_qr(product, lotnum)
        img.save(fn_qr)
        print(f"Saving QR image done: -> {fn_qr}")
        print(qr_string)

    
    def calculate_coeffs(self, degree = None):
        if degree is None:
            degree = self.degree

        df = pd.read_csv(self.fin)

        gt = df['original_value'].to_numpy()
        prediction = df['predicted_value'].to_numpy()

        pmodel = PolynomialFeatures(degree = degree)
        xpol = pmodel.fit_transform(prediction.reshape(-1, 1))
        preg = pmodel.fit(xpol,gt)

        # Builiding linear model
        lmodel = LinearRegression(fit_intercept = True)
        lmodel.fit(xpol, gt)


        # Fitting with linear model
        y_pred = lmodel.predict(preg.fit_transform(prediction.reshape(-1, 1)))

        if self.plot:
            # Plot results
            plt.scatter(prediction, gt, alpha=0.5)
            plt.plot(prediction, y_pred, color = 'red', label="correction function")
            plt.plot(gt,gt, label='1:1')
            plt.show()

        mse = mean_squared_error(gt, y_pred)
        r2 = r2_score(gt, y_pred)

        print("Fitting done:")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}\n")
        
        return lmodel.coef_        


    @staticmethod
    def write_correction_coeff(coefficients, fn="lot_correction.txt"):
        try:
            with open(fn, "w") as f:
                for vv in coefficients:
                    f.write(f"{vv:.14f}\n")
            print(f"writing Coeffs done: -> {fn}\n")
            return fn
        except:
            print("Something wrong")
        

    @staticmethod
    def fit_lin_model(x_train, y_train, degree=7):
        # building polynomial model
        polyModel = PolynomialFeatures(degree = degree)
        xpol = polyModel.fit_transform(x_train.reshape(-1, 1))
        preg = polyModel.fit(xpol,y_train)

        # Builiding linear model
        liniearModel = LinearRegression(fit_intercept = True)
        liniearModel.fit(xpol, y_train)
        
        return liniearModel, polyModel

    @staticmethod
    def get_fn_qr(product, lotnum):
        return f"{product} {lotnum}_qr.png"
    