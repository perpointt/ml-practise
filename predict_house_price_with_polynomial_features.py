from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


def predict_house_price():
    data = fetch_california_housing(as_frame=True)
    x = data['data']
    y = data['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    model = LinearRegression()

    model.fit(x_train, y_train)

    pred = model.predict(x_test)

    mse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    print('--------------------------------')
    print('Results without Polynomial Features:')
    print(f'MSE: {round(mse, 2)}')
    print(f'R2: {round(r2, 2)}')

    pf = PolynomialFeatures(degree=2)

    pf.fit(x_train)

    x_train_new = pf.transform(x_train)
    x_test_new = pf.transform(x_test)

    model1 = LinearRegression()

    model1.fit(x_train_new, y_train)

    pred = model1.predict(x_test_new)

    mse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    print('')
    print('Results with Polynomial Features:')
    print(f'MSE: {round(mse, 2)}')
    print(f'R2: {round(r2, 2)}')
    print('--------------------------------')


if __name__ == '__main__':
    predict_house_price()
