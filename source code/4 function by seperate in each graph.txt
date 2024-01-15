import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting tools
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pandas as pd

# Corrected columns list
columns = ["Country", "GDP", "Percent_Literacy", "Infant_Mortality_Rate", "Life_Expectancy", "Some_Column_Name"]

# Original dataset
data = np.array([
    ["Nigeria", 3.0, 62.02, 72.24, 53.9, 52.89],
    ["Ethiopia", 7.0, 51.77, 35.37, 48.3, 65.37],
    ["Egypt", 3.9, 73.09, 16.65, 49.6, 70.99],
    ["DR Congo", 3.0, 80.02, 63.79, 47.9, 59.74],
    ["Tanzania", 4.8, 81.80, 34.72, 60.0, 66.41],
    ["South Africa", 5.2, 95.02, 25.78, 55.7, 65.25],
    ["Kenya", 9.3, 82.62, 31.15, 52.5, 62.68],
    ["Uganda", 3.8, 79.00, 31.86,  51.4, 62.85],
    ["Sudan", 4.7, 60.70, 39.92, 32.8, 65.61],
    ["Algeria", 9.0, 81.41, 19.46, 43.2, 74.45],
    ["Morocco", 4.6, 75.93, 16.02, 58.4, 73.92],
    ["Angola", 4.7, 72.28, 48.34, 53.0, 62.26],
    ["Ghana", 3.3, 80.38, 33.02, 58.0, 64.11],
    ["Mozambique", 7.4, 63.42, 52.77, 52.5, 61.17],
    ["Madagascar", 7.6, 77.25, 36.26, 58.9, 65.18],
    ["Côte d'Ivoire", 6.5, 89.89, 55.9, 60.4, 59.03],
    ["Cameroon", 9.9 , 78.23, 48.34, 51.9, 60.83],
    ["Niger", 5.0, 37.34, 45.61, 53.7, 61.45],
    ["Mali", 5.1, 30.76, 58.77, 54.5, 58.63],
    ["Burkina Faso", 5.8, 46.04, 52.82, 56.2, 59.73],
    ["Malawi", 2.1, 67.31, 29.02, 52.8, 63.72],
    ["Zambia", 4.5, 87.50, 41.66, 47.8, 62.38],
    ["Chad", 4.2, 26.76, 67.4, 52.0, 52.78],
    ["Somalia", 5.4, 5.40, 72.72, 2.9, 55.97],
    ["Senegal", 6.8, 56.30, 28.85, 57.7, 68.01],
    ["Zimbabwe", 4.8, 89.70, 37.93, 39.0, 61.12],
    ["Guinea", 7.9, 45.33, 61.99, 44.6, 59.33],
    ["Rwanda", 2.1, 75.90, 30.27, 52.2, 66.77],
    ["Benin", 2.9, 45.84, 56.54, 59.8, 60.09],
    ["Burundi", 5.1, 74.71, 38.64, 41.9, 61.57],
])

# Convert data to a pandas DataFrame for better handling
df = pd.DataFrame(data, columns=columns)

# Split the data into features (X) and target variable (y)
X = df[["GDP", "Percent_Literacy", "Infant_Mortality_Rate"]].astype(float)
y = df["Life_Expectancy"].astype(float)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
linear_pred = linear_model.predict(X)
linear_r2 = r2_score(y, linear_pred)

# Polynomial Regression (degree 2)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
poly_pred = poly_model.predict(X_poly)
poly_r2 = r2_score(y, poly_pred)

# Exponential Regression
X_exp = X["GDP"].values.reshape(-1, 1)  # Using only GDP for simplicity
y_exp = np.log(y)  # Log-transform the target variable for exponential regression
exp_model = LinearRegression()
exp_model.fit(X_exp, y_exp)
exp_pred = np.exp(exp_model.predict(X_exp))
exp_r2 = r2_score(y, exp_pred)

# Quadratic Function (degree 2)
quad_features = PolynomialFeatures(degree=2)
X_quad = quad_features.fit_transform(X)
quad_model = LinearRegression()
quad_model.fit(X_quad, y)
quad_pred = quad_model.predict(X_quad)
quad_r2 = r2_score(y, quad_pred)

# Visualize the predicted values with labels and connecting lines
fig = plt.figure(figsize=(15, 15))

# Linear Regression Plot
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(X["GDP"], X["Percent_Literacy"], y, color='blue', label='Actual Data (Original)')

# Labeling data points
for i, country in enumerate(df["Country"]):
    ax1.text(X["GDP"].iloc[i], X["Percent_Literacy"].iloc[i], y.iloc[i], country, color='blue')

ax1.scatter(X["GDP"], X["Percent_Literacy"], linear_pred, color='green', label=f'Predicted (Linear)')

# Connect actual and predicted points with lines
for i in range(len(X)):
    ax1.plot([X["GDP"].iloc[i], X["GDP"].iloc[i]], [X["Percent_Literacy"].iloc[i], X["Percent_Literacy"].iloc[i]],
             [y.iloc[i], linear_pred[i]], color='red', linestyle='dashed', alpha=0.5)

ax1.set_xlabel('GDP')
ax1.set_ylabel('Percent Literacy')
ax1.set_zlabel('Life Expectancy')
ax1.set_title(f'Linear Regression (R2={linear_r2:.3f})')  # Include R2 value in the title

# Print predicted values for Linear Regression
print("Linear Regression Predicted Values:")
print(pd.DataFrame({
    "Country": df["Country"],
    "Predicted Life Expectancy": linear_pred
}))

# Polynomial Regression Plot
ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(X["GDP"], X["Percent_Literacy"], y, color='blue', label='Actual Data (Original)')

# Labeling data points
for i, country in enumerate(df["Country"]):
    ax2.text(X["GDP"].iloc[i], X["Percent_Literacy"].iloc[i], y.iloc[i], country, color='blue')

ax2.scatter(X["GDP"], X["Percent_Literacy"], poly_pred, color='green', label=f'Predicted (Poly)')

# Connect actual and predicted points with lines
for i in range(len(X)):
    ax2.plot([X["GDP"].iloc[i], X["GDP"].iloc[i]], [X["Percent_Literacy"].iloc[i], X["Percent_Literacy"].iloc[i]],
             [y.iloc[i], poly_pred[i]], color='red', linestyle='dashed', alpha=0.5)

ax2.set_xlabel('GDP')
ax2.set_ylabel('Percent Literacy')
ax2.set_zlabel('Life Expectancy')
ax2.set_title(f'Polynomial Regression (R2={poly_r2:.3f})')  # Include R2 value in the title

# Print predicted values for Polynomial Regression
print("\nPolynomial Regression Predicted Values:")
print(pd.DataFrame({
    "Country": df["Country"],
    "Predicted Life Expectancy": poly_pred
}))

# Exponential Regression Plot
ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(X["GDP"], X["Percent_Literacy"], y, color='blue', label='Actual Data (Original)')

# Labeling data points
for i, country in enumerate(df["Country"]):
    ax3.text(X["GDP"].iloc[i], X["Percent_Literacy"].iloc[i], y.iloc[i], country, color='blue')

ax3.scatter(X["GDP"], X["Percent_Literacy"], exp_pred, color='green', label=f'Predicted (Exponential)')

# Connect actual and predicted points with lines
for i in range(len(X)):
    ax3.plot([X["GDP"].iloc[i], X["GDP"].iloc[i]], [X["Percent_Literacy"].iloc[i], X["Percent_Literacy"].iloc[i]],
             [y.iloc[i], exp_pred[i]], color='red', linestyle='dashed', alpha=0.5)

ax3.set_xlabel('GDP')
ax3.set_ylabel('Percent Literacy')
ax3.set_zlabel('Life Expectancy')
ax3.set_title(f'Exponential Regression (R2={exp_r2:.3f})')  # Include R2 value in the title

# Print predicted values for Exponential Regression
print("\nExponential Regression Predicted Values:")
print(pd.DataFrame({
    "Country": df["Country"],
    "Predicted Life Expectancy": exp_pred
}))

# Quadratic Regression Plot
ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(X["GDP"], X["Percent_Literacy"], y, color='blue', label='Actual Data (Original)')

# Labeling data points
for i, country in enumerate(df["Country"]):
    ax4.text(X["GDP"].iloc[i], X["Percent_Literacy"].iloc[i], y.iloc[i], country, color='blue')

ax4.scatter(X["GDP"], X["Percent_Literacy"], quad_pred, color='green', label=f'Predicted (Quadratic)')

# Connect actual and predicted points with lines
for i in range(len(X)):
    ax4.plot([X["GDP"].iloc[i], X["GDP"].iloc[i]], [X["Percent_Literacy"].iloc[i], X["Percent_Literacy"].iloc[i]],
             [y.iloc[i], quad_pred[i]], color='red', linestyle='dashed', alpha=0.5)

ax4.set_xlabel('GDP')
ax4.set_ylabel('Percent Literacy')
ax4.set_zlabel('Life Expectancy')
ax4.set_title(f'Quadratic Regression (R2={quad_r2:.3f})')  # Include R2 value in the title

# Print predicted values for Quadratic Regression
print("\nQuadratic Regression Predicted Values:")
print(pd.DataFrame({
    "Country": df["Country"],
    "Predicted Life Expectancy": quad_pred
}))

# Show the plots
plt.tight_layout()
plt.legend()
plt.show()

# Analyze the results
results = pd.DataFrame({
    "Model": ["Linear", "Polynomial", "Exponential", "Quadratic"],
    "R2": [linear_r2, poly_r2, exp_r2, quad_r2]
})

print("\nResults:")
print(results)
