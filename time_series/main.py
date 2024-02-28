from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dates to Work With Python
today = datetime.now()
start_date = today - timedelta(days=5*365)
date_range = [start_date + timedelta(days=x) for x in range((today - start_date).days + 1)]
print(start_date, today, date_range[:5])

def generate_line(show=True):
    # Creating a line
    # Generate dummy data for x
    x = np.array(range(1, 11))

    # Assuming a simple linear relation y = 2x + 1
    # We generate y values based on our x values
    y = 2 * x + 1

    if show:
        plt.figure(figsize=(10, 6)) # Set the figure size for better visibility
        plt.plot(x, y, label='y = 2x + 1') # Plot x against y
        plt.title('Simple Line Plot') # Title of the plot
        plt.xlabel('X Axis') # Label for the X axis
        plt.ylabel('Y Axis') # Label for the Y axis
        plt.legend() # Show legend to identify the line
        plt.grid(True) # Show grid for better readability
        plt.show() # Display the plot

def generate_dummy_data():
    X = np.random.rand(100, 1) * 10  # 100 random X values
    y = 2 * X + 1 + np.random.randn(100, 1) * 2  # Corresponding y values with some noise
    return X, y

# X, y = generate_dummy_data()
# # Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initializing the Linear Regression model
# model = LinearRegression()

# # Fitting the model with the training data
# model.fit(X_train, y_train)

# # Predicting y values using the trained model
# y_pred = model.predict(X_test)

# # Plotting the results
# plt.figure(figsize=(10, 6))
# plt.scatter(X_train, y_train, color='blue', label='Training data') # Plot training data points
# plt.scatter(X_test, y_test, color='red', label='Testing data') # Plot testing data points
# plt.plot(X_test, y_pred, color='green', label='Linear regression line') # Plot the regression line
# plt.title('Linear Regression with Dummy Data')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend()
# plt.show()


# Displaying the coefficient and intercept of the line
# print(f"Coefficient (Slope): {model.coef_[0][0]}")
# print(f"Intercept: {model.intercept_[0]}")


class TimeTokenizer:
    def __init__(self, datetime):
        self.datetime_rx = "%Y-%m-%d %H:%M:%S:%f"
        self.datetime_str = datetime
        self.dt = self.parse_dt()
        
    def parse_dt(self):
        try:
            return datetime.strptime(self.datetime_str, self.datetime_rx)
        except ValueError as e:
            print(f"Error parsing datetime string: {e}")
            return None
    
    def tokenize_datetime(self):
        if not self.datetime_obj:
            return {}

        return {
            'year': self.datetime_obj.year,
            'month': self.datetime_obj.month,
            'day': self.datetime_obj.day,
            'hour': self.datetime_obj.hour,
            'minute': self.datetime_obj.minute,
            'second': self.datetime_obj.second
        }
        
    def dt_to_sec(self, other_dt):
        if not self.dt or not other_dt:
            return 0
        time_difference = (self.dt - other_dt).total_seconds()
        return int(time_difference)
    

dt_string = "2023-01-01 12:00:00:00"
tokenizer = TimeTokenizer(dt_string)
other_dt = datetime(2023, 1, 2, 12, 0, 0)
print("Seconds from parameter to self.dt:", tokenizer.dt_to_sec(other_dt))