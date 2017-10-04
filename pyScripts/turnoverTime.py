import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataloader as dl

# Returns the moving average of x with the average made over N points
def running_mean(x, N):
    return np.convolve(x, np.ones((N,)) / N)[(N - 1):]


# Creates a plot of the turnover over a given period of time on which
# the moving average (of x with the average made over N points) was performed on
def plot_turnover_over_time(filepath, ym1_1, ym1_2, ym2_1, ym2_2, N):
    col_list = ["turnover", "date"]
    df = dl.get_columns(
        dl.load_sales_files_ranges(filepath, ym1_1, ym1_2, ym2_1, ym2_2),
        col_list
    )

    x = df["date"]
    y = df["turnover"]

    newY = pd.DataFrame(running_mean(y, N))  # Moving average of turnover

    plt.plot(x, newY)

def save_std_plot():
    N = 20000
    # C:\Users\SorenM\Documents\GitHub\P5\GOFACT_DATA
    plot_turnover_over_time('C:/Users/SorenM/Documents/GitHub/P5/GOFACT_DATA/Sales_20', 1606, 1613, 1701, 1710, N)
    plt.savefig('%s.png' % "test")

def save_std_plot(N):
    # C:\Users\SorenM\Documents\GitHub\P5\GOFACT_DATA
    plot_turnover_over_time('C:/Users/SorenM/Documents/GitHub/P5/GOFACT_DATA/Sales_20', 1606, 1613, 1701, 1710, N)
    plt.savefig('%s.png' % "test")

def show_std_plot():
    N = 20000
    # C:\Users\SorenM\Documents\GitHub\P5\GOFACT_DATA
    plot_turnover_over_time('C:/Users/SorenM/Documents/GitHub/P5/GOFACT_DATA/Sales_20', 1606, 1613, 1701, 1710, N)
    plt.savefig('%s.png' % "test")

def show_std_plot(N):
    # C:\Users\SorenM\Documents\GitHub\P5\GOFACT_DATA
    plot_turnover_over_time('C:/Users/SorenM/Documents/GitHub/P5/GOFACT_DATA/Sales_20', 1606, 1613, 1701, 1710, N)
    plt.savefig('%s.png' % "test")


save_std_plot()