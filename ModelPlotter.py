import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class ModelPlotter:
    @staticmethod
    def plot_accuracies(results):
        accuracies = [result['accuracy'] for result in results.values()]
        model_names = list(results.keys())

        plt.figure(figsize=(10, 6))
        plt.bar(model_names, accuracies, color='skyblue')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.show()

    def plot_portfolio(portfolio_values, title="Portfolio Performance", initial_balance=10000):
        """
        Plots the portfolio value over time.

        :param portfolio_values: List of portfolio values over time.
        :param title: Title of the plot (default: "Portfolio Performance").
        :param initial_balance: The starting portfolio value for reference (default: 10000).
        """
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values, label="Portfolio Value", color="blue", linewidth=2)

        # Plot a horizontal line for the initial balance as a reference
        plt.axhline(initial_balance, color="gray", linestyle="--", label="Initial Balance")

        # Title and labels
        plt.title(title, fontsize=16)
        plt.xlabel("Time (Data Points)", fontsize=12)
        plt.ylabel("Portfolio Value (USD)", fontsize=12)

        # Grid and legend
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=12)

        # Show the plot
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_predictions(data, predictions):
        # Filter out neutral signals for plotting
        data = data[data['Signal'] == 1]

        # Main figure
        plt.figure(figsize=(14, 7))

        # Plotting the entire data
        plt.plot(data.index, data['close'], label="Close Price", color="blue", alpha=0.7)

        # Adding Buy signals
        plt.scatter(data.loc[data['Signal'] == 1].index, 
                    data.loc[data['Signal'] == 1]['close'], 
                    label="Buy Signal", marker="^", color="green", alpha=1, s=100)

        # Adding Sell signals
        plt.scatter(data.loc[data['Signal'] == -1].index, 
                    data.loc[data['Signal'] == -1]['close'], 
                    label="Sell Signal", marker="v", color="red", alpha=1, s=100)

        # Title and labels for the main plot
        plt.title("Stock Price with Buy/Sell Signals (Full Range + Last Week Inset)")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()

        # Add an inset plot for the last week's data
        last_week_data = data.tail(7)  # Filter data for the last 7 days
        ax_inset = inset_axes(plt.gca(), width="40%", height="40%", loc="upper center")  # Adjust size and location
        ax_inset.plot(last_week_data.index, last_week_data['close'], label="Close Price (Last Week)", color="blue", alpha=0.7)

        # Adding Buy signals for the inset
        ax_inset.scatter(last_week_data.loc[last_week_data['Signal'] == 1].index, 
                        last_week_data.loc[last_week_data['Signal'] == 1]['close'], 
                        label="Buy Signal", marker="^", color="green", alpha=1, s=80)

        # Adding Sell signals for the inset
        ax_inset.scatter(last_week_data.loc[last_week_data['Signal'] == -1].index, 
                        last_week_data.loc[last_week_data['Signal'] == -1]['close'], 
                        label="Sell Signal", marker="v", color="red", alpha=1, s=80)

        ax_inset.set_title("Last Week", fontsize=10)
        ax_inset.set_xlabel("Date", fontsize=8)
        ax_inset.set_ylabel("Close Price", fontsize=8)
        ax_inset.tick_params(axis="both", which="major", labelsize=8)

        plt.show()

