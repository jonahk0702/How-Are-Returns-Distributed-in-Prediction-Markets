
import matplotlib.pyplot as plt

class StockPrices:
    def __init__(self, symbol):
        self.symbol = symbol
        self.prices = []  # List of tuples: [(timestamp, price), ...]
    
    def add_price(self, timestamp, price):
        """
        Adds a new price with a timestamp.
        
        :param timestamp: Datetime object representing the time of the sale.
        :param price: Sale price of the stock.
        """

        self.prices.append((timestamp, price))
        self.prices.sort(key=lambda x: x[0], reverse=False)

    
    def get_prices(self):
        """
        Returns the list of recorded prices.
        
        :return: List of tuples [(timestamp, price), ...].
        """
        return self.prices
    
    def plot_prices(self):
        """
        Plots the evolution of stock prices over time.
        """
        if not self.prices:
            print("No prices to plot.")
            return
        
        # Extract timestamps and prices
        timestamps, prices = zip(*self.prices)

        if len(prices) > 2000:
            print("Too many prices to plot. Only plotting the last 2000.")

        else:        
            # Plot prices
            plt.figure(figsize=(10, 5))
            plt.plot(timestamps, prices, marker="o", linestyle="-")
            plt.title(f"Price Evolution for {self.symbol}")
            plt.xlabel("Timestamp")
            plt.ylabel("Price")
            plt.grid(True)
            plt.show()
    
    def get_latest_price(self):
        """
        Retrieves the latest price based on timestamp.
        
        :return: Latest price or None if no prices exist.
        """
        if not self.prices:
            return None
        return self.prices[-1]
    

