## step 1 class Person

class Person:
    def __init__(self, unique_id, balance):
        self.id = unique_id
        self.balance = balance
        self.portfolio = {}  # {stock_symbol: {"shares": total_shares, "total_cost": total_cost}}
    
    def buy(self, stock, quantity, total_cost):
        #if total_cost > self.balance:
        #    raise ValueError("Insufficient balance to buy the shares.")
        
        self.balance -= total_cost
        
        if stock not in self.portfolio:
            self.portfolio[stock] = {"shares": 0, "total_cost": 0}
        
        self.portfolio[stock]["shares"] += quantity
        self.portfolio[stock]["total_cost"] += total_cost
        

    
    def sell(self, stock, quantity, total_revenue):
        if stock not in self.portfolio or self.portfolio[stock]["shares"] < quantity:
            #raise ValueError("Insufficient shares to sell.")

            pass

        else:
            
            # Calculate the average price per share
            average_price = self.portfolio[stock]["total_cost"] / self.portfolio[stock]["shares"]
            
            # Reduce the number of shares and adjust the total cost proportionally
            self.portfolio[stock]["shares"] -= quantity
            self.portfolio[stock]["total_cost"] -= average_price * quantity
            
            print(self.balance)
            # Add the revenue to the balance
            self.balance += total_revenue
            print("psot")
            print(self.balance)
            
            # Remove the stock entry if all shares are sold
            if self.portfolio[stock]["shares"] == 0:
                del self.portfolio[stock]
            
            

    
    def get_portfolio_value(self, current_prices):
        total_value = 0
        for stock, data in self.portfolio.items():
            stock_value = data["shares"] * current_prices[stock]
            total_value += stock_value
        return total_value
    
    def get_average_price(self, stock):
        if stock in self.portfolio and self.portfolio[stock]["shares"] > 0:
            return self.portfolio[stock]["total_cost"] / self.portfolio[stock]["shares"]
        return 0

## Stock Class
import matplotlib.pyplot as plt
from datetime import datetime

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
    

## AllStocks class

class StockPortfolio:
    def __init__(self):
        self.stocks = {}  # Maps stock symbols to StockPrices objects
    
    def add_stock(self, symbol):
        if symbol not in self.stocks:
            self.stocks[symbol] = StockPrices(symbol)
    
    def add_price(self, symbol, timestamp, price):
        if symbol not in self.stocks:
            self.add_stock(symbol)
        self.stocks[symbol].add_price(timestamp, price)
    
    def get_stock(self, symbol):
        return self.stocks.get(symbol, None)
    
    def plot_all_prices(self):
        for stock in self.stocks.values():
            stock.plot_prices()

# Example Usage
portfolio = StockPortfolio()