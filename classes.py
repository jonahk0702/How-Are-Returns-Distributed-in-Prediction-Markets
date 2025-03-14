## step 1 class Person



class People:
    def __init__(self):
        self.people = {}

    def add_person(self, unique_id, balance):
        self.people[unique_id] = Person(unique_id, balance)
    
    
    def potential_add_person(self, unique_id):
        if unique_id not in self.people:
            self.add_person(unique_id, 0)

    def plot_returns(self, exchange):
        """
        Plots the returns for each person in the exchange.
        
        :param exchange: StockExchange object.
        """

        returns = []
        for person in self.people.values():
            returns.append(person.get_total_return(exchange))
        

        plt.hist(returns, bins=20, weights=[100/len(returns)]*len(returns), edgecolor='black')


        plt.title('Histogram of Returns')
        plt.xlabel('Value')
        plt.ylabel('Frequency (%)')

        # Show the plot
        plt.show()



class Person:
    def __init__(self, unique_id, balance):
        self.id = unique_id
        self.balance = balance
        self.portfolio = {}  # {stock_symbol: {"shares": total_shares, "total_cost": total_cost}}
        self.total_paid = 0
        self.total_made = 0
        self.order_history = []


    def buy(self, stock, quantity, total_cost, timeStamp):
        
        self.balance -= total_cost
        
        if stock not in self.portfolio:
            self.portfolio[stock] = {"shares": 0, "total_cost": 0}
        
        self.portfolio[stock]["shares"] += quantity
        self.portfolio[stock]["total_cost"] += total_cost

        self.total_paid += total_cost
        self.order_history.append({'action':"buy", "stock":stock ,"quant":quantity, "cost":total_cost, "time":timeStamp})

    
    def sell(self, stock, quantity, total_revenue, timeStamp):
        if stock not in self.portfolio or self.portfolio[stock]["shares"] < quantity:
            #print("Ignoring as bought before the study period")
            #raise ValueError("Insufficient shares to sell.")
            print(f"SELLING {stock}")
            print(self.portfolio)
            pass

        else:
            print("REAL SELLLLL")
            
            # Calculate the average price per share
            average_price = self.portfolio[stock]["total_cost"] / self.portfolio[stock]["shares"]
            
            # Reduce the number of shares and adjust the total cost proportionally
            self.portfolio[stock]["shares"] -= quantity
            self.portfolio[stock]["total_cost"] -= average_price * quantity
            

            # Add the revenue to the balance
            self.balance += total_revenue

            self.total_made += total_revenue

            self.order_history.append({'action':"sell", "stock":stock, "quant":quantity, "cost":total_revenue, "time":timeStamp})
            
            # Remove the stock entry if all shares are sold
            if self.portfolio[stock]["shares"] == 0:
                del self.portfolio[stock]
            
            

    
    def get_portfolio_value(self, exchange):
        total_value = 0
        for stock, data in self.portfolio.items():
            stock_value = data["shares"] * exchange.stocks[stock].get_latest_price()[1]
            total_value += stock_value
        return total_value
    
    def get_average_price(self, stock):
        if stock in self.portfolio and self.portfolio[stock]["shares"] > 0:
            return self.portfolio[stock]["total_cost"] / self.portfolio[stock]["shares"]
        return 0
    
    def get_total_return(self, exchange):
        if self.total_paid == 0:
            return 0
        total_income = self.total_made + self.get_portfolio_value(exchange)
        total_expense = self.total_paid
        return (total_income - total_expense) / total_expense
    
    def calculate_holding_times(self):
        """
        Calculates the time between purchase and sale for each stock.
        Returns a list of tuples (stock, holding_time in seconds).
        """
        sorted_orders = sorted(self.order_history, key=lambda x: x["time"])
        buy_times = {}  # Store buy timestamps for each stock
        holding_times = []

        for order in sorted_orders:
            stock = order["stock"]
            if order["action"] == "buy":
                # Store the first buy timestamp for each stock
                if stock not in buy_times:
                    buy_times[stock] = order["time"]
            elif order["action"] == "sell" and stock in buy_times:
                holding_time = (order["time"] - buy_times[stock]).total_seconds()
                holding_times.append((stock, holding_time))
                # Remove stock after sell to match only the first buy
                del buy_times[stock]

        return holding_times



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