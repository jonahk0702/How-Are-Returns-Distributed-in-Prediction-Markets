import datetime


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

    def identify_news_events(self, threshold_percent=5, time_window_minutes=5):
        news_events = []
        import datetime
        
        for symbol, stock in self.stocks.items():
            prices = stock.get_prices()
            for i in range(1, len(prices)):
                prev_time, prev_price = prices[i-1]
                curr_time, curr_price = prices[i]
                
                # Convert timestamps to datetime objects if they're strings
                if isinstance(prev_time, str):
                    prev_time = datetime.datetime.strptime(prev_time, '%Y-%m-%d %H:%M:%S%z')  # Note the %z for timezone
                if isinstance(curr_time, str):
                    curr_time = datetime.datetime.strptime(curr_time, '%Y-%m-%d %H:%M:%S%z')  # Note the %z for timezone
                
                # Calculate time difference in minutes
                time_diff = (curr_time - prev_time).total_seconds() / 60
                
                # Calculate price change percentage
                price_change_pct = abs((curr_price - prev_price) / prev_price * 100)
                
                # If significant price change within the time window
                if price_change_pct >= threshold_percent and time_diff <= time_window_minutes:
                    news_events.append((symbol, curr_time, prev_price, curr_price))
        
        return news_events
