import scipy.stats


from datetime import datetime
from collections import defaultdict
import datetime
import pytz
from dateutil import parser
import numpy as np
from dateutil import parser
from collections import deque
import datetime

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
        #if stock not in self.portfolio or self.portfolio[stock]["shares"] < quantity:
        #    #print("Ignoring as bought before the study period")
        #    #raise ValueError("Insufficient shares to sell.")

        #    pass

        #else:
        if 1 ==1:
            # the idea is return looks back and enperates this!
            
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
    
    def realized_return_ratio(self):
        """
        Calculate the Realized Return Ratio.
        Formula: (Total Proceeds from Sales − Total Purchase Cost) / Total Purchase Cost
        """
        if self.total_paid == 0:
            return 0  # Avoid division by zero
        return (self.total_made - self.total_paid) / self.total_paid
    
     
    def mark_to_market_return(self, exchange):
        """
        Calculate the Mark-to-Market Return.
        Formula: (Current Value of Holdings + Total Sales Proceeds) / Total Purchase Cost
        """
        if self.total_paid == 0:
            return 0  # Avoid division by zero
        portfolio_value = self.get_portfolio_value(exchange)
        return (portfolio_value + self.total_made) / self.total_paid
    

    def per_trade_profitability(self):
        """
        Calculate Per-Trade Profitability.
        Formula: (Sell Price − Buy Price) / Buy Price (averaged across all trades)
        """
        total_profitability = 0
        trade_count = 0

        for trade in self.order_history:
            if trade['action'] == 'sell':
                stock = trade['stock']
                sell_price = trade['cost'] / trade['quant']
                buy_price = self.get_average_price(stock)
                if buy_price > 0:  # Avoid division by zero
                    total_profitability += (sell_price - buy_price) / buy_price
                    trade_count += 1

        if trade_count == 0:
            return 0  # Avoid division by zero

        return total_profitability / trade_count
    
    def return_on_invested_capital(self, exchange):
        """
        Calculate Return on Invested Capital (ROIC).
        Formula: (Total Realized Profit + Unrealized Profit) / Total Capital Invested
        """
        if self.total_paid == 0:
            return 0  # Avoid division by zero
        
        portfolio_value = self.get_portfolio_value(exchange)
        total_profit = (self.total_made - self.total_paid) + portfolio_value
        return total_profit / self.total_paid
    
    def outcome_adjusted_return(self, correct_outcomes=None):
        """
        Calculate Outcome-Adjusted Returns for resolved markets.
        Formula: (Shares in Correct Outcome × $1.00 + Sales Proceeds) / Purchase Cost
        
        Parameters:
        correct_outcomes: Dictionary mapping stock symbols to boolean values 
                        indicating if they were correct outcomes (True/False)
        """
        if self.total_paid == 0:
            return 0  # Avoid division by zero

        resolved_value = 0
        
        # If correct_outcomes is provided, use it to adjust the value
        if correct_outcomes is not None:
            for stock, data in self.portfolio.items():
                if stock in correct_outcomes:
                    # If correct outcome, value at $1.00 per share, else $0.00
                    if correct_outcomes[stock]:
                        resolved_value += data["shares"] * 1.00
        else:
            # Default: assume all holdings are correct outcomes
            for stock, data in self.portfolio.items():
                resolved_value += data["shares"] * 1.00

        total_income = resolved_value + self.total_made
        return total_income / self.total_paid
    
    def time_weighted_return(self):
        """
        Calculate Time-Weighted Return based on order history.
        Formula: Compounded growth rate of capital over periods.
        """
        if len(self.order_history) < 2:
            return 0
            
        # Sort order history by timestamp
        sorted_history = sorted(self.order_history, key=lambda x: x['time'])
        
        # Calculate period returns
        period_returns = []
        current_value = 0
        
        for i in range(1, len(sorted_history)):
            prev_trade = sorted_history[i-1]
            curr_trade = sorted_history[i]
            
            # Calculate value change between trades
            prev_value = current_value
            
            if prev_trade['action'] == 'buy':
                current_value -= prev_trade['cost']
            else:  # sell
                current_value += prev_trade['cost']
                
            # Only calculate return if we have a meaningful previous value
            if abs(prev_value) > 0:
                period_return = (current_value - prev_value) / abs(prev_value)
                period_returns.append(period_return)
        
        # Calculate compounded return
        twr = 1.0
        for r in period_returns:
            twr *= (1 + r)
            
        return twr - 1
    
    def profit_factor(self):
        """
        Calculate Profit Factor.
        Formula: Total Gains / Total Losses
        """
        total_gains = 0
        total_losses = 0

        for trade in self.order_history:
            if trade['action'] == 'sell':
                stock = trade['stock']
                sell_price = trade['cost'] / trade['quant']
                # Find corresponding buy price - using average price as approximation
                avg_price = self.get_average_price(stock)
                profit = (sell_price - avg_price) * trade['quant']
                
                if profit > 0:
                    total_gains += profit
                else:
                    total_losses += abs(profit)

        if total_losses == 0:
            return float('inf')  # No losses means infinite profit factor

        return total_gains / total_losses
    
    
    def get_total_return(self, exchange):


        if self.total_paid == 0:
            return 0
        
        total_income = self.total_made + self.get_portfolio_value(exchange)
        total_expense = self.total_paid
        return (total_income - total_expense) / total_expense
    
    def calc_simple_cash_multiple(self, portfolio):
        """
        orders: list of dicts, each with:
        """
        total_spent = 0.0
        total_proceeds = 0.0
        holdings = {}  # {symbol: net quantity still held}

        # 1. Aggregate buys/sells
        for order in self.order_history:
            symbol = order["stock"]
            action = order["action"]
            quant = order["quant"]
            cost  = order["cost"]

            if action.lower() == "buy":
                total_spent += cost
                holdings[symbol] = holdings.get(symbol, 0) + quant
            elif action.lower() == "sell":
                total_proceeds += cost
                holdings[symbol] = holdings.get(symbol, 0) - quant

        # 2. Calculate final notional value of open positions
        total_remaining_value = 0.0
        for symbol, qty in holdings.items():
            if qty > 0:
                stock_obj = portfolio.get_stock(symbol)
                # For simplicity, use the latest recorded price
                latest_price = stock_obj.get_latest_price()[1] if stock_obj.get_latest_price() else 0.0
                total_remaining_value += qty * latest_price

        # Edge case: if no buy orders or zero total_spent => handle separately
        if total_spent <= 1e-12:
            return float("inf") if (total_proceeds + total_remaining_value) > 0 else 0

        return (total_proceeds + total_remaining_value) / total_spent
    

    

    def calc_cost_basis_returns(self, portfolio):
        """
        Maintains an average cost basis per symbol.
        When sells occur, realize gains vs. that average cost.
        Finally, compute unrealized gains on leftover shares using the latest price.
        """
        from collections import defaultdict
        # For each symbol, track total shares and total cost
        shares_held = defaultdict(float)
        cost_held   = defaultdict(float)
        realized_gains = 0.0

        for order in self.order_history:
            symbol = order["stock"]
            action = order["action"].lower()
            qty    = order["quant"]
            cost   = order["cost"]  # total cost of transaction

            if action == "buy":
                # Update average cost
                # current avg cost = cost_held[symbol] / shares_held[symbol] (if shares_held>0)
                # new total cost = cost_held[symbol] + cost
                # new shares     = shares_held[symbol] + qty
                total_shares_before = shares_held[symbol]
                total_cost_before   = cost_held[symbol]

                new_shares = total_shares_before + qty
                new_cost   = total_cost_before + cost

                shares_held[symbol] = new_shares
                cost_held[symbol]   = new_cost

            elif action == "sell":
                # Realize gain = (sell price per share - average cost) * qty
                if shares_held[symbol] > 0:
                    avg_cost = cost_held[symbol] / shares_held[symbol]
                else:
                    avg_cost = 0.0

                # Price per share from this transaction:
                # if cost is total revenue from the sell:
                #   price_per_share = cost / qty
                # realized gain:
                realized_gains += (cost / qty - avg_cost) * qty

                # Now reduce cost basis proportionally
                shares_held[symbol] -= qty
                cost_held[symbol]   -= avg_cost * qty

        # Now compute unrealized gains for leftover positions
        unrealized_gains = 0.0
        for symbol, qty in shares_held.items():
            if qty > 0:
                avg_cost = cost_held[symbol] / qty
                stock_obj = portfolio.get_stock(symbol)
                latest_price = stock_obj.get_latest_price()[1] if stock_obj.get_latest_price() else 0.0
                unrealized_gains += (latest_price - avg_cost) * qty

        total_gains = realized_gains + unrealized_gains
        total_cost = sum(cost_held.values()) + (realized_gains)  # total cost that was originally put in
        # The above might need adjustments depending on how you define "total capital"

        # Return ratio:
        if total_cost <= 1e-12:
            return float("inf") if total_gains > 0 else 0

        return total_gains / total_cost
    

    
    def calculate_average_holding_time(self):
        """
        Calculates the average holding time across all stocks that were both bought and sold.
        Returns the average time in days.
        """
        from dateutil import parser
        
        sorted_orders = sorted(self.order_history, key=lambda x: x["time"])
        buy_times = {}  # Store buy timestamps for each stock
        holding_seconds = []  # Store holding times in seconds

        for order in sorted_orders:
            stock = order["stock"]
            
            # Ensure timestamp is a datetime object
            timestamp = order["time"]
            if isinstance(timestamp, str):
                try:
                    timestamp = parser.parse(timestamp)
                except:
                    continue
            
            if order["action"] == "buy":
                if stock not in buy_times:
                    buy_times[stock] = timestamp
            elif order["action"] == "sell" and stock in buy_times:
                try:
                    holding_time = (timestamp - buy_times[stock]).total_seconds()
                    holding_seconds.append(holding_time)
                    del buy_times[stock]
                except Exception:
                    continue

        # Calculate average in days
        if not holding_seconds:
            return 0
        
        avg_seconds = sum(holding_seconds) / len(holding_seconds)
        avg_days = avg_seconds / (60 * 60 * 24)  # Convert seconds to days
        
        return avg_days


    
    def calculate_momentum_score(self, exchange, lookback_period=5, lambda_smoothing=1):
        """
        Calculates a momentum score for the person based on their order history.
        
        A positive score indicates momentum trading (buying rising stocks, selling falling ones).
        A negative score indicates contrarian trading (buying falling stocks, selling rising ones).
        
        :param exchange: StockPortfolio object containing historical prices
        :param lookback_period: Number of price points to look back to determine trend
        :param lambda_smoothing: Smoothing constant for small sample correction
        :return: Momentum score (float)
        """
        momentum_score = 0
        trade_count = 0
        
        for order in self.order_history:
            stock = order["stock"]
            action = order["action"]
            timestamp = order["time"]
            
            # Skip if stock doesn't exist in exchange
            if stock not in exchange.stocks:
                continue
                
            # Get stock price history before this order
            stock_prices = exchange.stocks[stock].get_prices()
            
            # Find prices before the current timestamp
            prior_prices = [(t, p) for t, p in stock_prices if t < timestamp]
            
            if len(prior_prices) >= lookback_period:
                # Get the most recent prices before the order
                recent_prices = prior_prices[-lookback_period:]
                
                # Calculate simple trend: comparing last price to first price in the window
                first_price = recent_prices[0][1]
                last_price = recent_prices[-1][1]
                
                is_uptrend = last_price > first_price
                
                # Score based on action and trend
                if action == "buy" and is_uptrend:
                    momentum_score += 1  # Buying in uptrend (momentum behavior)
                elif action == "buy" and not is_uptrend:
                    momentum_score -= 1  # Buying in downtrend (contrarian behavior)
                elif action == "sell" and is_uptrend:
                    momentum_score -= 1  # Selling in uptrend (contrarian behavior)
                elif action == "sell" and not is_uptrend:
                    momentum_score += 1  # Selling in downtrend (momentum behavior)
                
                trade_count += 1
        
        # Apply Laplace smoothing to avoid extreme values for small sample sizes
        smoothed_score = (momentum_score + lambda_smoothing) / (trade_count + 2 * lambda_smoothing)

        return smoothed_score

    
    def calculate_risk_tolerance(self, lambda_smoothing=1):
        """
        Calculates a risk tolerance score based on the prices of assets purchased.
        
        In Polymarket context:
        - Low-priced assets (near 0) are considered high risk (long shots)
        - High-priced assets (near 1) are considered low risk (sure bets)
        
        Returns a score from 0 to 1:
        - 0: Extremely risk-averse (only buys high-priced assets)
        - 1: Extremely risk-tolerant (only buys low-priced assets)
        """
        buy_orders = [order for order in self.order_history if order['action'] == 'buy']
        
        if not buy_orders:
            return 0.5  # Neutral score if no buy orders
        
        weighted_risk_score = 0
        total_spent = 0
        
        for order in buy_orders:
            quantity = order['quant']
            total_cost = order['cost']
            
            if quantity > 0:
                price_per_share = total_cost / quantity
                
                # Risk score: Inverse of price (low price = high risk = high score)
                risk_score = 1 - price_per_share
                
                # Weight by amount spent
                weighted_risk_score += risk_score * total_cost
                total_spent += total_cost
    
        # Apply Laplace smoothing to avoid extreme values
        if total_spent > 0:
            smoothed_score = (weighted_risk_score + lambda_smoothing) / (total_spent + 2 * lambda_smoothing)
            return min(max(smoothed_score, 0.01), 0.99)  # Keep within reasonable range
        
        return 0.5  # Default neutral score


   

    def calculate_intraday_closeout_rate(self, lambda_smoothing=1, global_median=None):
        """
        Calculates the percentage of trading days where positions opened
        were closed within the same day.
    
        Returns a value from 0 to 1:
        - 0: Never closes positions same day
        - 1: Always closes all positions same day (characteristic of HFT)
        
        :param lambda_smoothing: Smoothing constant to handle small sample bias
        :param global_median: If a user has no trades, return a meaningful median value
        :return: Intraday closeout rate (float)
        """
        # Group orders by day and stock
        daily_positions = defaultdict(lambda: defaultdict(lambda: {'buys': 0, 'sells': 0}))
        
        for order in self.order_history:
            timestamp = order['time']
            if isinstance(timestamp, str):
                try:
                    timestamp = parser.parse(timestamp)
                except:
                    continue
            
            day = timestamp.date()
            stock = order['stock']
            quantity = order['quant']
            
            if order['action'] == 'buy':
                daily_positions[day][stock]['buys'] += quantity
            elif order['action'] == 'sell':
                daily_positions[day][stock]['sells'] += quantity
    
        # Calculate closeout rate
        days_with_trades = 0
        days_with_full_closeout = 0
    
        for day, stocks in daily_positions.items():
            if not stocks:  # Skip days with no trading
                continue
                
            days_with_trades += 1
            
            # Check if all positions opened this day were closed by end of day
            day_closed = True
            for stock_data in stocks.values():
                if stock_data['buys'] > stock_data['sells']:
                    day_closed = False
                    break
                    
            if day_closed:
                days_with_full_closeout += 1
    
        # Apply Laplace smoothing
        if days_with_trades > 0:
            smoothed_rate = (days_with_full_closeout + lambda_smoothing) / (days_with_trades + 2 * lambda_smoothing)
            return smoothed_rate
    
        # Return global median if available, otherwise default to 0.5 (neutral)
        return global_median if global_median is not None else 0.5


    def calculate_order_clustering_score(self, time_window_seconds=300, lambda_smoothing=1, global_median=None):
        """
        Calculates an order clustering score based on order time gaps and burst intensity.
        
        :param time_window_seconds: Time window to detect bursts (default: 5 minutes)
        :param lambda_smoothing: Smoothing factor for small sample correction
        :param global_median: Median clustering score for missing values
        :return: Clustering score (float between 0 and 1)
        """
        
        # Extract and parse timestamps efficiently
        timestamps = [
            parser.parse(order['time']) if isinstance(order['time'], str) else order['time']
            for order in self.order_history
        ]
        
        if len(timestamps) <= 1:
            return global_median if global_median is not None else 0.5  # Use median or neutral default
        
        # Sort timestamps
        timestamps.sort()
        
        # Convert to NumPy array for fast calculations
        timestamps_sec = np.array([t.timestamp() for t in timestamps])  # Convert to seconds since epoch
    
        # Compute time gaps between consecutive orders
        time_gaps = np.diff(timestamps_sec)
        
        if len(time_gaps) == 0:
            return global_median if global_median is not None else 0.5  # Default to neutral
    
        # Compute clustering score based on time gaps
        clustering_score = np.sum(time_gaps < time_window_seconds) / (len(time_gaps) + lambda_smoothing)
    
        # Efficient burst intensity calculation using sliding window (O(n) instead of O(n²))
        max_burst = 0
        left = 0
    
        for right in range(len(timestamps)):
            while timestamps[right] - timestamps[left] > timedelta(seconds=time_window_seconds):
                left += 1
            max_burst = max(max_burst, right - left + 1)
    
        # Normalize burst intensity (cap at 20 orders per window for stability)
        normalized_burst = min((max_burst + lambda_smoothing) / 20, 1.0)
    
        # Combine clustering score and burst intensity
        combined_score = 0.7 * clustering_score + 0.3 * normalized_burst
    
        return combined_score
    
    def calculate_diversification_index(self, lambda_smoothing=1, global_median=None):
        """
        Calculate a single diversification score (0–1) based on distribution of volumes across assets.
    
        This function computes each asset's share of total volume and sums their squares to get the
        Herfindahl-Hirschman Index (HHI). From there, it calculates the “effective number of assets”
        (1 / HHI) and normalizes it by the actual number of distinct assets, so that:
    
        - 0 means you have all your volume in a single asset (no diversification).
        - 1 means you have an equal share of your volume in every asset (perfect diversification).
    
        Uses Laplace smoothing to avoid extreme values for small traders.
        
        :param lambda_smoothing: Small constant to smooth small sample cases.
        :param global_median: Median diversification score for missing data handling.
        :return: A diversification score in [0, 1].
        """
        if not self.order_history:
            return global_median if global_median is not None else 0.5  # Neutral diversification
    
        # Count unique assets
        unique_assets = set(order['stock'] for order in self.order_history)
        unique_assets_count = len(unique_assets)
    
        if unique_assets_count == 1:
            return lambda_smoothing / (1 + lambda_smoothing)  # Avoid hard zero for one-asset traders
    
        # Calculate volume by asset and total volume
        asset_volumes = {}
        total_volume = 0.0
        for order in self.order_history:
            stock = order['stock']
            quantity = order['quant']
            asset_volumes[stock] = asset_volumes.get(stock, 0.0) + quantity
            total_volume += quantity
    
        if total_volume == 0:
            return global_median if global_median is not None else 0.5  # No trades = neutral
    
        # Compute HHI
        hhi = sum((vol / total_volume) ** 2 for vol in asset_volumes.values())
    
        # Compute effective number of assets
        effective_num_assets = 1.0 / (hhi + lambda_smoothing)  # Smooth denominator
    
        # Normalize by unique assets
        diversification_index = effective_num_assets / unique_assets_count
    
        return min(max(diversification_index, 0.01), 0.99)  # Keep within reasonable bounds


    
    
    
    def calculate_stop_loss_index(self, exchange, lambda_smoothing=1, global_median=None):
        """
        Calculate a single stop-loss index (0–1) for how consistently and effectively stop-loss orders are used.
        
        :param exchange: StockPortfolio object containing historical prices.
        :param lambda_smoothing: Smoothing constant for small sample correction.
        :param global_median: Median stop-loss index for missing data handling.
        :return: Stop-loss index in [0, 1].
        """
    
        buys_by_stock = {}  # Track buys using a dictionary of deques for fast pop/append
        price_changes = []
    
        for order in self.order_history:
            stock = order['stock']
            timestamp = order['time']
    
            if isinstance(timestamp, str):
                try:
                    timestamp = parser.parse(timestamp)
                except:
                    continue  # Skip invalid timestamps
            
            if order['action'] == 'buy':
                quantity = order['quant']
                cost = order['cost']
                price_per_share = cost / quantity if quantity > 0 else 0
    
                if stock not in buys_by_stock:
                    buys_by_stock[stock] = deque()
                
                buys_by_stock[stock].append({
                    'timestamp': timestamp,
                    'quantity': quantity,
                    'price': price_per_share
                })
            
            elif order['action'] == 'sell' and stock in buys_by_stock and buys_by_stock[stock]:
                quantity = order['quant']
                revenue = order['cost']
                sell_price = revenue / quantity if quantity > 0 else 0
    
                # Process all valid buys efficiently
                while quantity > 0 and buys_by_stock[stock]:
                    most_recent_buy = buys_by_stock[stock][0]
    
                    if most_recent_buy['timestamp'] >= timestamp:
                        break  # Ignore invalid buy entries (future purchases)
    
                    buy_price = most_recent_buy['price']
                    if buy_price != 0:
                        price_change_pct = (sell_price - buy_price) / buy_price
                        price_changes.append(price_change_pct)
    
                    if quantity >= most_recent_buy['quantity']:
                        quantity -= most_recent_buy['quantity']
                        buys_by_stock[stock].popleft()  # Remove fully used buy order
                    else:
                        most_recent_buy['quantity'] -= quantity
                        quantity = 0
    
        if not price_changes:
            return global_median if global_median is not None else 0.5
    
        losses = np.array([change for change in price_changes if change < 0])
    
        # Compute stop-loss frequency
        stop_loss_frequency = (len(losses) + lambda_smoothing) / (len(price_changes) + 2 * lambda_smoothing)
    
        # Compute stop-loss threshold
        stop_loss_threshold = abs(np.mean(losses)) if losses.size > 0 else 0.0
    
        # Compute stop-loss consistency
        if losses.size > 1:
            loss_std = np.std(losses)
            loss_mean = abs(np.mean(losses)) or 1.0
            consistency = 1 - min(loss_std / loss_mean, 1.0)
        else:
            consistency = 1.0
    
        # Normalize threshold
        threshold_score = 1 / (1 + stop_loss_threshold + lambda_smoothing)
    
        # Weighted final index
        stop_loss_index = (0.3 * stop_loss_frequency) + (0.3 * threshold_score) + (0.4 * consistency)
    
        return min(max(stop_loss_index, 0.01), 0.99)
    
    
    
    def _get_closed_trades(self):
        """
        Helper: Derives a list of 'closed trades' from order_history using FIFO matching.
        
        Each closed trade dict includes:
        {
            'stock': str,
            'buy_time': datetime,
            'sell_time': datetime,
            'quantity': float,
            'buy_price': float,
            'sell_price': float,
            'pnl': float       # total profit/loss for this lot
        }
    
        Returns a list of these closed trades.
        """
        buys_by_stock = {}
        closed_trades = []
    
        for order in self.order_history:
            timestamp = order['time']
            if isinstance(timestamp, str):
                try:
                    timestamp = parser.parse(timestamp)
                except:
                    continue
            
            stock = order['stock']
            action = order['action']
            quantity = order['quant']
            cost = order['cost']
    
            if action == 'buy':
                price_per_share = cost / quantity if quantity > 0 else 0
    
                if stock not in buys_by_stock:
                    buys_by_stock[stock] = deque()  # Use deque for fast FIFO access
                buys_by_stock[stock].append({
                    'quantity': quantity,
                    'price': price_per_share,
                    'time': timestamp
                })
    
            elif action == 'sell':
                revenue = cost  # "cost" represents revenue for sells
                sell_price_per_share = revenue / quantity if quantity > 0 else 0
                qty_to_close = quantity
    
                # Check if selling more than available
                if stock not in buys_by_stock or not buys_by_stock[stock]:
                    continue
    
                while qty_to_close > 0 and buys_by_stock[stock]:
                    oldest_buy = buys_by_stock[stock][0]
    
                    if oldest_buy['quantity'] <= qty_to_close:
                        matched_qty = oldest_buy['quantity']
                        buy_price = oldest_buy['price']
                        
                        trade_pnl = matched_qty * (sell_price_per_share - buy_price)
                        closed_trades.append({
                            'stock': stock,
                            'buy_time': oldest_buy['time'],
                            'sell_time': timestamp,
                            'quantity': matched_qty,
                            'buy_price': buy_price,
                            'sell_price': sell_price_per_share,
                            'pnl': trade_pnl
                        })
                        
                        qty_to_close -= matched_qty
                        buys_by_stock[stock].popleft()  # Efficient FIFO removal
                    else:
                        matched_qty = qty_to_close
                        buy_price = oldest_buy['price']
                        
                        trade_pnl = matched_qty * (sell_price_per_share - buy_price)
                        closed_trades.append({
                            'stock': stock,
                            'buy_time': oldest_buy['time'],
                            'sell_time': timestamp,
                            'quantity': matched_qty,
                            'buy_price': buy_price,
                            'sell_price': sell_price_per_share,
                            'pnl': trade_pnl
                        })
                        
                        buys_by_stock[stock][0]['quantity'] -= matched_qty
                        qty_to_close = 0
    
        return closed_trades


    def calculate_win_rate(self, lambda_smoothing=1, global_median=None):
        """
        Calculate the proportion of closed trades (buy→sell) that result in a net positive PnL.
        
        Uses Laplace smoothing to prevent extreme values when sample size is small.
        
        :param lambda_smoothing: Small constant to adjust win rate for low sample sizes.
        :param global_median: Global median win rate for missing data handling.
        :return: Win rate in [0,1], where 1 = all trades are profitable.
        """
        closed_trades = self._get_closed_trades()
        if not closed_trades:
            return global_median if global_median is not None else 0.5  # Neutral default
    
        wins = sum(1 for trade in closed_trades if trade['pnl'] > 0)
    
        # Apply Laplace smoothing
        smoothed_win_rate = (wins + lambda_smoothing) / (len(closed_trades) + 2 * lambda_smoothing)
    
        return smoothed_win_rate



    def calculate_profit_factor(self, max_profit_factor=10, global_median=None):
        """
        Calculate Profit Factor = (Sum of all winning trades) / (Sum of all losing trades).
        
        :param max_profit_factor: Cap to replace `inf` for better ML handling.
        :param global_median: Global median profit factor for missing data handling.
        
        :return: Profit Factor (>1 is good). Returns a finite value even when no losing trades exist.
        """
        closed_trades = self._get_closed_trades()
        if not closed_trades:
            return global_median if global_median is not None else 1.0  # Neutral profit factor
    
        total_win = sum(trade['pnl'] for trade in closed_trades if trade['pnl'] > 0)
        total_loss = abs(sum(trade['pnl'] for trade in closed_trades if trade['pnl'] < 0))
    
        if total_loss == 0:
            # If no losing trades, return a high but finite profit factor
            return max_profit_factor if total_win > 0 else 1.0  # Neutral PF
    
        return total_win / total_loss


    

    def calculate_time_weighted_return(self, lambda_smoothing=0.01, global_median=None):
        """
        Calculate a trade-based time-weighted return (TWR).
    
        TWR = Π(1 + r_i)^(1/n) - 1, where n is the number of closed trades.
    
        :param lambda_smoothing: Small constant to stabilize geometric mean calculation.
        :param global_median: Global median TWR for missing data handling.
        
        :return: TWR as a decimal (e.g., 0.10 = +10%). Returns a stable value even in edge cases.
        """
        closed_trades = self._get_closed_trades()
        if not closed_trades:
            return global_median if global_median is not None else 0.0  # Neutral return
    
        trade_returns = []
        for trade in closed_trades:
            if trade['buy_price'] > 0:
                r = (trade['sell_price'] - trade['buy_price']) / trade['buy_price']
                trade_returns.append(r)
    
        if not trade_returns:
            return global_median if global_median is not None else 0.0  # No valid returns
    
        # Apply Laplace smoothing to prevent extreme values
        trade_returns = [r + lambda_smoothing for r in trade_returns]
    
        # Compute geometric mean
        product = math.prod(1.0 + r for r in trade_returns)
        n = len(trade_returns)
        twr = product**(1.0/n) - 1.0
    
        return twr



    import numpy as np

    def calculate_sharpe_ratio(self, risk_free_rate=0.0, lambda_smoothing=0.01, global_median=None):
        """
        Calculate an approximate Sharpe Ratio using trade-based returns.
    
        Sharpe = (mean(r) - risk_free_rate) / std_dev(r),
        where r are the returns on closed trades.
    
        :param risk_free_rate: Risk-free rate for excess return calculation.
        :param lambda_smoothing: Small constant to prevent zero variance issues.
        :param global_median: Global median Sharpe Ratio for missing data handling.
    
        :return: Sharpe Ratio. Returns a stable value even with small trade samples.
        """
        closed_trades = self._get_closed_trades()
        if len(closed_trades) < 2:
            return global_median if global_median is not None else 0.0  # Use global median if available
    
        returns = []
        for trade in closed_trades:
            if trade['buy_price'] > 0:
                r = (trade['sell_price'] - trade['buy_price']) / trade['buy_price']
                returns.append(r)
    
        if len(returns) < 2:
            return global_median if global_median is not None else 0.0  # Still not enough data
    
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1) + lambda_smoothing  # Apply smoothing
    
        sharpe = (avg_return - risk_free_rate) / std_return
        return sharpe

    
    def calculate_volatility_of_return(self):
        """
        Calculate the standard deviation of trade returns (treating each trade as a 'period').
        
        Returns:
            float: Volatility (std dev of returns). 0 if no data or too few trades.
        """


        closed_trades = self._get_closed_trades()
        if len(closed_trades) < 2:
            return 0.0

        returns = []
        for trade in closed_trades:
            if trade['buy_price'] != 0:
                r = (trade['sell_price'] - trade['buy_price']) / trade['buy_price']
                returns.append(r)
        
        if len(returns) < 2:
            return 0.0

        return np.std(returns, ddof=1)
    
    
    
    
    def calculate_largest_exposure(self, global_median=None):
        """
        Optimized version of max total cost basis at any order event.
        
        :param global_median: Global median exposure for missing data handling.
        :return: Largest exposure (float), the max total capital at risk.
        """
        open_positions = {}  # stock -> deque of {'quantity': q, 'price': p}
        max_exposure = 0.0
        current_exposure = 0.0  
    
        # Sort orders by time (convert if needed)
        sorted_orders = sorted(
            self.order_history,
            key=lambda o: o['time'] if not isinstance(o['time'], str) else parser.parse(o['time'])
        )
        
        if not sorted_orders:
            return global_median if global_median is not None else 0.0  # Handle missing data
    
        for order in sorted_orders:
            stock = order['stock']
            action = order['action']
            quantity = order['quant']
            cost = order['cost']
    
            if action == 'buy':
                price_per_share = cost / quantity if quantity > 0 else 0
                if stock not in open_positions:
                    open_positions[stock] = deque()
                open_positions[stock].append({'quantity': quantity, 'price': price_per_share})
                current_exposure += cost  
    
            elif action == 'sell':
                if stock not in open_positions or not open_positions[stock]:
                    #print(f"Warning: Selling more {stock} than owned!")  # Optional logging
                    continue  
    
                qty_to_sell = quantity
                while qty_to_sell > 0 and open_positions[stock]:
                    lot = open_positions[stock][0]
                    lot_qty = lot['quantity']
    
                    if lot_qty <= qty_to_sell:
                        qty_to_sell -= lot_qty
                        current_exposure -= lot_qty * lot['price']
                        open_positions[stock].popleft()
                    else:
                        lot['quantity'] -= qty_to_sell
                        current_exposure -= qty_to_sell * lot['price']
                        qty_to_sell = 0
    
            max_exposure = max(max_exposure, current_exposure)
    
        return max_exposure

    
    def calculate_mean_reversion_score(self, exchange, lookback_period=5, lambda_smoothing=1, global_median=None):
        """
        Calculate a mean reversion score based on a trader's order behavior relative to stock trends.
    
        - Positive Score: Indicates a mean-reverting trader (buying in downtrends, selling in uptrends).
        - Negative Score: Indicates a momentum trader (buying in uptrends, selling in downtrends).
    
        :param exchange: StockPortfolio object containing historical prices.
        :param lookback_period: Number of historical price points to consider for trend calculation.
        :param lambda_smoothing: Smoothing factor for small sample correction.
        :param global_median: If a user has no trades, return a meaningful median value.
    
        :return: Mean reversion score (normalized, with smoothing).
        """
        score = 0
        trade_count = 0
    
        for order in self.order_history:
            stock = order["stock"]
            action = order["action"]
            timestamp = order["time"]
    
            if stock not in exchange.stocks:
                continue
    
            # Get historical prices
            stock_prices = exchange.stocks[stock].get_prices()
            prior_prices = [(t, p) for t, p in stock_prices if t < timestamp]
            
            if len(prior_prices) < lookback_period:
                continue
    
            recent_prices = prior_prices[-lookback_period:]
            first_price = recent_prices[0][1]
            last_price = recent_prices[-1][1]
    
            # Determine trend direction
            is_uptrend = last_price > first_price
    
            if action == "buy":
                score += 1 if not is_uptrend else -1
                trade_count += 1
            elif action == "sell":
                score += 1 if is_uptrend else -1
                trade_count += 1
    
        # Apply Laplace smoothing to prevent extreme values
        if trade_count > 0:
            smoothed_score = (score + lambda_smoothing) / (trade_count + 2 * lambda_smoothing)
            return smoothed_score
    
        # Return global median if available, otherwise default to 0.0
        return global_median if global_median is not None else 0.0

    

    
    
    def calculate_position_sizing_score(self, lambda_smoothing=0.01, global_median=None):
        """
        Estimate how aggressively the user sizes positions.
        Metric: Average fraction of balance used for each BUY order.
    
        :param lambda_smoothing: Small constant to prevent extreme values for small samples.
        :param global_median: Median position sizing score for missing data handling.
        
        :return: A higher number => bigger trades relative to (balance + portfolio cost basis).
        """
        buy_orders = [o for o in self.order_history if o['action'] == 'buy']
        if not buy_orders:
            return global_median if global_median is not None else 0.0  # Handle missing data
    
        # Sort orders by time
        sorted_orders = sorted(self.order_history, key=lambda o: o['time'])
    
        sizing_ratios = []
        running_balance = self.balance
        cost_basis = 0.0  
    
        for order in sorted_orders:
            action = order['action']
            quantity = order['quant']
            cost = order['cost']
            timestamp = order['time']
    
            if isinstance(timestamp, str):
                try:
                    timestamp = parser.parse(timestamp)
                except:
                    continue
    
            if action == 'buy':
                denom = running_balance + cost_basis + lambda_smoothing  # Prevent division errors
                fraction = cost / denom
                sizing_ratios.append(fraction)
    
                # Update balance and cost basis
                running_balance -= cost
                cost_basis += cost
    
            elif action == 'sell':
                revenue = cost
                running_balance += revenue
                cost_basis = max(0, cost_basis - revenue)  # Avoid negative cost basis
    
        # Apply Laplace smoothing to stabilize values
        if sizing_ratios:
            return float(np.mean(sizing_ratios) + lambda_smoothing)
    
        return global_median if global_median is not None else 0.0

    
    def calculate_trade_sequencing_score(self, lambda_smoothing=1, global_median=None):
        """
        Gauge how often the user 'doubles down' (increases position size) after a losing trade.
    
        We define a trade as a closed buy→sell pair and check whether the next trade has a
        larger position size if the previous trade was a loss.
    
        :param lambda_smoothing: Small constant for stability with small samples.
        :param global_median: If a user has no losing trades, return a meaningful median value.
    
        :return: A fraction in [0,1] representing the probability of increasing size after a loss.
        """
        closed_trades = self._get_closed_trades()
        if len(closed_trades) < 2:
            return global_median if global_median is not None else 0.5  # Neutral score
    
        # Sort closed trades by sell_time
        closed_trades.sort(key=lambda x: x['sell_time'])
    
        double_down_count = 0
        losing_trades = 0
    
        for i in range(len(closed_trades) - 1):
            this_trade = closed_trades[i]
            next_trade = closed_trades[i + 1]
    
            if this_trade['pnl'] < 0:
                losing_trades += 1
                if next_trade['quantity'] > this_trade['quantity']:
                    double_down_count += 1
    
        # Apply Laplace smoothing to stabilize estimates
        smoothed_score = (double_down_count + lambda_smoothing) / (losing_trades + 2 * lambda_smoothing)
    
        return smoothed_score
    

    import numpy as np
    
    def calculate_trading_confidence(self, lambda_smoothing=0.01, global_median=None):
        """
        Estimate 'trading confidence' by checking if larger trades (fraction of capital used)
        result in higher PnL.
    
        We approximate:
        1. Fraction of capital used = (buy_price * quantity) / (balance at that time + portfolio value)
        2. Compute the correlation between fraction_of_capital_used and the actual PnL of that trade.
    
        :param lambda_smoothing: Small constant to stabilize variance estimates.
        :param global_median: Median trading confidence for missing data handling.
        
        :return: Correlation in [-1, 1], where 1 => big trades are more profitable,
                 -1 => big trades lose more often, 0 => no relationship.
        """
        closed_trades = self._get_closed_trades()
        if len(closed_trades) < 2:
            return global_median if global_median is not None else 0.0  # Use median if available
    
        fractions = []
        pnls = []
        baseline_balance = max(1.0, self.balance)  # Prevent division by zero
    
        for trade in closed_trades:
            buy_cost = trade['buy_price'] * trade['quantity']
            frac = buy_cost / (baseline_balance + lambda_smoothing)  # Apply small smoothing
            fractions.append(frac)
            pnls.append(trade['pnl'])
    
        if len(fractions) < 2 or np.std(fractions) == 0 or np.std(pnls) == 0:
            return global_median if global_median is not None else 0.0  # Prevent NaN correlation
    
        correlation = np.corrcoef(fractions, pnls)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    def calculate_trade_expectancy(self):
        """
        Calculate expectancy: The average profit per trade, adjusted for win rate.
        
        :return: Expectancy (float) – higher means better risk-adjusted performance.
        """
        closed_trades = self._get_closed_trades()
        if not closed_trades:
            return 0.0
    
        wins = [trade['pnl'] for trade in closed_trades if trade['pnl'] > 0]
        losses = [abs(trade['pnl']) for trade in closed_trades if trade['pnl'] < 0]
    
        win_rate = len(wins) / len(closed_trades) if closed_trades else 0
        loss_rate = len(losses) / len(closed_trades) if closed_trades else 0
    
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
    
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        return expectancy

    def calculate_risk_of_ruin(self, initial_capital=None):
        """
        Estimate risk of ruin probability using win rate and average loss.
        
        :param initial_capital: Starting capital for the trader (if available).
        :return: Risk of ruin probability (0-1), where 1 means likely bankruptcy.
        """
        closed_trades = self._get_closed_trades()
        if not closed_trades:
            return 1.0  # No trades → high risk
    
        wins = len([trade['pnl'] for trade in closed_trades if trade['pnl'] > 0])
        total_trades = len(closed_trades)
        win_rate = wins / total_trades if total_trades else 0
    
        losses = [abs(trade['pnl']) for trade in closed_trades if trade['pnl'] < 0]
        avg_loss = np.mean(losses) if losses else 0.01  # Avoid zero loss
    
        capital = initial_capital if initial_capital else self.balance  # Use balance if no input
        if capital <= 0:
            return 1.0  # Already bankrupt
    
        risk_of_ruin = ((1 - win_rate) / (1 + win_rate)) ** (capital / avg_loss)
        return min(max(risk_of_ruin, 0), 1)  # Keep in [0,1] range

    def calculate_kelly_criterion(self):
        """
        Estimate the optimal fraction of capital to risk per trade using the Kelly Criterion.
        
        :return: Kelly fraction (0-1) for optimal bet sizing.
        """
        closed_trades = self._get_closed_trades()
        if not closed_trades:
            return 0.0
    
        wins = [trade['pnl'] for trade in closed_trades if trade['pnl'] > 0]
        losses = [abs(trade['pnl']) for trade in closed_trades if trade['pnl'] < 0]
    
        win_rate = len(wins) / len(closed_trades) if closed_trades else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0.01  # Avoid division by zero
    
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    
        kelly_fraction = win_rate - ((1 - win_rate) / risk_reward_ratio) if risk_reward_ratio > 0 else 0
        return max(kelly_fraction, 0)  # Ensure non-negative bet sizing

    def calculate_trade_duration(self):
        """
        Calculate the average holding time of trades in days.
    
        :return: Average trade duration in days.
        """
        from datetime import timedelta
    
        closed_trades = self._get_closed_trades()
        if not closed_trades:
            return 0.0
    
        durations = [(trade['sell_time'] - trade['buy_time']).total_seconds() / (60 * 60 * 24)
                     for trade in closed_trades]
        
        return np.mean(durations) if durations else 0.0

    def calculate_max_drawdown(self):
        """
        Calculate the maximum drawdown (worst peak-to-trough loss) in trading history.
    
        :return: Maximum drawdown as a fraction of peak balance.
        """
        closed_trades = self._get_closed_trades()
        if not closed_trades:
            return 0.0
    
        balance = self.balance
        pnl_cumsum = np.cumsum([trade['pnl'] for trade in closed_trades])
    
        peak = np.maximum.accumulate(pnl_cumsum)
        drawdowns = (peak - pnl_cumsum) / (peak + 1e-6)  # Avoid division by zero
    
        return np.max(drawdowns) if len(drawdowns) > 0 else 0.0


    def infer_starting_balance(self, exchange):
        """
        Estimates the starting balance based on order history.
        Includes assets sold but never bought in this dataset as part of the initial balance.

        :param exchange: StockPortfolio object to fetch initial asset prices.
        :return: Estimated starting balance.
        """
        # Sort transactions by time
        sorted_orders = sorted(self.order_history, key=lambda x: x["time"])
        
        cash_balance = 0
        min_balance = 0
        stock_holdings = defaultdict(int)
        stocks_sold_but_never_bought = defaultdict(int)

        # Step 1: Track cash balance and stock holdings
        for order in sorted_orders:
            stock = order["stock"]
            action = order["action"]
            quantity = order["quant"]
            cost = order["cost"]

            if action == "buy":
                cash_balance -= cost  # Deduct cash spent
                stock_holdings[stock] += quantity  # Increase holdings
            elif action == "sell":
                cash_balance += cost  # Add revenue from sale
                stock_holdings[stock] -= quantity  # Reduce holdings
                
                # If we sell a stock that was never bought in the dataset, track it
                if stock_holdings[stock] < 0:
                    stocks_sold_but_never_bought[stock] += abs(stock_holdings[stock])
                    stock_holdings[stock] = 0  # Reset to avoid double counting
            
            min_balance = min(min_balance, cash_balance)  # Track worst deficit

        # Step 2: Estimate the initial balance required to cover the worst-case cash deficit
        estimated_starting_balance = abs(min_balance)

        # Step 3: Add value of initial stock holdings
        for stock, quantity in stock_holdings.items():
            if quantity > 0:
                first_price = exchange.get_stock(stock).get_prices()[0][1] if stock in exchange.stocks else 0
                estimated_starting_balance += quantity * first_price  # Add value of initial holdings

        # Step 4: Add value of stocks that were sold but never bought in the dataset
        for stock, quantity in stocks_sold_but_never_bought.items():
            first_price = exchange.get_stock(stock).get_prices()[0][1] if stock in exchange.stocks else 0
            estimated_starting_balance += quantity * first_price  # Assume they held it from before

        self.starting_balance = estimated_starting_balance  # Store inferred balance
        return estimated_starting_balance


    def calculate_news_reaction_metrics(self, news_events, reaction_window_minutes=30):
        """
        Calculate metrics related to how quickly and effectively the person reacts to news events.
        
        :param news_events: List of tuples (symbol, event_time, price_before, price_after)
        :param reaction_window_minutes: Maximum time window to consider for reactions
        :return: Dictionary of reaction metrics
        """
        reactions = []
        
        for event in news_events:
            symbol, event_time, price_before, price_after = event
            
            # Ensure event_time is a datetime object
            if isinstance(event_time, str):
                event_time = parser.parse(event_time)
            
            # Price direction (1 for up, -1 for down)
            price_direction = 1 if price_after > price_before else -1
            price_change_pct = abs((price_after - price_before) / price_before * 100)
            
            # Look for trades in this stock after the event
            for order in self.order_history:
                order_time = order['time']
                
                # Ensure order_time is a datetime object
                if isinstance(order_time, str):
                    order_time = parser.parse(order_time)
                
                # Normalize timezone information
                if order_time.tzinfo and not event_time.tzinfo:
                    event_time = event_time.replace(tzinfo=order_time.tzinfo)
                elif event_time.tzinfo and not order_time.tzinfo:
                    order_time = order_time.replace(tzinfo=event_time.tzinfo)
                
                if order['stock'] == symbol and order_time > event_time:
                    # Calculate reaction time in minutes
                    reaction_time = (order_time - event_time).total_seconds() / 60
                    
                    # Only consider reactions within the window
                    if reaction_time <= reaction_window_minutes:
                        # Check if trade direction aligns with price movement
                        trade_direction = 1 if order['action'] == 'buy' else -1
                        aligned_with_movement = (trade_direction == price_direction)
                        
                        reactions.append({
                            'event_time': event_time,
                            'reaction_time': reaction_time,
                            'symbol': symbol,
                            'price_change_pct': price_change_pct,
                            'aligned_with_movement': aligned_with_movement,
                            'trade_size': order['quant']
                        })
                        break  # Only consider first reaction to each event
        
        # Calculate metrics
        if not reactions:
            print("not enough reactions")
            return {
                'reaction_count': 0,
                'avg_reaction_time': None,
                'median_reaction_time': None,
                'alignment_rate': None,
                'reaction_speed_score': None,
                'weighted_reaction_score': None
            }
        
        reaction_times = [r['reaction_time'] for r in reactions]
        alignment_rate = sum(1 for r in reactions if r['aligned_with_movement']) / len(reactions)
        
        # Calculate weighted metrics (larger price moves and larger trades get more weight)
        weighted_reaction_times = []
        weights = []
        
        for r in reactions:
            # Weight by price change percentage and trade size
            weight = r['price_change_pct'] * r['trade_size']
            weighted_reaction_times.append(r['reaction_time'] * weight)
            weights.append(weight)
        
        weighted_avg_reaction_time = sum(weighted_reaction_times) / sum(weights) if sum(weights) > 0 else None
        
        # Calculate a composite score (lower is better for reaction time, higher is better for alignment)
        # Normalize reaction time (1 = instant reaction, 0 = slowest possible within window)
        normalized_reaction_time = 1 - (sum(reaction_times) / len(reaction_times) / reaction_window_minutes)
        reaction_speed_score = normalized_reaction_time * alignment_rate

        print(f"Reaction Speed Score: {reaction_speed_score}, Alignment Rate: {alignment_rate}")
        print(f"Weighted Avg Reaction Time: {weighted_avg_reaction_time}, Avg Reaction Time: {sum(reaction_times) / len(reaction_times)}")
        print(f"Median Reaction Time: {sorted(reaction_times)[len(reaction_times) // 2]}")
        print(f"Reaction Count: {len(reactions)}")
        print(f"Reactions: {reactions}")
        # Return metrics
        
        return {
            'reaction_count': len(reactions),
            'avg_reaction_time': sum(reaction_times) / len(reaction_times),
            'median_reaction_time': sorted(reaction_times)[len(reaction_times) // 2],
            'alignment_rate': alignment_rate,
            'reaction_speed_score': reaction_speed_score,
            'weighted_reaction_score': weighted_avg_reaction_time
        }
    


    def calculate_log_return_from_orders(self, exchange):
        """
        Calculates log return from order history using valid buys and sells.
        Ignores sells that exceed owned quantities.

        :param exchange: Exchange object with market prices.
        :return: log return (float)
        """
        from collections import defaultdict
        import numpy as np

        invested_cash = 0.0
        sell_proceeds = 0.0
        holdings = defaultdict(float)

        for order in sorted(self.order_history, key=lambda x: x['time']):
            action = order['action']
            stock = order['stock']
            quantity = order['quant']
            cost = order['cost']

            if action == 'buy':
                invested_cash += cost
                holdings[stock] += quantity
            elif action == 'sell':
                if holdings[stock] >= quantity:
                    holdings[stock] -= quantity
                    sell_proceeds += cost
                else:
                    # Skip invalid sell
                    continue

        # Add value of remaining holdings to final wealth
        portfolio_value = 0.0
        for stock, qty in holdings.items():
            if qty <= 0:
                continue
            try:
                latest_price = exchange.get_stock(stock).get_latest_price()[1]
                if not isinstance(latest_price, (float, int)) or np.isnan(latest_price):
                    latest_price = 0.0
                portfolio_value += qty * latest_price
            except:
                continue

        final_wealth = sell_proceeds + portfolio_value

        if invested_cash <= 0 or final_wealth <= 0:
            print(f"Invalid cash or invested cash values. Returning 0.0")
            print(f"invested: {invested_cash}, final_wealth: {final_wealth}")
            return 0.0

        return float(np.log(final_wealth / invested_cash))


    
    def calculate_news_alpha(self, news_events, reaction_window_minutes=30, market_avg_reaction_time=None):
        """
        Calculate a trader's "news alpha" - how much better they react to news compared to the market.
        
        :param news_events: List of news events
        :param reaction_window_minutes: Maximum time window to consider
        :param market_avg_reaction_time: Average reaction time across all traders
        :return: News alpha score (higher is better)
        """
        metrics = self.calculate_news_reaction_metrics(news_events, reaction_window_minutes)
        
        if metrics['reaction_count'] < 5 or not market_avg_reaction_time:
            return 0.0  # Not enough data
        
        # How much faster than average (as a percentage)
        speed_advantage = (market_avg_reaction_time - metrics['avg_reaction_time']) / market_avg_reaction_time
        
        # Combine speed advantage with accuracy
        news_alpha = speed_advantage * metrics['alignment_rate']
        
        return news_alpha

    def calculate_market_reaction_stats(people, news_events):
        """
        Calculate market-wide statistics for news reaction times.
        
        :param people: List of Person objects
        :param news_events: List of news events
        :return: Dictionary of market-wide statistics
        """
        all_reaction_times = []
        
        for person in people:
            metrics = person.calculate_news_reaction_metrics(news_events)
            if metrics['reaction_count'] > 0 and metrics['avg_reaction_time'] is not None:
                all_reaction_times.append(metrics['avg_reaction_time'])
        
        if not all_reaction_times:
            return {
                'market_avg_reaction_time': None,
                'market_median_reaction_time': None
            }
        
        return {
            'market_avg_reaction_time': sum(all_reaction_times) / len(all_reaction_times),
            'market_median_reaction_time': sorted(all_reaction_times)[len(all_reaction_times) // 2]
        }

    def generate_news_reaction_table(people, news_events):
        """
        Generate a table of news reaction metrics for all traders.
        
        :param people: List of Person objects
        :param news_events: List of news events
        :return: List of dictionaries with trader metrics
        """
        market_stats = calculate_market_reaction_stats(people, news_events)
        market_avg_reaction_time = market_stats['market_avg_reaction_time']
        
        table_data = []
        
        for person in people:
            metrics = person.calculate_news_reaction_metrics(news_events)
            news_alpha = person.calculate_news_alpha(news_events, market_avg_reaction_time=market_avg_reaction_time)
            
            row = {
                'trader_id': person.id,
                'reaction_count': metrics['reaction_count'],
                'avg_reaction_time': metrics['avg_reaction_time'],
                'median_reaction_time': metrics['median_reaction_time'],
                'alignment_rate': metrics['alignment_rate'],
                'reaction_speed_score': metrics['reaction_speed_score'],
                'news_alpha': news_alpha
            }
            
            table_data.append(row)
        
        # Sort by news alpha (higher is better)
        return sorted(table_data, key=lambda x: x['news_alpha'] if x['news_alpha'] is not None else -float('inf'), reverse=True)


    def calculate_return_consistency(self, lambda_smoothing=0.01, global_median=None):
        """
        Calculate the consistency of returns using lag-1 autocorrelation.
        
        A high positive value suggests skill or repeatability.
        Uses closed trades for return series.
        
        :param lambda_smoothing: To prevent instability with low sample sizes.
        :param global_median: If insufficient data, use this as a fallback.
        :return: Lag-1 autocorrelation of trade returns, smoothed.
        """
        closed_trades = self._get_closed_trades()
        
        # Extract returns
        returns = [
            (t['sell_price'] - t['buy_price']) / t['buy_price']
            for t in closed_trades
            if t['buy_price'] > 0
        ]
        
        if len(returns) < 3:
            return global_median if global_median is not None else 0.0

        # Convert to numpy for calculation
        returns = np.array(returns)
        mean_return = np.mean(returns)

        # Compute lag-1 autocorrelation: cov(r_t, r_{t-1}) / var(r)
        numerator = np.sum((returns[1:] - mean_return) * (returns[:-1] - mean_return))
        denominator = np.sum((returns - mean_return) ** 2) + lambda_smoothing  # Prevent zero-variance

        autocorr = numerator / denominator
        return float(autocorr)

    def calculate_return_skewness(self, global_median=None):
        """
        Calculates skewness of returns from closed trades.
        
        Positive skew → occasional big wins, mostly small gains/losses.
        Negative skew → occasional big losses.
        """
        closed_trades = self._get_closed_trades()
        returns = [
            (t['sell_price'] - t['buy_price']) / t['buy_price']
            for t in closed_trades
            if t['buy_price'] > 0
        ]

        if len(returns) < 3:
            return global_median if global_median is not None else 0.0
        
        return float(scipy.stats.skew(returns))
    
    def calculate_return_kurtosis(self, global_median=None):
        """
        Calculates kurtosis of returns from closed trades.
        
        High kurtosis → more extreme outliers (either wins or losses).
        Low kurtosis → more stable, consistent return profile.
        """
        closed_trades = self._get_closed_trades()
        returns = [
            (t['sell_price'] - t['buy_price']) / t['buy_price']
            for t in closed_trades
            if t['buy_price'] > 0
        ]

        if len(returns) < 3:
            return global_median if global_median is not None else 0.0
        
        return float(scipy.stats.kurtosis(returns, fisher=True))  # Fisher=True → normal dist has kurtosis 0
    
    def calculate_information_gain_from_price(self, exchange, threshold=0.8, global_median=None):
        """
        Approximates information gain by inferring market outcome from final price.

        If price > threshold → market likely resolved TRUE
        If price < 1 - threshold → market likely resolved FALSE
        Prices in between are ignored.

        :param exchange: StockPortfolio object (to access final prices)
        :param threshold: Confidence threshold (e.g. 0.8 = 80%)
        :param global_median: Fallback value if not enough scorable trades
        :return: Estimated information gain (0–1)
        """
        correct = 0
        total = 0

        for order in self.order_history:
            stock = order['stock']
            action = order['action']

            stock_obj = exchange.get_stock(stock)
            if not stock_obj:
                continue

            latest_price = stock_obj.get_latest_price()
            if not latest_price or not isinstance(latest_price[1], (float, int)):
                continue

            price = latest_price[1]

            # Infer outcome from price
            if price >= threshold:
                inferred_outcome = True
            elif price <= 1 - threshold:
                inferred_outcome = False
            else:
                continue  # Price too close to 0.5 — skip ambiguous cases

            if action == 'buy' and inferred_outcome:
                correct += 1
            elif action == 'sell' and not inferred_outcome:
                correct += 1

            total += 1

        if total == 0:
            return global_median if global_median is not None else 0.5

        return correct / total


    def calculate_return_mad(self, global_median=None):
        """
        Calculates the Median Absolute Deviation (MAD) of trade returns.
        
        This is a robust alternative to standard deviation, less sensitive to outliers.
        
        :param global_median: Value to return if insufficient data
        :return: MAD of returns (float)
        """
        closed_trades = self._get_closed_trades()
        
        returns = [
            (t['sell_price'] - t['buy_price']) / t['buy_price']
            for t in closed_trades
            if t['buy_price'] > 0
        ]
        
        if len(returns) < 2:
            return global_median if global_median is not None else 0.0

        median = np.median(returns)
        abs_devs = [abs(r - median) for r in returns]
        mad = np.median(abs_devs)

        return float(mad)


    def _calculate_gini(self, values):
        """
        Computes the Gini coefficient for a list of values.
        Returns 0 if all values are equal, 1 if all inequality.
        """
        if len(values) == 0:
            return 0.0
        
        values = np.array(values)
        if np.all(values == 0):
            return 0.0

        values = np.sort(values)
        n = len(values)
        cumulative = np.cumsum(values)
        gini = (2 * np.sum((np.arange(1, n+1) * values))) / (n * np.sum(values)) - (n + 1) / n

        return float(gini)
    
    def calculate_gini_bet_size(self):
        """
        Measures inequality of trade sizes.
        A value near 1 = one or two huge trades dominate activity.
        Near 0 = even distribution of trade sizes.
        """
        sizes = [order['quant'] for order in self.order_history if order['quant'] > 0]
        return self._calculate_gini(sizes)
    
    def calculate_gini_return_distribution(self):
        """
        Measures inequality of per-trade profit/loss.
        Value near 1 = a few trades contribute most of the PnL.
        """
        closed_trades = self._get_closed_trades()
        profits = [abs(t['pnl']) for t in closed_trades if t['pnl'] != 0]
        return self._calculate_gini(profits)

    def calculate_outlier_ratio(self, k=3.0, global_median=None):
        """
        Calculates the fraction of trades that are outliers based on return IQR.
        
        A trade is an outlier if its return is more than k×IQR below Q1 or above Q3.
        
        :param k: Outlier threshold multiplier (default: 3.0)
        :param global_median: Fallback if too few trades
        :return: Fraction of returns that are outliers
        """
        closed_trades = self._get_closed_trades()

        returns = [
            (t['sell_price'] - t['buy_price']) / t['buy_price']
            for t in closed_trades
            if t['buy_price'] > 0
        ]

        if len(returns) < 4:
            return global_median if global_median is not None else 0.0

        q1 = np.percentile(returns, 25)
        q3 = np.percentile(returns, 75)
        iqr = q3 - q1

        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr

        outliers = [r for r in returns if r < lower_bound or r > upper_bound]
        ratio = len(outliers) / len(returns)

        return float(ratio)

    def calculate_relative_exit_efficiency(self, exchange, lookahead_minutes=60, global_median=None):
        """
        Estimates how well the trader captures price swings after entry.

        For each closed trade:
        - Compare exit price to the best price reached within N minutes after entry
        - Score is normalized: 1 = perfect exit, 0 = worst case

        :param exchange: StockPortfolio with historical prices
        :param lookahead_minutes: Time window to evaluate price swing
        :param global_median: Fallback value
        :return: Average relative efficiency across trades
        """
        from bisect import bisect_right
        from datetime import timedelta

        closed_trades = self._get_closed_trades()
        efficiencies = []

        for trade in closed_trades:
            stock = trade['stock']
            buy_time = trade['buy_time']
            sell_time = trade['sell_time']
            buy_price = trade['buy_price']
            sell_price = trade['sell_price']

            if stock not in exchange.stocks:
                continue

            prices = exchange.stocks[stock].get_prices()
            if not prices:
                continue

            # Extract prices in the lookahead window after entry
            lookahead_end = buy_time + timedelta(minutes=lookahead_minutes)
            from dateutil import parser  # add at top if not already

            future_prices = [
                p for t, p in prices
                if buy_time < parser.parse(t) <= lookahead_end
            ]



            if not future_prices:
                continue

            if sell_price >= buy_price:
                # For long trades: compare to max price
                best_price = max(future_prices)
                if best_price != buy_price:
                    efficiency = (sell_price - buy_price) / (best_price - buy_price)
                    efficiency = min(max(efficiency, 0), 1)
                    efficiencies.append(efficiency)
            else:
                # For short trades: compare to min price
                best_price = min(future_prices)
                if best_price != buy_price:
                    efficiency = (buy_price - sell_price) / (buy_price - best_price)
                    efficiency = min(max(efficiency, 0), 1)
                    efficiencies.append(efficiency)

        if not efficiencies:
            return global_median if global_median is not None else 0.5  # Neutral default

        return float(np.mean(efficiencies))
    
    def calculate_panic_sell_score(self, exchange, short_holding_days=1.0, price_drop_threshold=0.01, global_median=None):
        """
        Approximates panic selling behavior.
        
        A panic sell is defined as:
        - Selling at a price lower than entry
        - Within a short holding time (e.g., 1 day)
        - Optional: price didn’t rebound after sell (not implemented here)
        
        :param exchange: StockPortfolio object for price reference
        :param short_holding_days: Max duration (in days) considered "short"
        :param price_drop_threshold: Min % drop to count as a panic-trigger (e.g. 0.01 = 1%)
        :param global_median: Fallback return value if no trades
        :return: Fraction of closed trades flagged as panic sells
        """
        closed_trades = self._get_closed_trades()
        if not closed_trades:
            return global_median if global_median is not None else 0.0

        panic_count = 0
        total_closed = 0

        for trade in closed_trades:
            holding_days = (trade['sell_time'] - trade['buy_time']).total_seconds() / (60 * 60 * 24)
            pnl_pct = (trade['sell_price'] - trade['buy_price']) / trade['buy_price']

            if holding_days <= short_holding_days and pnl_pct < -price_drop_threshold:
                panic_count += 1

            total_closed += 1

        return panic_count / total_closed if total_closed > 0 else 0.0
    
    def calculate_average_portfolio_risked(self, exchange, lambda_smoothing=1e-6, global_median=None):
        """
        Calculates average proportion of portfolio risked per BUY trade.
        
        Formula:
        amount_spent / (portfolio_value + balance at that time)
        
        :param exchange: StockPortfolio object to get valuations
        :param lambda_smoothing: To avoid division by zero
        :param global_median: Fallback return value
        :return: Average portfolio risk ratio (0–1+), higher = riskier
        """
        if not self.order_history:
            return global_median if global_median is not None else 0.0

        running_balance = self.balance
        portfolio_snapshot = {}  # stock -> (shares, total cost)
        ratios = []

        for order in sorted(self.order_history, key=lambda o: o['time']):
            stock = order['stock']
            action = order['action']
            quant = order['quant']
            cost = order['cost']

            if action.lower() == 'buy':
                # Estimate portfolio value BEFORE the trade
                portfolio_value = 0.0
                for s, (shares, total_cost) in portfolio_snapshot.items():
                    stock_obj = exchange.get_stock(s)
                    price = stock_obj.get_latest_price()[1] if stock_obj and stock_obj.get_latest_price() else 0.0
                    portfolio_value += shares * price

                denom = portfolio_value + running_balance + lambda_smoothing
                risk_ratio = cost / denom
                ratios.append(risk_ratio)

                # Update state after trade
                running_balance -= cost
                if stock not in portfolio_snapshot:
                    portfolio_snapshot[stock] = [0.0, 0.0]
                portfolio_snapshot[stock][0] += quant
                portfolio_snapshot[stock][1] += cost

            elif action.lower() == 'sell':
                running_balance += cost
                if stock in portfolio_snapshot:
                    avg_price = portfolio_snapshot[stock][1] / max(portfolio_snapshot[stock][0], lambda_smoothing)
                    portfolio_snapshot[stock][0] -= quant
                    portfolio_snapshot[stock][1] -= quant * avg_price
                    if portfolio_snapshot[stock][0] <= 0:
                        del portfolio_snapshot[stock]

        if not ratios:
            return global_median if global_median is not None else 0.0

        return float(np.mean(ratios))




