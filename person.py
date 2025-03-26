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
    
    def get_total_return(self, exchange):


        if self.total_paid == 0:
            return 0
        
        total_income = self.total_made + self.get_portfolio_value(exchange)
        total_expense = self.total_paid
        return (total_income - total_expense) / total_expense
    
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
    
    from dateutil import parser
    from collections import deque
    
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



