
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


    
    def measure_execution_time(self, label, func, *args, **kwargs):
        print(label)
        start_time = time.time()
        for person in itertools.islice(self.people.values(), 100):
            func(person, *args, **kwargs)
        end_time = time.time()
        print(f"{label} done in {(end_time - start_time)/100} seconds")
    

    def make_df(self, exchange):
        """
        Creates a DataFrame from the people data and includes additional trading metrics.

        Returns a DataFrame with columns:
            ['id', 'return', 'momentum_score', 'average_holding_time', 'risk_tolerance', 
            'intraday_closeout_rate', 'order_clustering_score', 'asset_diversification_index', 
            'stop_loss_index', 'win_rate', 'profit_factor', 'time_weighted_return', 
            'sharpe_ratio', 'volatility_of_return', 'largest_exposure', 
            'mean_reversion_score', 'position_sizing_score', 'trade_sequencing_score',
            'trading_confidence']
        """
        import itertools
        data = []
        i = 0

        #self.measure_execution_time("calculate_momentum_score", lambda p: p.calculate_momentum_score(exchange))
        #self.measure_execution_time("calculate_average_holding_time", lambda p: p.calculate_average_holding_time())
        #self.measure_execution_time("calculate_risk_tolerance", lambda p: p.calculate_risk_tolerance())
        #self.measure_execution_time("calculate_intraday_closeout_rate", lambda p: p.calculate_intraday_closeout_rate())
        #self.measure_execution_time("calculate_order_clustering_score", lambda p: p.calculate_order_clustering_score())
        #self.measure_execution_time("calculate_diversification_index", lambda p: p.calculate_diversification_index())
        self.measure_execution_time("calculate_stop_loss_index", lambda p: p.calculate_stop_loss_index(exchange))
        #self.measure_execution_time("calculate_win_rate", lambda p: p.calculate_win_rate())
        #self.measure_execution_time("calculate_profit_factor", lambda p: p.calculate_profit_factor())
        ##self.measure_execution_time("calculate_time_weighted_return", lambda p: p.calculate_time_weighted_return())
        #self.measure_execution_time("calculate_sharpe_ratio", lambda p: p.calculate_sharpe_ratio())
        #self.measure_execution_time("calculate_volatility_of_return", lambda p: p.calculate_volatility_of_return())
        #self.measure_execution_time("calculate_largest_exposure", lambda p: p.calculate_largest_exposure())
        ##self.measure_execution_time("calculate_mean_reversion_score", lambda p: p.calculate_mean_reversion_score(exchange))
        #self.measure_execution_time("calculate_position_sizing_score", lambda p: p.calculate_position_sizing_score())
        #self.measure_execution_time("calculate_trade_sequencing_score", lambda p: p.calculate_trade_sequencing_score())
        #self.measure_execution_time("calculate_trading_confidence", lambda p: p.calculate_trading_confidence())

        #self.measure_execution_time("calculate_trade_expectancy", lambda p: p.calculate_trade_expectancy())
        #self.measure_execution_time("calculate_risk_of_ruin", lambda p: p.calculate_risk_of_ruin())
        #self.measure_execution_time("calculate_kelly_criterion", lambda p: p.calculate_kelly_criterion())
        #self.measure_execution_time("calculate_trade_duration", lambda p: p.calculate_trade_duration())
        #self.measure_execution_time("calculate_max_drawdown", lambda p: p.calculate_max_drawdown())
        
        for person in self.people.values():
            """
            
        
            data.append([
                person.id,
                person.get_total_return(exchange),
                person.calculate_momentum_score(exchange),
                person.calculate_average_holding_time(),
                person.calculate_risk_tolerance(),
                person.calculate_intraday_closeout_rate(),
                person.calculate_order_clustering_score(),
                person.calculate_diversification_index(),
                person.calculate_stop_loss_index(exchange),
                person.calculate_win_rate(),                   # 1) Win Rate
                
                person.calculate_profit_factor(),              # 2) Profit Factor
                person.calculate_time_weighted_return(),       # 3) Time-Weighted Return
                person.calculate_sharpe_ratio(),               # 4) Sharpe Ratio
                person.calculate_volatility_of_return(),       # 5) Volatility of Return
                person.calculate_largest_exposure(),           # 6) Largest Exposure
                person.calculate_mean_reversion_score(exchange), # 7) Mean Reversion Score
                person.calculate_position_sizing_score(),      # 8) Position Sizing
                person.calculate_trade_sequencing_score(),     # 9) Trade Sequencing
                person.calculate_trading_confidence(),         # 10) Trading Confidence
                
                person.calculate_trade_expectancy(),
                
                person.calculate_risk_of_ruin(),
                person.calculate_kelly_criterion(),
                person.calculate_trade_duration(),
                person.calculate_max_drawdown()
    
                
            ])
            i = i + 1
            print(i)
            """
            
        return pd.DataFrame(data, columns=[
            'id',
            'return',
            'momentum_score',
            'average_holding_time',
            'risk_tolerance',
            'intraday_closeout_rate',
            'order_clustering_score',
            'asset_diversification_index',
            'stop_loss_index',
            'win_rate',
            
            'profit_factor',
            'time_weighted_return',
            'sharpe_ratio',
            'volatility_of_return',
            'largest_exposure',
            'mean_reversion_score',
            'position_sizing_score',
            'trade_sequencing_score',
            'trading_confidence',
    
            'trade_expectnacy',
            'risk_of_ruin',
            'kelly_criteria',
            'trade_duration',
            'max_drawdown'
        ])
    
    