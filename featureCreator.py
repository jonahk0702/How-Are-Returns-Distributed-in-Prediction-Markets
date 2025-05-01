import pandas as pd

def create_feature_df(poeple, stocks, extension_old, extension_new, feature_id):
    df = pd.read_csv(f"./data/features_{extension_old}.csv")

    def add_feature(column_name, func):
        df[column_name] = [func(person) for person in poeple.people.values()]
        print(f"done {column_name}!")

    if feature_id == 0:
        df['id'] = [person.id for person in poeple.people.values()]
        print("done id!")

    if feature_id == 1:
        add_feature("return", lambda p: p.get_total_return(stocks))

    if feature_id == 2:
        add_feature("momentum", lambda p: p.calculate_momentum_score(stocks))

    if feature_id == 3:
        add_feature("average_holding_time", lambda p: p.calculate_average_holding_time())

    if feature_id == 4:
        add_feature("risk_tolerance", lambda p: p.calculate_risk_tolerance())

    if feature_id == 5:
        add_feature("intraday_closeout_rate", lambda p: p.calculate_intraday_closeout_rate())

    if feature_id == 6:
        add_feature("diversification_index", lambda p: p.calculate_diversification_index())

    if feature_id == 7:
        add_feature("stop_loss_index", lambda p: p.calculate_stop_loss_index(stocks))

    if feature_id == 8:
        add_feature("win_rate", lambda p: p.calculate_win_rate())

    if feature_id == 9:
        add_feature("profit_factor", lambda p: p.calculate_profit_factor())

    if feature_id == 10:
        add_feature("time_weighted_return", lambda p: p.calculate_time_weighted_return())

    if feature_id == 11:
        add_feature("sharpe_ratio", lambda p: p.calculate_sharpe_ratio())

    if feature_id == 12:
        add_feature("volatility_of_return", lambda p: p.calculate_volatility_of_return())

    if feature_id == 13:
        add_feature("largest_exposure", lambda p: p.calculate_largest_exposure())

    if feature_id == 14:
        add_feature("mean_reversion_score", lambda p: p.calculate_mean_reversion_score(stocks))

    if feature_id == 15:
        add_feature("position_sizing_score", lambda p: p.calculate_position_sizing_score())

    if feature_id == 16:
        add_feature("trade_sequencing_score", lambda p: p.calculate_trade_sequencing_score())

    if feature_id == 17:
        add_feature("trading_confidence", lambda p: p.calculate_trading_confidence())

    if feature_id == 18:
        add_feature("trade_expectancy", lambda p: p.calculate_trade_expectancy())

    if feature_id == 19:
        add_feature("risk_of_ruin", lambda p: p.calculate_risk_of_ruin())

    if feature_id == 20:
        add_feature("kelly_criterion", lambda p: p.calculate_kelly_criterion())

    if feature_id == 21:
        add_feature("trade_duration", lambda p: p.calculate_trade_duration())

    if feature_id == 22:
        add_feature("max_drawdown", lambda p: p.calculate_max_drawdown())

    if feature_id == 23:
        add_feature("realized_return_ratio", lambda p: p.realized_return_ratio())

    if feature_id == 24:
        add_feature("mark_to_market_return", lambda p: p.mark_to_market_return(stocks))

    if feature_id == 25:
        add_feature("per_trade_profitability", lambda p: p.per_trade_profitability())

    if feature_id == 26:
        add_feature("return_on_invested_capital", lambda p: p.return_on_invested_capital(stocks))

    if feature_id == 27:
        add_feature("outcome_adjusted_return", lambda p: p.outcome_adjusted_return())

    if feature_id == 28:
        add_feature("simple_cash_multiple", lambda p: p.calc_simple_cash_multiple(stocks))

    if feature_id == 29:
        add_feature("cost_basis_returns", lambda p: p.calc_cost_basis_returns(stocks))

    if feature_id == 30:
        add_feature("inferred_starting_balance", lambda p: p.infer_starting_balance(stocks))

    if feature_id == 31:
        print(31)
        add_feature("return_consistency", lambda p: p.calculate_return_consistency())

    if feature_id == 32:
        add_feature("return_skewness", lambda p: p.calculate_return_skewness())

    if feature_id == 33:
        add_feature("return_kurtosis", lambda p: p.calculate_return_kurtosis())

    if feature_id == 34:
        add_feature("return_mad", lambda p: p.calculate_return_mad())

    if feature_id == 35:
        add_feature("outlier_ratio", lambda p: p.calculate_outlier_ratio())

    if feature_id == 36:
        add_feature("gini_bet_size", lambda p: p.calculate_gini_bet_size())

    if feature_id == 37:
        add_feature("gini_return_distribution", lambda p: p.calculate_gini_return_distribution())

    if feature_id == 38:
        add_feature("log_return", lambda p: p.calculate_log_return_from_orders(stocks))

    if feature_id == 39:
        add_feature("average_portfolio_risked", lambda p: p.calculate_average_portfolio_risked(stocks))

    if feature_id == 40:
        add_feature("panic_sell_score", lambda p: p.calculate_panic_sell_score(stocks))

    df.to_csv(f"./data/features_{extension_new}.csv", index=False)
    print("done!")


