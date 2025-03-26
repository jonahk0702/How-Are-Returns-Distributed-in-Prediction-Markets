import pandas as pd

def create_feature_df(poeple, stocks, extension_old,extension_new , feauture_id):
    df = pd.read_csv(f"./data/features_{extension_old}.csv")

    def add_feature(column_name, func):
        df[column_name] = [func(person) for person in poeple.people.values()]
        print(f"done {column_name}!")

    if feauture_id == 0:
        df['id'] = [person.id for person in poeple.people.values()]
        print("done id!")

    if feauture_id == 1:
        add_feature("return", lambda p: p.get_total_return(stocks))

    if feauture_id == 2:
        add_feature("momentum", lambda p: p.calculate_momentum_score(stocks))

    if feauture_id == 3:
        add_feature("average_holding_time", lambda p: p.calculate_average_holding_time())

    if feauture_id == 4:
        add_feature("risk_tolerance", lambda p: p.calculate_risk_tolerance())

    if feauture_id == 5:
        add_feature("intraday_closeout_rate", lambda p: p.calculate_intraday_closeout_rate())

    if feauture_id == 6:
        add_feature("diversification_index", lambda p: p.calculate_diversification_index())

    if feauture_id == 7:
        add_feature("stop_loss_index", lambda p: p.calculate_stop_loss_index(stocks))

    if feauture_id == 8:
        add_feature("win_rate", lambda p: p.calculate_win_rate())

    if feauture_id == 9:
        add_feature("profit_factor", lambda p: p.calculate_profit_factor())

    if feauture_id == 10:
        add_feature("time_weighted_return", lambda p: p.calculate_time_weighted_return())

    if feauture_id == 11:
        add_feature("sharpe_ratio", lambda p: p.calculate_sharpe_ratio())

    if feauture_id == 12:
        add_feature("volatility_of_return", lambda p: p.calculate_volatility_of_return())

    if feauture_id == 13:
        add_feature("largest_exposure", lambda p: p.calculate_largest_exposure())

    if feauture_id == 14:
        add_feature("mean_reversion_score", lambda p: p.calculate_mean_reversion_score(stocks))

    if feauture_id == 15:
        add_feature("position_sizing_score", lambda p: p.calculate_position_sizing_score())

    if feauture_id == 16:
        add_feature("trade_sequencing_score", lambda p: p.calculate_trade_sequencing_score())

    if feauture_id == 17:
        add_feature("trading_confidence", lambda p: p.calculate_trading_confidence())

    if feauture_id == 18:
        add_feature("trade_expectancy", lambda p: p.calculate_trade_expectancy())

    if feauture_id == 19:
        add_feature("risk_of_ruin", lambda p: p.calculate_risk_of_ruin())

    if feauture_id == 20:
        add_feature("kelly_criterion", lambda p: p.calculate_kelly_criterion())

    if feauture_id == 21:
        add_feature("trade_duration", lambda p: p.calculate_trade_duration())

    if feauture_id == 22:
        add_feature("max_drawdown", lambda p: p.calculate_max_drawdown())


    df.to_csv(f"./data/features_{extension_new}.csv")
    print("done!")
