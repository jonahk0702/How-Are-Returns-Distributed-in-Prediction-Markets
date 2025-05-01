## Possibly for colab
import numpy as np
import matplotlib.pyplot as plt

# Your metric categories
categories = [
    # RETURNS & PROFITABILITY
    'realized_return_ratio', 
    'calc_simple_cash_multiple', 'calc_cost_basis_returns', 'return_on_invested_capital', 
    'per_trade_profitability', 'calculate_win_rate', 
    'calculate_profit_factor', 'calculate_trade_expectancy', 'calculate_kelly_criterion',
    
    # 📉 RISK & VOLATILITY
    'calculate_volatility_of_return', 'calculate_sharpe_ratio', 'calculate_risk_of_ruin',
    'calculate_max_drawdown', 'calculate_largest_exposure', 'calculate_stop_loss_index',
    'calculate_position_sizing_score', 'calculate_return_consistency', 'calculate_return_skewness',
    'calculate_return_kurtosis', 'calculate_return_mad', 'calculate_outlier_ratio','calculate_average_portfolio_risked',

    # BEHAVIORAL / STYLE-BASED
    'calculate_momentum_score', 'calculate_mean_reversion_score', 'calculate_risk_tolerance',
    'calculate_order_clustering_score', 'calculate_intraday_closeout_rate', 'calculate_trade_sequencing_score',
    'calculate_trading_confidence', 'calculate_diversification_index', 'calculate_gini_bet_size',
    'calculate_gini_return_distribution', 'calculate_news_reaction_metrics', 'calculate_news_alpha',
    'generate_news_reaction_table',

    # TIME & HOLDING-BASED
    'calculate_average_holding_time', 'calculate_trade_duration', 'calculate_relative_exit_efficiency',
    'calculate_panic_sell_score',

    # MARKET-SPECIFIC / RESOLUTION-BASED
    'outcome_adjusted_return', 'calculate_information_gain_from_price',

   
]

# Generate fake data between 0 and 1
values = 0.2 + (0.8 - 0.2) * np.random.rand(len(categories))


# Radar charts must be circular, so we repeat the first value at the end
values = np.append(values, values[0])

# Set the angle for each axis
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

# Create the figure
fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))

# Draw the radar chart
ax.plot(angles, values, linewidth=2, linestyle='solid')
ax.fill(angles, values, alpha=0.25)

# Add category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=7)

# Rotate labels to avoid clutter
for label, angle in zip(ax.get_xticklabels(), angles):
    if angle in [0, np.pi]:
        label.set_horizontalalignment('center')
    elif 0 < angle < np.pi:
        label.set_horizontalalignment('left')
        label.set_rotation(angle * 180/np.pi - 90)
    else:
        label.set_horizontalalignment('right')
        label.set_rotation(angle * 180/np.pi + 90)

# Set a title
plt.title('Spider Chart: Trading Metrics', size=18, y=1.1)

# Adjust layout and display
plt.tight_layout()
plt.show()

