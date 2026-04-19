CHECKPOINT
- dataset/timeframe : SOLUSDT 1h, 2020-09-14 -> 2025-12-27
- scope : relance complète des 21 stratégies canon (`reports/retained21.txt`)
- benchmark : Buy & Hold net (frais + slippage) = 401,831.06$ (+3918.31%)
- meilleure stratégie des 21 : meanrev_drawdown_bounce = 36,412.21$ (+264.12%), Sharpe 6.690, PF 2.589, MDD 3.07%, trades 394
- constat clé : sur cette période fortement haussière, aucune des 21 stratégies actives ne bat Buy&Hold (écart meilleur cas ≈ -365,418.85$)
- artefacts run : reports/ultra_benchmark_sol/full21-updated-20200914-20251227/
- publication webapp : micro_results_compact.csv + buyhold_ref.txt générés dans ce run pour affichage auto du dernier run
- prochain axe pour améliorer le score vs buy&hold : inclure une stratégie trend longue durée / hold-overlay parmi les 21 (sinon underperform structurelle en bull market)
